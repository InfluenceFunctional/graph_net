import wandb
from utils import *
import glob
from model_utils import *
from dataset_manager import Miner
from torch import (backends, optim)
import torch
from dataset_utils import (BuildDataset, get_dataloaders)
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from torch_models import  *


class Predictor():
    def __init__(self, config):
        self.config = config
        self.setup()

    def setup(self):
        '''
        setup working directory
        move to relevant directory
        :return:
        '''

        if self.config.device == 'cuda':
            backends.cudnn.benchmark = True  # auto-optimizes certain backend processes

        miner = Miner(config=self.config, dataset_path=self.config.dataset_path, collect_chunks=False) # TODO load your dataset here, it will save to the workdir

        if not self.config.skip_run_init:
            if (self.config.run_num == 0) or (self.config.explicit_run_enumeration == True):  # if making a new workdir
                if self.config.run_num == 0:
                    self.makeNewWorkingDirectory()
                else:
                    self.workDir = self.config.workdir + '/run%d' % self.config.run_num  # explicitly enumerate the new run directory
                    os.mkdir(self.workDir)

                os.mkdir(self.workDir + '/ckpts')
                os.mkdir(self.workDir + '/datasets')
                os.chdir(self.workDir)  # move to working dir
                print('Starting Fresh Run %d' % self.config.run_num)
                t0 = time.time()
                miner.load_for_modelling()
                print('Initializing dataset took {} seconds'.format(int(time.time() - t0)))
        else:
            if self.config.explicit_run_enumeration:
                # move to working dir
                self.workDir = self.config.workdir + '/' + 'run%d' % self.config.run_num
                os.chdir(self.workDir)
                self.class_labels = list(np.load('group_labels.npy', allow_pickle=True))
                print('Resuming run %d' % self.config.run_num)
            else:
                print("Must provide a run_num if not creating a new workdir!")

    def makeNewWorkingDirectory(self):  # make working directory
        '''
        make a new working directory
        non-overlapping previous entries
        :return:
        '''
        workdirs = glob.glob(self.config.workdir + '/' + 'run*')  # check for prior working directories
        if len(workdirs) > 0:
            prev_runs = []
            for i in range(len(workdirs)):
                prev_runs.append(int(workdirs[i].split('run')[-1]))

            prev_max = max(prev_runs)
            self.workDir = self.config.workdir + '/' + 'run%d' % (prev_max + 1)
            self.config.workdir = self.workDir
            os.mkdir(self.workDir)
            self.config.run_num = int(prev_max + 1)
        else:
            self.workDir = self.config.workdir + '/' + 'run1'
            self.config.run_num = 1
            os.mkdir(self.workDir)

    def prep_metrics(self):
        metrics_list = ['train loss', 'test loss', 'epoch', 'learning rate']
        metrics_dict = initialize_metrics_dict(metrics_list)

        return metrics_dict

    def update_metrics(self, epoch, metrics_dict, err_tr, err_te, lr):
        metrics_dict['train loss'].append(torch.mean(torch.stack(err_tr)).cpu().detach().numpy())
        metrics_dict['test loss'].append(torch.mean(torch.stack(err_te)).cpu().detach().numpy())
        metrics_dict['epoch'].append(epoch)
        metrics_dict['learning rate'].append(lr)

        return metrics_dict

    def init_model(self, config, dataDims, print_status=True):
        '''
        Initialize model and optimizer
        :return:
        '''
        # init model
        model = molecule_graph_model(config, dataDims)
        if config.device == 'cuda':
            model = model.cuda()

        # init optimizers
        amsgrad = False
        beta1 = config.beta1  # 0.9
        beta2 = config.beta2  # 0.999
        weight_decay = config.weight_decay  # 0.01
        momentum = 0

        if config.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), amsgrad=amsgrad, lr=config.learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
        elif config.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), amsgrad=amsgrad, lr=config.learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
        elif config.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            print(config.optimizer + ' is not a valid optimizer')
            sys.exit()

        # init learning rate schedulers (all optional)
        scheduler1 = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=15,
            threshold=1e-4,
            threshold_mode='rel',
            cooldown=15
        )
        lr_lambda = lambda epoch: 1.25
        scheduler3 = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda)
        lr_lambda2 = lambda epoch: 0.95
        scheduler4 = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda2)

        nconfig = get_n_config(model)
        if print_status:
            print('Proxy model has {:.3f} million or {} parameters'.format(nconfig / 1e6, int(nconfig)))

        return model, optimizer, [scheduler1, scheduler3, scheduler4], nconfig

    def get_batch_size(self, dataset, config):
        '''
        automatically search for the maximum batch size we can fit on the device
        then take some fraction of that max as the actual batch size
        optional
        :param dataset:
        :param config:
        :return:
        '''
        finished = 0
        batch_size = config.initial_batch_size.real
        batch_reduction_factor = config.auto_batch_reduction

        model, optimizer, schedulers, n_params = self.init_model(config, config.dataDims, print_status=False)

        while finished == 0:
            if config.device.lower() == 'cuda':
                torch.cuda.empty_cache()  # clear GPU cache

            if config.add_spherical_basis is False:  # initializing spherical basis is too expensive to do repetitively
                model, optimizer, schedulers, n_params = self.init_model(config, config.dataDims, print_status=False)  # for some reason necessary for memory reasons

            try:
                tr, te = get_dataloaders(dataset, config, override_batch_size=batch_size)
                self.model_epoch(config, dataLoader=tr, model=model, optimizer=optimizer, update_gradients=True, iteration_override=2)  # train & compute loss

                finished = 1

                if batch_size < 10:
                    leeway = batch_reduction_factor / 2
                elif batch_size > 20:
                    leeway = batch_reduction_factor
                else:
                    leeway = batch_reduction_factor / 1.33

                batch_size = max(1, int(batch_size * leeway))  # give a margin for molecule sizes - larger margin for smaller batch sizes

                print('Final batch size is {}'.format(batch_size))

                tr, te = get_dataloaders(dataset, config, override_batch_size=batch_size)

                if config.device.lower() == 'cuda':
                    torch.cuda.empty_cache()  # clear GPU cache

                return tr, te, batch_size
            except:  # MemoryError or RuntimeError:
                batch_size = int(batch_size * 0.95)
                print('Training batch size reduced to {}'.format(batch_size))
                if batch_size <= 2:
                    print('Model is too big! (or otherwise broken)')
                    if config.device.lower() == 'cuda':
                        torch.cuda.empty_cache()  # clear GPU cache

                    # do another one for debugging purposes, to throw the explicit error
                    tr, te = get_dataloaders(dataset, config, override_batch_size=batch_size)
                    self.model_epoch(config, dataLoader=tr, model=model, optimizer=optimizer, update_gradients=True, iteration_override=2)  # train & compute loss

                    sys.exit()

    def train(self):
        '''
        config turns into a wandb config here - which sometimes causes problems. best not to edit it much after this if possible
        :return:
        '''
        with wandb.init(config=self.config, project=self.config.project_name, entity=self.config.wandb_username, tags=self.config.experiment_tag):
            config = wandb.config
            print(config)

            # prep dataset
            dataset_builder = BuildDataset(config)
            config.dataDims = dataset_builder.get_dimension()
            self.dataDims = dataset_builder.get_dimension()

            # get batch size
            if config.auto_batch_sizing:
                print('Finding optimal batch size')
                train_loader, test_loader, config.final_batch_size = self.get_batch_size(dataset_builder, config)
            else:
                print('Getting dataloaders for pre-determined batch size')
                train_loader, test_loader = get_dataloaders(dataset_builder, config)
                config.final_batch_size = config.initial_batch_size

            print("Training batch size set to {}".format(config.final_batch_size))

            # initialize model, optimizer, schedulers
            print('Reinitializing model and optimizer')
            model, optimizer, schedulers, n_params = self.init_model(config, self.dataDims)

            # cuda
            if config.device.lower() == 'cuda':
                print('Putting model on CUDA')
                torch.backends.cudnn.benchmark = True
                # model = torch.nn.DataParallel(model) # send to multiple GPUs - not always playing nice with wandb
                model.cuda()

            wandb.watch(model, log_graph=True)

            wandb.log({"Model Num Parameters": n_params,
                       "Final Batch Size": config.final_batch_size})

            metrics_dict = self.prep_metrics()

            # training loop
            hit_max_lr, converged, epoch = False, False, 0
            if config.anomaly_detection:
                torch.autograd.set_detect_anomaly = True
                while (epoch < config.max_epochs) and not converged:
                    '''
                    run the model until it converges or until we hit the max epoch count
                    '''
                    print("  .--.      .-'.      .--.      .--.      .--.      .--.      .`-.      .--.")
                    print(":::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.\::::::::.")
                    print("'      `--'      `.-'      `--'      `--'      `--'      `-.'      `--'      `")
                    print("Starting Epoch {}".format(epoch))  # index epochs from 0

                    err_tr, tr_record, time_train = \
                        self.model_epoch(config, dataLoader=train_loader, model=model,
                                         optimizer=optimizer, update_gradients=True)  # compute train loss and update weights

                    err_te, te_record, epoch_stats_dict, time_test = \
                        self.model_epoch(config, dataLoader=test_loader, model=model,
                                         update_gradients=False, record_stats=True)  # compute loss and accuracy metrics on test set

                    print('epoch={}; nll_tr={:.5f}; nll_te={:.5f}; time_tr={:.1f}s; time_te={:.1f}s'.format(epoch, torch.mean(torch.stack(err_tr)), torch.mean(torch.stack(err_te)), time_train, time_test))

                    # update learning rate
                    optimizer = set_lr(schedulers, optimizer, config, err_tr, hit_max_lr)
                    learning_rate = optimizer.param_groups[0]['lr']
                    if learning_rate >= config.max_lr: hit_max_lr = True

                    # logging
                    self.update_metrics(epoch, metrics_dict, err_tr, err_te, learning_rate)
                    if epoch % config.sample_reporting_frequency == 0:
                        # todo this is where you would report accuracy metrics to wandb
                        pass


                    # check for convergence
                    if checkConvergence(config, metrics_dict['test loss']) and (epoch > config.history + 2):
                        config.finished = True
                        # todo this is where you would report accuracy metrics to wandb
                        break

                    epoch += 1

                if config.device.lower() == 'cuda':
                    torch.cuda.empty_cache()  # clear GPU


    def model_epoch(self, config, dataLoader=None, model=None, optimizer=None, update_gradients=True,
                    iteration_override=None, record_stats=False):
        t0 = time.time()
        if update_gradients:
            model.train(True)
        else:
            model.eval()

        err = []
        loss_record = []
        epoch_stats_dict = {
            'predictions': [],
            'targets': [],
            'tracking features': [],
        }

        for i, data in enumerate(dataLoader):
            if config.device.lower() == 'cuda':
                data = data.cuda()

            if config.test_mode or config.anomaly_detection:
                '''
                check the inputs are actually numbers
                '''
                assert torch.sum(torch.isnan(data.x)) == 0, "NaN in training input"

            losses, predictions = self.get_loss(model, data, config)

            loss = losses.mean()
            err.append(loss.data.cpu())  # average loss
            loss_record.extend(losses.cpu().detach().numpy())  # loss distribution

            if record_stats:
                epoch_stats_dict['predictions'].extend(predictions)
                epoch_stats_dict['targets'].extend(data.y[0].cpu().detach().numpy())

            if update_gradients:
                optimizer.zero_grad()  # reset gradients from previous passes
                loss.backward()  # back-propagation
                optimizer.step()  # update parameters

            if iteration_override is not None:
                if i >= iteration_override:
                    break  # stop training early - for debugging purposes

        total_time = time.time() - t0
        if record_stats:
            return err, loss_record, epoch_stats_dict, total_time
        else:
            return err, loss_record, total_time

    def get_loss(self, model, data, config):
        output = model(data) # raw model output
        targets = data.y[0] # training targets

        if targets.ndim > 1: # fix the dimension
            targets = targets[:, 0]

        losses = F.cross_entropy(output, targets.long(), reduction='none') # TODO custom loss function for your particular problem
        probs = F.softmax(output, dim=1).cpu().detach().numpy()

        return losses, probs
