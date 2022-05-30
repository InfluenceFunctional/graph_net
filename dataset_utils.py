import torch
import numpy as np
from utils import standardize
from torch_geometric.data import Data
import sys
from torch_geometric.loader import DataLoader
import tqdm
import time
import matplotlib.pyplot as plt
import pandas as pd


class BuildDataset:
    """
    build dataset object
    """

    def __init__(self, config):
        self.max_atomic_number = 20
        self.atom_dict_size = {'atom z': self.max_atomic_number + 1}  # for embeddings
        self.dataset_seed = config.dataset_seed

        dataset = pd.read_pickle('datasets/dataset')  # dataset_manager will put your data here
        self.dataset_length = len(dataset)
        self.final_dataset_length = min(self.dataset_length, config.dataset_length)

        self.atom_keys = ['xxx']  # put the dict keys here which you want to add
        self.molecule_keys = ['yyy']  # ditto
        self.target = ['target']  # what we want to train on

        molecule_features_array = self.concatenate_molecule_features(dataset)  # todo add your custom molecule features
        targets = self.get_targets(dataset)
        atom_features_list = self.concatenate_atom_features(dataset)  # todo add your custom atom-wise features
        self.datapoints = self.generate_training_data(dataset['atom coords'], atom_features_list,
                                                      molecule_features_array, targets, dataset['crystal reference cell coords'])

        self.shuffle_datapoints()

    def shuffle_datapoints(self):
        np.random.seed(self.dataset_seed)
        good_inds = np.random.choice(self.final_dataset_length, size=self.final_dataset_length, replace=False)
        self.dataset_length = len(good_inds)

        self.datapoints = [self.datapoints[i] for i in good_inds]

    def shuffle_final_dataset(self, atom_features_list, molecule_features_array, targets, smiles, coords, tracking_features):
        np.random.seed(self.dataset_seed)
        good_inds = np.random.choice(self.dataset_length, size=self.final_dataset_length, replace=False)
        self.dataset_length = len(good_inds)
        return [atom_features_list[i] for i in good_inds], molecule_features_array[good_inds], targets[good_inds], \
               [smiles[i] for i in good_inds], [coords[i] for i in good_inds], tracking_features[good_inds]

    def generate_training_data(self, atom_coords, atom_features_list, mol_features, targets):
        '''
        convert feature, target and tracking vectors into torch.geometric data objects
        :param atom_coords:
        :param smiles:
        :param atom_features_list:
        :param mol_features:
        :param targets:
        :param tracking_features:
        :return:
        '''
        datapoints = []

        print("Generating final training datapoints")
        for i in tqdm.tqdm(range(self.final_dataset_length)):
            if targets[i].ndim == 1:
                target = torch.tensor(targets[i][np.newaxis, :])
            else:
                target = torch.tensor(targets[i])

            # append molecule features to atom features for each atom
            input_features = np.concatenate((atom_features_list[i], np.repeat(mol_features[i][np.newaxis, :], len(atom_features_list[i]), axis=0)), axis=1)

            input_features = torch.Tensor(input_features)
            assert torch.sum(torch.isnan(input_features)) == 0, "NaN in training input"  # ensure no NaN generated in training setup
            datapoints.append(Data(x=input_features.float(), pos=torch.Tensor(atom_coords[i]), y=target))

        return datapoints

    def concatenate_atom_features(self, dataset):
        """
        collect and normalize/standardize relevant atomic features
        :param dataset:
        :return:
        """

        keys_to_add = self.atom_keys
        if self.target in keys_to_add:  # don't add atom target if we are going to model it
            keys_to_add.remove(self.target)
        print("Preparing atom-wise features")
        atom_features_list = [np.zeros((len(dataset['atom Z'][i]), len(keys_to_add))) for i in range(self.dataset_length)]
        stds, means = {}, {}
        for column, key in enumerate(keys_to_add):
            flat_feature = np.concatenate(dataset[key])
            stds[key] = np.std(flat_feature)
            means[key] = np.mean(flat_feature)
            for i in range(self.dataset_length):
                feature_vector = dataset[key][i]

                if type(feature_vector) is not np.ndarray:
                    feature_vector = np.asarray(feature_vector)

                if key == 'atom Z':
                    pass
                elif feature_vector.dtype == bool:
                    pass
                elif (feature_vector.dtype == float) or (np.issubdtype(feature_vector.dtype, np.floating)):
                    feature_vector = standardize(feature_vector, known_std=stds[key], known_mean=means[key])
                elif (feature_vector.dtype == int) or (np.issubdtype(feature_vector.dtype, np.integer)):
                    if len(np.unique(feature_vector)) > 2:
                        feature_vector = standardize(feature_vector, known_std=stds[key], known_mean=means[key])
                    else:
                        feature_vector = np.asarray(feature_vector == np.amax(feature_vector))  # turn it into a bool

                atom_features_list[i][:, column] = feature_vector

        return atom_features_list

    def concatenate_molecule_features(self, dataset):
        """
        collect features of 'molecules' and append to atom-level data
        """
        # normalize everything
        keys_to_add = []
        keys_to_add.extend(self.molecule_keys)

        print("Preparing molcule-wise features")

        molecule_feature_array = np.zeros((self.dataset_length, len(keys_to_add)), dtype=float)
        for column, key in enumerate(keys_to_add):
            feature_vector = dataset[key]
            if type(feature_vector) is not np.ndarray:
                feature_vector = np.asarray(feature_vector)

            if feature_vector.dtype == bool:
                pass
            elif (feature_vector.dtype == float) or (np.issubdtype(feature_vector.dtype, np.floating)):
                feature_vector = standardize(feature_vector)
            elif (feature_vector.dtype == int) or (np.issubdtype(feature_vector.dtype, np.integer)):
                if len(np.unique(feature_vector)) > 2:
                    feature_vector = standardize(feature_vector)
                else:
                    feature_vector = np.asarray(feature_vector == np.amax(feature_vector))  # turn it into a bool

            molecule_feature_array[:, column] = feature_vector

        self.n_mol_features = len(keys_to_add)
        # store known info for training analysis
        self.mol_dict_keys = keys_to_add

        return molecule_feature_array

    def get_targets(self, dataset):
        """
        get training target classes
        maybe do some statistics
        etc.
        """
        print("Preparing training targets")
        target_features = dataset[self.target] # TODO do any binarization/standardization or preprocessing here

        return target_features

    def get_dimension(self):
        '''
        dimensions relevant to building and training the model
        :return:
        '''

        dim = {
            'atom features': self.datapoints[0].x.shape[1],
            'mol features': self.n_mol_features,
            'output classes': [2],
            'dataset length': len(self.datapoints),
            'atom embedding dict sizes': self.atom_dict_size,
        }

        return dim

    def __getitem__(self, idx):
        return self.datapoints[idx]

    def __len__(self):
        return len(self.datapoints)


def get_dataloaders(dataset_builder, config, override_batch_size=None):
    '''
    need special torch geometric dataloaders for this rather than standard pytorch
    :param dataset_builder:
    :param config:
    :param override_batch_size:
    :return:
    '''
    if override_batch_size is not None:
        batch_size = override_batch_size
    else:
        batch_size = config.initial_batch_size
    train_size = int(0.8 * len(dataset_builder))  # split data into training and test sets
    test_size = len(dataset_builder) - train_size

    train_dataset = []
    test_dataset = []

    for i in range(test_size, test_size + train_size):
        train_dataset.append(dataset_builder[i])
    for i in range(test_size):
        test_dataset.append(dataset_builder[i])

    tr = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    te = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    return tr, te


def delete_from_dataset(dataset, good_inds):
    print("Deleting unwanted entries")

    for key in dataset.keys():
        if type(dataset[key]) == list:
            dataset[key] = [dataset[key][i] for i in good_inds]
        elif type(dataset[key]) == np.ndarray:
            dataset[key] = dataset[key][np.asarray(good_inds)]

    return dataset
