from utils import *
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
import matplotlib.colors as colors

class Miner():
    def __init__(self, config):
        self.config = config
        self.dataset_path = config.dataset_path

    def load_for_modelling(self):
        self.dataset = pd.read_pickle(self.dataset_path)
        self.dataset_keys = list(self.dataset.columns)
        #self.filter_dataset() # option to filter the dataset
        self.datasetPath = 'datasets/dataset'
        self.dataset.to_pickle(self.datasetPath)
        del (self.dataset)



    def filter_dataset(self):
        print('Filtering dataset starting from {} samples'.format(len(self.dataset)))
        ## filtering out unwanted characteristics
        bad_inds = []

        n_bad_inds = len(bad_inds)
        bad_inds.extend(np.argwhere('SOMETHING IS BAD')[:, 0]) # TODO option to pre-filter your dataset
        print('some filter caught {} samples'.format(int(len(bad_inds) - n_bad_inds)))

        # collate bad indices
        bad_inds = np.unique(bad_inds)

        # apply filtering
        self.dataset = delete_from_dataframe(self.dataset, bad_inds)
        print("Filtering removed {} samples, leaving {}".format(len(bad_inds), len(self.dataset)))
