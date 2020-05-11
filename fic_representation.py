""" Superclass for any extracted elements from a fic. """

import os
import pickle
import pandas as pd

class FicRepresentation():
    
    def __init__(self, fandom_fname, fic_csv_dirpath=None):

        self.fandom_fname = fandom_fname
        if fic_csv_dirpath is not None:
            self.fic_csvpath = os.path.join(fic_csv_dirpath, f'{fandom_fname}.csv')
        self.fic_csv = None

    def load_fic_csv(self):
        self.fic_csv = pd.read_csv(self.fic_csvpath)

    def save_quotes(self, save_dirpath):
        """ Save pickle of the quotes in the dirpath """
        if not os.path.exists(save_dirpath):
            os.makedirs(save_dirpath)
        save_fpath = os.path.join(save_dirpath, f'{self.fandom_fname}.pkl')
        with open(save_fpath, 'wb') as f: 
            pickle.dump(self.quotes, f)
