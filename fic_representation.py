""" Superclass for any extracted elements from a fic. """

import os
import pickle
import pandas as pd

class FicRepresentation():
    
    def __init__(self, fandom_fname, fic_csv_dirpath=None):

        self.fandom_fname = fandom_fname
        self.fic_csv_dirpath = fic_csv_dirpath
        if fic_csv_dirpath is not None:
            self.fic_csvpath = os.path.join(fic_csv_dirpath, f'{fandom_fname}.csv')
        self.fic_csv = None

    def load_fic_csv(self):
        self.fic_csv = pd.read_csv(self.fic_csvpath)

    def pickle_output(self, save_dirpath, struct):
        """ Pickle a structure to a file in the save_dirpath """
        outpath = os.path.join(save_dirpath, f'{self.fandom_fname}.pkl')
        with open(os.path.join(outpath), 'wb') as f:
                pickle.dump(struct, f)
