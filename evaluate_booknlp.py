#!/usr/bin/env python
# coding: utf-8

# # Post-process BookNLP output to get entity mention clusters

import os
import sys
import pandas as pd
import numpy as np
import csv
import pdb
import itertools
import pickle
from configparser import ConfigParser
import argparse

from evaluator import Evaluator
from annotation import QuoteAnnotation
from booknlp_output import BookNLPOutput
import evaluation_utils as utils


class BookNLPEvaluator(Evaluator):
    """ Evaluate BookNLP coreference and quote attribution.
        The evaluator handles the settings for multiple fics. 
    """

    def __init__(self, token_output_dirpath, json_output_dirpath, fic_csv_dirpath, 
                    evaluate_coref=False, evaluate_quotes=False,
                    coref_annotations_dirpath=None,
                    quote_annotations_dirpath=None,
                    predicted_coref_outpath=None,
                    predicted_quotes_outpath=None,
                    token_file_ext='.tokens'):

        super().__init__(fic_csv_dirpath,
                    evaluate_coref, evaluate_quotes,
                    coref_annotations_dirpath,
                    quote_annotations_dirpath,
                    predicted_coref_outpath,
                    predicted_quotes_outpath)

        self.token_output_dirpath = token_output_dirpath
        self.json_output_dirpath = json_output_dirpath
        self.token_file_ext = token_file_ext

    def evaluate(self):
        for fname in sorted(os.listdir(self.token_output_dirpath)):
            fandom_fname = fname.split('.')[0]
        
            print(fandom_fname)
            sys.stdout.flush()

            self.evaluate_fic(fandom_fname)

    def evaluate_fic(self, fandom_fname):

        print("\tLoading file...")
        sys.stdout.flush()
        # Load output, CSV file of fic
        booknlp_output = BookNLPOutput(self.token_output_dirpath, self.json_output_dirpath, fandom_fname, fic_csv_dirpath=self.fic_csv_dirpath, token_file_ext=self.token_file_ext)

        # Whatever you do, need to align with annotations
        print("\tAligning with annotated fic...")
        sys.stdout.flush()
        booknlp_output.align_with_annotations()

        if self.whether_evaluate_coref:
            self.evaluate_coref(fandom_fname, booknlp_output)

        if self.whether_evaluate_quotes:
            self.evaluate_quotes(fandom_fname, booknlp_output, exact_match=False)


def main():

    parser = argparse.ArgumentParser(description='Evaluate BookNLP on coref and quote attribution')
    parser.add_argument('config_fpath', nargs='?', help='File path of config file')
    args = parser.parse_args()

    config = ConfigParser(allow_no_value=False)
    config.read(args.config_fpath)

    token_output_dirpath = str(config.get('Filepaths', 'token_output_dirpath'))
    json_output_dirpath = str(config.get('Filepaths', 'json_output_dirpath'))
    fic_csv_dirpath = str(config.get('Filepaths', 'fic_csv_dirpath'))

    evaluator = BookNLPEvaluator(token_output_dirpath, json_output_dirpath, fic_csv_dirpath,
        evaluate_coref=config.getboolean('Settings', 'evaluate_coref'), 
        evaluate_quotes=config.getboolean('Settings', 'evaluate_quotes'), 
        coref_annotations_dirpath = str(config.get('Filepaths', 'coref_annotations_dirpath')),
        quote_annotations_dirpath = str(config.get('Filepaths', 'quote_annotations_dirpath')),
        predicted_coref_outpath = str(config.get('Filepaths', 'predicted_coref_outpath')),
        predicted_quotes_outpath = str(config.get('Filepaths', 'predicted_quotes_outpath')),
        token_file_ext = config.get('Settings', 'token_file_ext')
        )

    evaluator.evaluate()


if __name__ == '__main__':
    main()
