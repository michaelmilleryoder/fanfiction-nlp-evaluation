#!/usr/bin/env python
# coding: utf-8

""" Evaluate fanfiction pipeline for coreference and quote attribution. """

import os
import re
import sys
import pandas as pd
import numpy as np
import csv
import pdb
import itertools
import pickle
from collections import defaultdict
import argparse
from configparser import ConfigParser

from evaluator import Evaluator
from annotation import Annotation
import evaluation_utils as utils
from pipeline_output import PipelineOutput
import scorer


class PipelineEvaluator(Evaluator):
    """ Evaluate FanfictionNLP pipeline coreference and quote attribution.
        The evaluator handles the settings for multiple fics. 
    """

    def __init__(self, output_dirpath, fic_csv_dirpath, 
                    evaluate_coref=False, evaluate_quotes=False,
                    coref_annotations_dirpath=None,
                    quote_annotations_dirpath=None,
                    predicted_coref_outpath=None,
                    predicted_quotes_outpath=None):

        super().__init__(fic_csv_dirpath,
                    evaluate_coref, evaluate_quotes,
                    coref_annotations_dirpath,
                    quote_annotations_dirpath,
                    predicted_coref_outpath,
                    predicted_quotes_outpath)

        self.output_dirpath = output_dirpath
        self.char_output_dirpath = os.path.join(output_dirpath, 'char_coref_chars')

    def evaluate(self):
        for fname in sorted(os.listdir(self.char_output_dirpath)):
            fandom_fname = fname.split('.')[0]
        
            print(fandom_fname)
            sys.stdout.flush()

            self.evaluate_fic(fandom_fname)

    def evaluate_fic(self, fandom_fname):

        print("\tLoading file...")
        sys.stdout.flush()

        # Load output, CSV file of fic
        pipeline_output = PipelineOutput(self.output_dirpath, fandom_fname)

        if self.whether_evaluate_coref:
            self.evaluate_coref(fandom_fname, pipeline_output)

        if self.whether_evaluate_quotes:
            self.evaluate_quotes(fandom_fname, pipeline_output)


def main():

    parser = argparse.ArgumentParser(description='Evaluate FanfictionNLP pipeline on coref and quote attribution')
    parser.add_argument('config_fpath', nargs='?', help='File path of config file')
    args = parser.parse_args()

    config = ConfigParser(allow_no_value=False)
    config.read(args.config_fpath)

    fic_csv_dirpath = config.get('Filepaths', 'fic_csv_dirpath')
    output_dirpath = config.get('Filepaths', 'output_dirpath')

    evaluator = PipelineEvaluator(output_dirpath, fic_csv_dirpath,
        evaluate_coref=config.getboolean('Settings', 'evaluate_coref'), 
        evaluate_quotes=config.getboolean('Settings', 'evaluate_quotes'), 
        coref_annotations_dirpath = config.get('Filepaths', 'coref_annotations_dirpath'),
        quote_annotations_dirpath = config.get('Filepaths', 'quote_annotations_dirpath'),
        predicted_coref_outpath = config.get('Filepaths', 'predicted_coref_outpath'),
        predicted_quotes_outpath = config.get('Filepaths', 'predicted_quotes_outpath'),
        )

    evaluator.evaluate()


if __name__ == '__main__':
    main()
