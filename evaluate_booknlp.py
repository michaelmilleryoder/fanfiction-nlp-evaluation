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

from booknlp_output import BookNLPOutput
import evaluation_utils as utils

class BookNLPEvaluator():
    """ Evaluate BookNLP coreference and quote attribution """

    def __init__(self, predictions_dirpath, fic_csv_dirpath, 
                    evaluate_coref=False, evaluate_quotes=False,
                    coref_annotations_dirpath=None,
                    quote_annotations_dirpath=None,
                    predicted_entities_outpath=None,
                    predicted_quotes_outpath=None):

        self.predictions_dirpath = predictions_dirpath
        self.fic_csv_dirpath = fic_csv_dirpath
        self.evaluate_coref = evaluate_coref
        self.evaluate_quotes = evaluate_quotes
        self.coref_annotations_dirpath = coref_annotations_dirpath
        self.quote_annotations_dirpath = quote_annotations_dirpath
        self.predicted_entities_outpath = predicted_entities_outpath
        self.predicted_quotes_outpath = predicted_quotes_outpath

    def evaluate(self):
        for fname in sorted(os.listdir(self.predictions_dirpath)):
        
            print(fname)
            sys.stdout.flush()

            self.evaluate_fic(fname)

    def evaluate_fic(self, fname):

        fandom_fname = fname.split('.')[0]

        print("\tLoading file...")
        sys.stdout.flush()
        # Load output, CSV file of fic
        booknlp_output = BookNLPOutput(self.predictions_dirpath, fname, fic_csvpath=os.path.join(self.csv_dirpath, fname))

        # Whatever you do, need to align with annotations
        print("\tAligning with annotated fic...")
        sys.stdout.flush()
        booknlp_output.align_with_annotations()

        if self.evaluate_coref:
            # TODO: fill in with coref
            pass

        if self.evaluate_quotes:
            self.evaluate_quotes()

    def evaluate_quotes(self, fandom_fname):
        
        # Quote extraction evaluation
        # Load gold quote spans
        gold_quotes = utils.gold_quotes(self.quote_annotations_dirpath, fandom_fname)

        # Load predicted quote spans (from BookNLP output to Quote objects)
        predicted_quotes = booknlp_output.extract_quotes()

        # Quote attribution evaluation


def main():

    # I/O
    predictions_dirpath = '/projects/book-nlp/data/tokens/annotated_10fandom_dev/'
    fic_csv_dirpath = '/data/fanfiction_ao3/annotated_10fandom/dev/fics/'

    evaluator = BookNLPEvaluator(predictions_dirpath, fic_csv_dirpath,
        evaluate_coref=False, 
        evaluate_quotes=True, 
        coref_annotations_dirpath='/data/fanfiction_ao3/annotated_10fandom/dev/entity_clusters',
        predicted_entities_outpath = '/projects/book-nlp/tmp/predicted_entities/',
        predicted_quotes_outpath = '/projects/book-nlp/tmp/predicted_entities/',
        )   

    evaluator.evaluate()


if __name__ == '__main__':
    main()
