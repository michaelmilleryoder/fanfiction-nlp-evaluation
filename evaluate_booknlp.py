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
                    predicted_entities_outpath=None,
                    predicted_quotes_outpath=None):

        self.token_output_dirpath = token_output_dirpath
        self.json_output_dirpath = json_output_dirpath
        self.fic_csv_dirpath = fic_csv_dirpath
        self.whether_evaluate_coref = evaluate_coref
        self.whether_evaluate_quotes = evaluate_quotes
        self.coref_annotations_dirpath = coref_annotations_dirpath
        self.quote_annotations_dirpath = quote_annotations_dirpath
        self.predicted_entities_outpath = predicted_entities_outpath
        self.predicted_quotes_outpath = predicted_quotes_outpath

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
        booknlp_output = BookNLPOutput(self.token_output_dirpath, self.json_output_dirpath, fandom_fname, fic_csv_dirpath=self.fic_csv_dirpath)

        # Whatever you do, need to align with annotations
        print("\tAligning with annotated fic...")
        sys.stdout.flush()
        booknlp_output.align_with_annotations()

        if self.whether_evaluate_coref:
            # TODO: fill in with coref
            pass

        if self.whether_evaluate_quotes:
            self.evaluate_quotes(fandom_fname, booknlp_output)

    def evaluate_quotes(self, fandom_fname, booknlp_output):
        
        # Quote extraction evaluation
        # Load gold quote spans
        gold = QuoteAnnotation(self.quote_annotations_dirpath, fandom_fname, fic_csv_dirpath=self.fic_csv_dirpath)

        # Load predicted quote spans (from BookNLP output to Quote objects)
        booknlp_output.extract_quotes()

        # Print scores
        utils.print_quote_scores(booknlp_output.quotes, gold.quotes, exact_match=False)


def main():

    # I/O
    # TODO: make argparse command line args
    token_output_dirpath = '/projects/book-nlp/data/tokens/annotated_10fandom_dev/'
    json_output_dirpath = '/projects/book-nlp/data/output/annotated_10fandom_dev/'
    fic_csv_dirpath = '/data/fanfiction_ao3/annotated_10fandom/dev/fics/'

    evaluator = BookNLPEvaluator(token_output_dirpath, json_output_dirpath, fic_csv_dirpath,
        evaluate_coref=False, 
        evaluate_quotes=True, 
        coref_annotations_dirpath='/data/fanfiction_ao3/annotated_10fandom/dev/entity_clusters',
        quote_annotations_dirpath='/data/fanfiction_ao3/annotated_10fandom/dev/quote_attribution',
        predicted_entities_outpath = '/projects/book-nlp/tmp/predicted_entities/',
        predicted_quotes_outpath = '/projects/book-nlp/tmp/predicted_entities/'
        )   

    evaluator.evaluate()


if __name__ == '__main__':
    main()
