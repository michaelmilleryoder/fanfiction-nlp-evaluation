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
from collections import defaultdict

from evaluator import Evaluator
from annotation import Annotation
import evaluation_utils as utils
from pipeline_output import PipelineOutput
from pipeline_wrapper import PipelineWrapper
import scorer


class PipelineEvaluator(Evaluator):
    """ Evaluate FanfictionNLP pipeline coreference and quote attribution.
        The evaluator handles the settings for multiple fics. 
    """

    def __init__(self, output_dirpath, fic_csv_dirpath, 
                    evaluate_coref=False, evaluate_quotes=False,
                    coref_from='pipeline', quotes_from='pipeline',
                    run_quote_attribution=False,
                    coref_annotations_dirpath=None,
                    quote_annotations_dirpath=None,
                    dataset_name=None,
                    scores_outpath=None,
                    predicted_coref_outpath=None,
                    predicted_quotes_outpath=None):

        super().__init__(fic_csv_dirpath,
                    evaluate_coref=evaluate_coref, evaluate_quotes=evaluate_quotes,
                    coref_annotations_dirpath = coref_annotations_dirpath,
                    quote_annotations_dirpath = quote_annotations_dirpath,
                    predicted_coref_outpath = predicted_coref_outpath,
                    predicted_quotes_outpath = predicted_quotes_outpath,
                    coref_from=coref_from,
                    quotes_from=quotes_from,
                    dataset_name=dataset_name,
                    scores_outpath=scores_outpath,
                    run_quote_attribution=run_quote_attribution)

        self.output_dirpath = output_dirpath
        self.char_output_dirpath = os.path.join(output_dirpath, 'char_coref_chars')
        self.coref_output_dirpath = os.path.join(output_dirpath, 'char_coref_stories')
        self.quote_output_dirpath = os.path.join(output_dirpath, 'quote_attribution')

    def evaluate(self):
        fic_scores = {'coref': [], 'quotes': []} 
        quote_groups = defaultdict(list)

        # Modify fics if need be
        modified_suffix = ''
        if self.whether_evaluate_quotes and self.coref_from == 'gold':
            for fname in sorted(os.listdir(self.char_output_dirpath)):
                fandom_fname = fname.split('.')[0]
                modified_suffix = ''
                pipeline_output = PipelineOutput(self.output_dirpath, fandom_fname, fic_csv_dirpath=self.fic_csv_dirpath)
                if self.quotes_from == 'gold': # TODO: doesn't work
                    modified_suffix += pipeline_output.modify_quote_spans(self.quote_annotations_dirpath, self.quote_annotations_ext)
                modified_suffix += pipeline_output.modify_coref_files(self.coref_annotations_dirpath, self.coref_annotations_ext)
    
            modified_coref_output_dirpath = self.coref_output_dirpath.rstrip('/') + modified_suffix
            modified_char_output_dirpath = self.char_output_dirpath.rstrip('/') + modified_suffix
            modified_quote_output_dirpath = self.quote_output_dirpath.rstrip('/') + modified_suffix
            if self.run_quote_attribution:
                self.run_pipeline_quote_attribution(modified_coref_output_dirpath, modified_char_output_dirpath, modified_quote_output_dirpath)

        # Evaluate fics
        for fname in sorted(os.listdir(self.char_output_dirpath)):
            fandom_fname = fname.split('.')[0]
            print(fandom_fname)
            sys.stdout.flush()
            coref_scores, quote_scores, fic_quote_groups = self.evaluate_fic(fandom_fname, modified_suffix=modified_suffix)
            if self.whether_evaluate_coref:
                fic_scores['coref'].append(coref_scores)
            if self.whether_evaluate_quotes:
                fic_scores['quotes'].append(quote_scores)
                for group, values in fic_quote_groups.items():
                    quote_groups[group].extend(values)

        # Calculate corpus-wide stats, save scores
        if self.whether_evaluate_coref:
            self.save_scores(fic_scores['coref'], 'pipeline', ['coref'])
            f1_scores = [scores['lea_f1'] for scores in fic_scores['coref']]
            print(f"Average LEA F1: {np.mean(f1_scores): .2%}")

        if self.whether_evaluate_quotes:
            attribution_f1_scores = [scores['attribution_f1'] for scores in fic_scores['quotes']]
            print(f"Average attibution F1: {np.mean(attribution_f1_scores): .2%}")

            aggregate_quote_scores = {}
            aggregate_quote_scores['extraction_f1'], aggregate_quote_scores['extraction_precision'], aggregate_quote_scores['extraction_recall'] = scorer.quote_extraction_scores(quote_groups['predicted_quotes'], quote_groups['gold_quotes'], quote_groups['matched_pred_quotes'], quote_groups['matched_gold_quotes'], quote_groups['false_positives'], quote_groups['false_negatives'])
            aggregate_quote_scores['attribution_f1'], aggregate_quote_scores['attribution_precision'], aggregate_quote_scores['attribution_recall'] = scorer.quote_attribution_scores(quote_groups['predicted_quotes'], quote_groups['gold_quotes'], quote_groups['correct_attributions'], quote_groups['incorrect_attributions'])
            for measure, val in sorted(aggregate_quote_scores.items()):
                print(f"Overall {measure}: {val: .2%}")

            pdb.set_trace()
            self.save_scores(fic_scores['quotes'], 
                'pipeline', ['quotes', f'{self.coref_from}_coref', f'{self.quotes_from}_quotes'])

    def evaluate_fic(self, fandom_fname, modified_suffix=''):
        coref_scores, quote_scores, quote_groups = None, None, None

        print("\tLoading file...")
        sys.stdout.flush()

        # Load output, CSV file of fic
        pipeline_output = PipelineOutput(self.output_dirpath, fandom_fname, fic_csv_dirpath=self.fic_csv_dirpath, modified_suffix=modified_suffix)

        if self.whether_evaluate_coref:
            coref_scores = self.evaluate_coref(fandom_fname, pipeline_output)

        if self.whether_evaluate_quotes:
            quote_scores, quote_groups = self.evaluate_quotes(fandom_fname, pipeline_output)
            quote_scores['filename'] = fandom_fname

        return coref_scores, quote_scores, quote_groups

    def run_pipeline_quote_attribution(self, coref_output_dirpath, coref_chars_output_dirpath, quote_output_dirpath):
        """ Runs pipeline quote attribution.
            Saves to coref_output_dirpath, which is then read in load_quotes_json
        """
        # Run pipeline
        wrapper = PipelineWrapper(coref_output_dirpath, coref_chars_output_dirpath, quote_output_dirpath)
        wrapper.run()


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
        coref_from=config.get('Settings', 'load_coref_from'), 
        quotes_from=config.get('Settings', 'load_quotes_from'), 
        run_quote_attribution=config.getboolean('Settings', 'run_quote_attribution'), 
        dataset_name = config.get('Settings', 'dataset_name'),
        scores_outpath = config.get('Filepaths', 'scores_outpath'),
        coref_annotations_dirpath = config.get('Filepaths', 'coref_annotations_dirpath'),
        quote_annotations_dirpath = config.get('Filepaths', 'quote_annotations_dirpath'),
        predicted_coref_outpath = config.get('Filepaths', 'predicted_coref_outpath'),
        predicted_quotes_outpath = config.get('Filepaths', 'predicted_quotes_outpath'),
        )

    evaluator.evaluate()


if __name__ == '__main__':
    main()
