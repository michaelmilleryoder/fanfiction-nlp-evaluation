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

import pipeline_coref as coref

from evaluator import Evaluator
from annotation import QuoteAnnotation
import evaluation_utils as utils
from pipeline_output import PipelineOutput


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
            # TODO: fill in, hopefully don't have to do checks for alignment

            #print("\tChecking/fixing paragraph breaks...")
            #sys.stdout.flush()
            ### Check paragraph breaks
            ## Compare number of paragraphs
            #n_diff = check_pipeline_paragraph_breaks(system_output, fic_csv)
            #if n_diff != 0:
            #    pdb.set_trace()

            #print("\tChecking/fixing token alignment...")
            #sys.stdout.flush()
            #misaligned_rows = get_pipeline_misaligned_paragraphs(system_output, fic_csv)
            #if len(misaligned_rows) > 0:
            #    #print(f"\tFound {len(misaligned_rows)} misaligned rows")
            #    system_output = fix_pipeline_token_misalignment(misaligned_rows, system_output, fic_csv)

            #print("\tExtracting predicted entities and clusters...")
            #sys.stdout.flush()

            ### Extract entity mention tuples, clusters
            #predicted_entities = coref.extract_pipeline_entity_clusters(system_output, fname, save_path=predicted_coref_outpath)
            #gold_entities = utils.extract_gold_character_spans(coref_annotations_dirpath, fname)

            #print("\tCalculating LEA...")
            #sys.stdout.flush()
            #utils.calculate_lea(predicted_entities, gold_entities)
            #print()

            self.evaluate_coref(fandom_fname, pipeline_output)
    
            pass

        if self.whether_evaluate_quotes:
            self.evaluate_quotes(fandom_fname, pipeline_output)

    def evaluate_coref(self, fandom_fname, pipeline_output):
        """ Evaluate coref for a fic. 
            Args:
                save: save AnnotatedSpan objects in a pickled file in a tmp directory
        """
        
        # Load gold mentions
        gold = CorefAnnotation(self.coref_annotations_dirpath, fandom_fname, fic_csv_dirpath=self.fic_csv_dirpath)

        # Load predicted mentions
        pipeline_output.extract_character_mentions(save_dirpath=self.predicted_coref_outpath)

        # Print scores
        utils.print_quote_scores(pipeline_output.character_mentions, gold.character_mentions, exact_match=True)

    def evaluate_quotes(self, fandom_fname, pipeline_output):
        """ Evaluate quotes for a fic. 
            Args:
                save: save Quote objects in a pickled file in a tmp directory
        """
        
        # Quote extraction evaluation
        # Load gold quote spans
        gold = QuoteAnnotation(self.quote_annotations_dirpath, fandom_fname, fic_csv_dirpath=self.fic_csv_dirpath)

        # Load predicted quote spans (from BookNLP output to Quote objects)
        pipeline_output.extract_quotes(save_dirpath=self.predicted_quotes_outpath)

        # Print scores
        utils.print_quote_scores(pipeline_output.quotes, gold.quotes, exact_match=True)


# OLD FUNCTIONS
def load_pipeline_output(coref_predictions_dirpath, fname):
    return pd.read_csv(os.path.join(coref_predictions_dirpath, fname))


def fix_pipeline_token_misalignment(misaligned_rows, sys_output, fic_csv):
    """ Fix token misalignment issues. 
        Assumes more tokens were found in system output.
    """

    system_output = sys_output.copy()

    for selected_chap_id, selected_para_id in zip(misaligned_rows['chapter_id'], misaligned_rows['para_id']):

        gold_para = fic_csv.loc[(fic_csv['chapter_id']==selected_chap_id) & (fic_csv['para_id']==selected_para_id), 'text_tokenized'].tolist()[0]
        gold_tokens = utils.preprocess_para(gold_para).split()
        pipeline_para = system_output.loc[(system_output['chapter_id']==selected_chap_id) & (system_output['para_id']==selected_para_id), 'text_tokenized'].tolist()[0]
        preprocessed = utils.preprocess_para(pipeline_para)
        pipeline_tokens = preprocessed.split()

        total_offset = 0
        token_replacements = defaultdict(int) # {(original_token, modified_token, start_index): n_replacements}

        for i, gold_tok in enumerate(gold_tokens):

            if i + total_offset >= len(pipeline_tokens):
                pdb.set_trace()

            current_pipeline_token = pipeline_tokens[i + total_offset]
            if not gold_tok == current_pipeline_token:

                # Try adding tokens
                added = [current_pipeline_token]
                for offset in range(1, 4):
                    if i + total_offset + offset >= len(pipeline_tokens):
                        pdb.set_trace()
                    next_token = pipeline_tokens[i + total_offset + offset]
                    added.append(next_token)

                    if ''.join(added) == gold_tok:
                        total_offset += offset
                        token_replacements[(' '.join(added), ''.join(added), preprocessed.index(' '.join(added)))] += 1
                        break

                else:

                    # rare double-token case
                    if gold_tok in added:
                        to_delete = added[:added.index(gold_tok)] # delete the other tokens
                        total_offset += len(to_delete)
                        token_replacements[(' '.join(to_delete), '', preprocessed.index(' '.join(added)))] += 1

                    else:
                        pdb.set_trace()

        # Modify paragraph
        modified_para = preprocessed
        char_diff = 0
        for (old, new, token_index), count in token_replacements.items():
            idx = token_index + char_diff
            modified_para = modified_para[:idx] + modified_para[idx:].replace(old, new, count)
            char_diff += len(new)-len(old)

        system_output.loc[(system_output['chapter_id']==selected_chap_id) & (system_output['para_id']==selected_para_id), 'text_tokenized'] = modified_para

    # Confirm token length match
    new_misaligned_rows = get_pipeline_misaligned_paragraphs(system_output, fic_csv)
    if len(new_misaligned_rows) > 0:
        pdb.set_trace()

    return system_output


def check_pipeline_paragraph_breaks(pipeline_output, fic_csv):
    return abs(len(pipeline_output) - len(fic_csv))


def remove_character_tags(text):
    # Remove character parentheses
    modified_para = re.sub(r'\(\$_.*?\)\ ', '', text)

    # Split up character underscore mentions
    modified_para = re.sub(r'([^ ])_([^ ])', r'\1 \2', modified_para)
    modified_para = re.sub(r'([^ ])_([^ ])', r'\1 \2', modified_para) # twice

    return modified_para


def count_pipeline_output_tokens(output_para):
    modified_para = remove_character_tags(output_para)

    return len(modified_para.split())


def get_pipeline_misaligned_paragraphs(system_output, fic_csv):
    # Get token counts for paragraphs, make sure they match the original fic token counts

    fic_csv['system_para_length'] = system_output['text_tokenized'].map(count_pipeline_output_tokens)
    fic_csv['token_count'] = fic_csv['text_tokenized'].map(lambda x: len(x.split()))
    misaligned_rows = fic_csv.loc[fic_csv['token_count'] != fic_csv['system_para_length'], ['chapter_id', 'para_id', 'token_count', 'system_para_length']]

    return misaligned_rows

# END OLD FUNCTIONS


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
