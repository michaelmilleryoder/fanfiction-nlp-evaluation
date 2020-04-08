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

import evaluation_utils as utils
import pipeline_coref as coref


def load_pipeline_output(predictions_dirpath, fname):
    return pd.read_csv(os.path.join(predictions_dirpath, fname))


def preprocess_para(text):

    new_text = remove_character_tags(text)

    changes = {
        '’': "'",
        '‘': "'",
        '“': '"',
        '”': '"',
        "''": '"',
        '``': '"',
        '`': "'",
        '…': "...",
        '-LRB-': "(",
        '-RRB-': ")",
        "--": '—',
        '«': '"',
        '»': '"',
    }

    for change, new in changes.items():
        new_text = new_text.replace(change, new)

    return new_text


def fix_pipeline_token_misalignment(misaligned_rows, sys_output, fic_csv):
    """ Fix token misalignment issues. 
        Assumes more tokens were found in system output.
    """

    system_output = sys_output.copy()

    for selected_chap_id, selected_para_id in zip(misaligned_rows['chapter_id'], misaligned_rows['para_id']):

        gold_para = fic_csv.loc[(fic_csv['chapter_id']==selected_chap_id) & (fic_csv['para_id']==selected_para_id), 'text_tokenized'].tolist()[0]
        gold_tokens = preprocess_para(gold_para).split()
        pipeline_para = system_output.loc[(system_output['chapter_id']==selected_chap_id) & (system_output['para_id']==selected_para_id), 'text_tokenized'].tolist()[0]
        preprocessed = preprocess_para(pipeline_para)
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
        #if selected_para_id == 10:
        #    pdb.set_trace()

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


def main():

    # I/O
    # Input
    dataset_dirpath = '/data/fanfiction_ao3_annotated_10fandom/test'
    coref_annotations_dirpath = os.path.join(dataset_dirpath, 'entity_clusters')
    quote_annotations_dirpath = os.path.join(dataset_dirpath, 'quote_attribution')
    csv_dirpath = os.path.join(dataset_dirpath, 'fics')
    predictions_dirpath = os.path.join(dataset_dirpath, 'pipeline_output/char_coref_stories'

    # Output
    predicted_entities_outpath = '/projects/fanfiction-nlp/tmp/predicted_entities/'

    # Settings
    eval_coref = False
    eval_quote_attribution = True

    csv_fnames = [fname for fname in sorted(os.listdir(predictions_dirpath)) if fname.endswith('.csv')]

    for fname in csv_fnames:
        
        if not fname.endswith('.csv'):
            continue

        print(fname)
        sys.stdout.flush()

        print("\tLoading files...")
        sys.stdout.flush()
        # Load output, CSV file of fic
        system_output = load_pipeline_output(predictions_dirpath, fname)
        fic_csv = utils.load_fic_csv(csv_dirpath, fname)

        print("\tChecking/fixing paragraph breaks...")
        sys.stdout.flush()
        ## Check paragraph breaks
        # Compare number of paragraphs
        n_diff = check_pipeline_paragraph_breaks(system_output, fic_csv)
        if n_diff != 0:
            pdb.set_trace()

        print("\tChecking/fixing token alignment...")
        sys.stdout.flush()
        misaligned_rows = get_pipeline_misaligned_paragraphs(system_output, fic_csv)
        if len(misaligned_rows) > 0:
            #print(f"\tFound {len(misaligned_rows)} misaligned rows")
            system_output = fix_pipeline_token_misalignment(misaligned_rows, system_output, fic_csv)

        if eval_coref:
            print("\tExtracting predicted entities and clusters...")
            sys.stdout.flush()

            ## Extract entity mention tuples, clusters
            predicted_entities = coref.extract_pipeline_entity_clusters(system_output, fname, save_path=predicted_entities_outpath)
            gold_entities = utils.extract_gold_character_spans(coref_annotations_dirpath)

            print("\tCalculating LEA...")
            sys.stdout.flush()
            utils.calculate_lea(predicted_entities, gold_entities)
            print()

        if eval_quote_attribution:
            print("\tExtracting predicted quote attribution...")
            sys.stdout.flush()

            ## Extract pipeline quote attribution
            #predicted_quotes = coref.extract_pipeline_quotes(system_output, fname, save_path=predicted_entities_outpath)
            gold_quotes = utils.extract_gold_character_spans(quote_annotations_dirpath)


if __name__ == '__main__':
    main()
