#!/usr/bin/env python
# coding: utf-8

# # Post-process pipeline output to get entity mention clusters

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

import coref_evaluation_utils as utils


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


def nth_repl(s, sub, repl, nth):
    """ From https://stackoverflow.com/questions/35091557/replace-nth-occurrence-of-substring-in-string/35092436#35092436 """

    find = s.find(sub)
    # if find is not p1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match

    while find != -1 and i != nth:
        # find + 1 means we start at the last match start index + 1
        find = s.find(sub, find + 1)
        i += 1
    # if i  is equal to nth we found nth matches so replace

    if i == nth:
        return s[:find]+repl+s[find + len(sub):]
    return s


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


def extract_pipeline_entity_mentions(text):
    """ Return token start and endpoints of entity mentions embedded in text. """
    
    token_count = 1 # pointer to current token in subscripted file
    entities = {} # cluster_name: {(token_id_start, token_id_end), ...}
    
    tokens = text.split()
    
    for i, token in enumerate(tokens):
        
        if token.startswith('($_'): # entity cluster name
            if not token in entities:
                entities[token] = set()
                
            mention = tokens[i-1] # token before the parentheses mention
            mention_len = len(mention.split('_'))
            token_id_start = token_count - 1 # token before the parentheses mention
            token_id_end = token_id_start + (mention_len - 1)
            
            token_count = token_id_end + 1
                
            entities[token].add((token_id_start, token_id_end))
            
        else:
            # Advance token count
            token_count += 1
            
    return entities
    

def extract_pipeline_entity_clusters(system_output, fname, save_path=None):
    """ For a particular fic, extract entity mentions and group them into associated clusters. """

    predicted_entities = {}

    for row in list(system_output.itertuples()):
        fic_id = row.fic_id
        chapter_id = row.chapter_id
        para_id = row.para_id
        entities = extract_pipeline_entity_mentions(row.text_tokenized)
#         print(entities)
#         print(row.text_tokenized)
        
        if not fic_id in predicted_entities:
            predicted_entities[fic_id] = {}
        
        for cluster_name in entities:
            if not cluster_name in predicted_entities[fic_id]:
                predicted_entities[fic_id][cluster_name] = set()
            
            for mention in entities[cluster_name]:
                token_id_start = mention[0]
                token_id_end = mention[1]
                predicted_entities[fic_id][cluster_name].add((chapter_id, para_id, token_id_start, token_id_end))

    if save_path:
        outpath = os.path.join(save_path, f'pipeline_clusters_{fic_id}.pkl')
        with open(os.path.join(outpath), 'wb') as f:
            pickle.dump(predicted_entities, f)

    return predicted_entities


def main():

    # I/O
    # Input
    predictions_dirpath = '/data/fanfiction_ao3/annotated_10fandom/test/pipeline_output/char_coref_stories'
    annotations_dirpath = '/data/fanfiction_ao3/annotated_10fandom/test/entity_clusters'
    csv_dirpath = '/data/fanfiction_ao3/annotated_10fandom/test/fics/'

    # Output
    predicted_entities_outpath = '/projects/fanfiction-nlp/tmp/predicted_entities/'

    #csv_fnames = [fname for fname in sorted(os.listdir(predictions_dirpath)) if fname.endswith('.csv') and not fname.startswith('sherlock')]
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

        print("\tExtracting predicted entities and clusters...")
        sys.stdout.flush()

        ## Extract entity mention tuples, clusters
        predicted_entities = extract_pipeline_entity_clusters(system_output, fname, save_path=predicted_entities_outpath)
        gold_entities = utils.extract_gold_entities(annotations_dirpath)

        print("\tCalculating LEA...")
        sys.stdout.flush()
        utils.calculate_lea(predicted_entities, gold_entities)
        print()


if __name__ == '__main__':
    main()
