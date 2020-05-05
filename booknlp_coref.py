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


def modify_paragraph_id(para_id, trouble_line):
   
    if para_id == trouble_line: # trouble line
        new_para_id = trouble_line + 2 # add 1 anyway for the index-0 issue
        
    if para_id >= trouble_line + 1:
        new_para_id = para_id
    
    else:
        new_para_id = para_id + 1 # add 1 since BookNLP starts with 0
        
    return new_para_id


def find_mismatched_paragraphs(booknlp_output, fic_csv, n_diff):
    """ Right now only handles 1 mismatch """

    if n_diff != 1:
        pdb.set_trace()

    trouble_line = -1

    booknlp_paras = booknlp_output.groupby('paragraphId').agg({'originalWord': lambda x: ' '.join(x.tolist())})['originalWord']

    for i, (booknlp_para, fic_para) in enumerate(zip(booknlp_paras, fic_csv['text_tokenized'])):
        # There will be tokenization differences, so look for dramatic differences
        if abs(len(booknlp_para.split()) - len(fic_para.split())) > 10:
            trouble_line = i
            break

    return trouble_line


def load_booknlp_output(predictions_dirpath, fname):
    return pd.read_csv(os.path.join(predictions_dirpath, fname), sep='\t', quoting=csv.QUOTE_NONE)


def load_fic_csv(csv_dirpath, fname):
    csv_fpath = os.path.join(csv_dirpath, f'{fname.split(".")[0]}.csv')
    return pd.read_csv(csv_fpath)


def check_paragraph_breaks(booknlp_output, fic_csv):

    if not 'full_paragraphId' in booknlp_output.columns:
        booknlp_output['full_paragraphId'] = booknlp_output['paragraphId'] + 1

    n_para_diff = len(booknlp_output['full_paragraphId'].unique()) - len(set(zip(fic_csv['chapter_id'], fic_csv['para_id'])))

    # Add in chapter ID column
    para2chap = {} # modified_paragraphId: (chapter_id, para_id)
    para_offset = 0
    if n_para_diff == 0:
        chap_paras = list(zip(fic_csv['chapter_id'], fic_csv['para_id']))
        all_paras = sorted(booknlp_output['full_paragraphId'].unique().tolist())
        
        for i, para_num in enumerate(all_paras):
            para2chap[para_num] = chap_paras[i]

    booknlp_output['chapterId'] = booknlp_output['full_paragraphId'].map(lambda x: para2chap[x][0])
    booknlp_output['modified_paragraphId'] = booknlp_output['full_paragraphId'].map(lambda x: para2chap[x][1])

    return (n_para_diff, booknlp_output)


def fix_paragraph_breaks(booknlp_output, fic_csv, n_diff):
    # Fix paragraph break issues
    trouble_line = find_mismatched_paragraphs(booknlp_output, fic_csv, n_diff)
    booknlp_output['modified_paragraphId'] = [modify_paragraph_id(para_id, trouble_line) for para_id in booknlp_output['paragraphId']]

    # Confirm that fixed it
    n_para_diff, _ = check_paragraph_breaks(booknlp_output, fic_csv)
    if n_para_diff != 0:
        pdb.set_trace()

    return booknlp_output


def get_misaligned_paragraph(booknlp_output, fic_csv):
    # Get token counts for paragraphs from BookNLP, make sure they match the original fic token counts

    fic_csv['booknlp_para_length'] = booknlp_output.groupby('full_paragraphId').size().tolist() # Paragraphs should be in the same order (already checked this)
    fic_csv['token_count'] = fic_csv['text_tokenized'].map(lambda x: len(x.split()))
    misaligned_rows = fic_csv.loc[fic_csv['token_count'] != fic_csv['booknlp_para_length'], ['chapter_id', 'para_id', 'token_count', 'booknlp_para_length']]

    return misaligned_rows
    

def fix_token_misalignment(misaligned_rows, booknlp_output, fic_csv):
    # Fix token misalignment issues
    modified_booknlp = booknlp_output.copy()

    for selected_chap_id, selected_para_id in zip(misaligned_rows['chapter_id'], misaligned_rows['para_id']):

        gold_tokens = fic_csv.loc[(fic_csv['chapter_id']==selected_chap_id) & (fic_csv['para_id']==selected_para_id), 'text_tokenized'].tolist()[0].split()
        booknlp_tokens = booknlp_output.loc[(booknlp_output['chapterId']==selected_chap_id) & (booknlp_output['modified_paragraphId']==selected_para_id), 'originalWord'].tolist()

        total_offset = 0
        trouble_offsets = {} # line_number: offset
        first_tokenId = booknlp_output.loc[(booknlp_output['chapterId']==selected_chap_id) & (booknlp_output['modified_paragraphId']==selected_para_id), 'tokenId'].tolist()[0]

        for i, gold_tok in enumerate(gold_tokens):

            current_booknlp_token = booknlp_tokens[i + total_offset]
            if not gold_tok == current_booknlp_token:

                # Try adding tokens
                added = current_booknlp_token
                for offset in range(1, 4):
                    added += booknlp_tokens[i + total_offset + offset]
                    if added == gold_tok:
                        total_offset += offset
                        trouble_offsets[first_tokenId + i] = offset
                        break

                else:
                    print(gold_tok)
                    print(booknlp_tokens[i])
                    pdb.set_trace()

        # Modify BookNLP output
        for line, offset in trouble_offsets.items():
            row_filter = (modified_booknlp['chapterId']==selected_chap_id) & (modified_booknlp['modified_paragraphId']==selected_para_id) & (modified_booknlp['tokenId'].isin(range(line, line+offset+1)))

            # Modify offset word
            new_word = ''.join(modified_booknlp.loc[row_filter, 'originalWord'].tolist())
            modified_row_filter = (modified_booknlp['chapterId']==selected_chap_id) & (modified_booknlp['modified_paragraphId']==selected_para_id) & (modified_booknlp['tokenId']==line)
            modified_booknlp.loc[modified_row_filter, 'originalWord'] = new_word

            # Delete offset words
            delete_row_filter = (modified_booknlp['chapterId']==selected_chap_id) & (modified_booknlp['modified_paragraphId']==selected_para_id) & (modified_booknlp['tokenId'].isin(range(line+1, line+offset+1)))
            delete_index = modified_booknlp.loc[delete_row_filter].index
            modified_booknlp.drop(index=delete_index, inplace=True)

    # Confirm token length match
    misaligned_rows = get_misaligned_paragraph(modified_booknlp, fic_csv)
    if len(misaligned_rows) > 0:
        pdb.set_trace()

    return modified_booknlp


def renumber_token_ids(modified_booknlp):
    para_token_lengths = modified_booknlp.groupby('full_paragraphId').size().tolist()
    new_tokenIds = sum([list(range(1, para_length+1)) for para_length in para_token_lengths], [])
    modified_booknlp['modified_tokenId'] = new_tokenIds
    return modified_booknlp


def extract_entity_mentions(modified_booknlp, fname, save_path=None):
    selected_cols = ['chapterId', 'modified_paragraphId', 'modified_tokenId', 'characterId', 'originalWord']
    mentions = modified_booknlp[modified_booknlp['characterId']>-1].loc[:, selected_cols]

    # Calculate end tokens for any entity mentions
    mentions['next_entity_tokenId'] = mentions['modified_tokenId'].tolist()[1:] + [0]
    mentions['next_entity_paragraphId'] = mentions['modified_paragraphId'].tolist()[1:] + [0]
    mentions['next_entity_characterId'] = mentions['characterId'].tolist()[1:] + [0]
    mentions['sequential'] = [(next_entity_tokenId == modified_tokenId + 1) and                               (next_entity_paragraphId == modified_paragraphId) and                               (next_entity_characterId == characterId) 
                            for next_entity_tokenId, modified_tokenId, next_entity_paragraphId, modified_paragraphId, next_entity_characterId, characterId in \
                              zip(mentions['next_entity_tokenId'], mentions['modified_tokenId'], mentions['next_entity_paragraphId'], \
                                  mentions['modified_paragraphId'], mentions['next_entity_characterId'], mentions['characterId'])
                                 ]

    predicted_entities = {}

    prev_was_sequential = False
    prev_token_id_start = 0

    fic_id = int(fname.split('.')[0].split('_')[1])

    for row in list(mentions.itertuples()):
        chapter_id = row.chapterId
        para_id = row.modified_paragraphId
        character_id = row.characterId
        token_id_start = row.modified_tokenId

        if row.sequential: # Store last token ID
            if prev_was_sequential: # in the middle of an entity mention
                continue
            else:
                prev_was_sequential = True
                prev_token_id_start = token_id_start
                continue

        # Save entity mention
        if not fic_id in predicted_entities:
            predicted_entities[fic_id] = {}

        if not character_id in predicted_entities[fic_id]:
            predicted_entities[fic_id][character_id] = set()

        if prev_was_sequential:
            token_id_start = prev_token_id_start

        token_id_end = row.modified_tokenId

        predicted_entities[fic_id][character_id].add((chapter_id, para_id, token_id_start, token_id_end))

        prev_was_sequential = row.sequential

    if save_path:
        outpath = os.path.join(save_path, f'booknlp_{fic_id}.pkl')
        with open(os.path.join(outpath), 'wb') as f:
            pickle.dump(predicted_entities, f)

    return predicted_entities


def extract_gold_entities(annotations_dirpath):
    # Load ground-truth annotated entity mentions
    gold_entities = {} # fic_id: {cluster_name: {(chapter_id, paragraph_id, token_id_start, token_id_end), ...}}

    for fname in sorted(os.listdir(annotations_dirpath)):
        
        fic_id = int(fname.split('_')[1])
        gold_entities[fic_id] = {}
        
        df = pd.read_csv(os.path.join(annotations_dirpath, fname))
        for colname in df.columns:
            gold_entities[fic_id][colname] = set()
            for mention in df[colname].dropna():
                parts = mention.split('.')
                chapter_id = int(parts[0])
                paragraph_id = int(parts[1])
                if '-' in parts[2]:
                    token_id_start = int(parts[2].split('-')[0])
                    token_id_end = int(parts[2].split('-')[-1])
                else:
                    token_id_start = int(parts[2])
                    token_id_end = int(parts[2])
                    
                gold_entities[fic_id][colname].add((chapter_id, paragraph_id, token_id_start, token_id_end))

    return gold_entities


def fic_stats(fic_dirpath, fname, gold_entities):
    # Length of fic

    fic = pd.read_csv(os.path.join(fic_dirpath, fname))
    n_words = sum(fic['text_tokenized'].map(lambda x: len(x.split())))
    print(f'{fname}: {n_words}')

    # Number of characters
    if fname in gold_entities:
        print(f'{fname}: {len(gold_entities[fname])}')


def links(entity_mentions):
    """ Returns all the links in an entity between mentions """
    
    if len(entity_mentions) == 1: # self-link
        links = {list(entity_mentions)[0], list(entity_mentions)[0]}

    else:
        links = set(itertools.combinations(entity_mentions, 2))
        
    return links


def lea_recall(predicted_entities, gold_entities):
    
    fic_recalls = {}
    
    for fic_id in gold_entities:
        
        cluster_resolutions = {}
        cluster_sizes = {}
        
        for gold_cluster, gold_mentions in gold_entities[fic_id].items():
            gold_links = links(gold_mentions)
            
            cluster_resolution = 0
            
            for predicted_cluster, predicted_mentions in predicted_entities[fic_id].items():
                predicted_links = links(predicted_mentions)
                
                cluster_resolution += len(predicted_links.intersection(gold_links))
                
            cluster_resolution = cluster_resolution/len(gold_links)
            cluster_resolutions[gold_cluster] = cluster_resolution
            cluster_sizes[gold_cluster] = len(gold_mentions)
            
        # take importance (size) of clusters into account
        fic_recalls[fic_id] = sum([cluster_sizes[c] * cluster_resolutions[c] for c in gold_entities[fic_id]])/sum(cluster_sizes.values())
        
    # Total recall as mean across fics
    total_recall = np.mean(list(fic_recalls.values()))
    return total_recall, fic_recalls


def lea_precision(predicted_entities, gold_entities):
    
    fic_precisions = {}
    
    for fic_id in gold_entities:
        
        cluster_resolutions = {}
        cluster_sizes = {}
        
        for predicted_cluster, predicted_mentions in predicted_entities[fic_id].items():
            predicted_links = links(predicted_mentions)
            
            cluster_resolution = 0
            
            for gold_cluster, gold_mentions in gold_entities[fic_id].items():
                gold_links = links(gold_mentions)
                cluster_resolution += len(predicted_links.intersection(gold_links))
            
            cluster_resolution = cluster_resolution/len(predicted_links)
            cluster_resolutions[predicted_cluster] = cluster_resolution
            cluster_sizes[predicted_cluster] = len(predicted_mentions)
            
        # take importance (size) of clusters into account
        fic_precisions[fic_id] = sum([cluster_sizes[c] * cluster_resolutions[c] for c in predicted_entities[fic_id]])/sum(cluster_sizes.values())
        
    # Total precision as mean across fics
    total_precision = np.mean(list(fic_precisions.values()))
    return total_precision, fic_precisions


def f_score(precision, recall):
    return 2 * (precision * recall)/(precision + recall)


def calculate_lea(predicted_entities, gold_entities):
    # # Calculate LEA coreference evaluation

    fics = set(predicted_entities.keys()).intersection(set(gold_entities.keys()))

    selected_predicted = {fic: predicted_entities[fic] for fic in fics}
    selected_gold = {fic: gold_entities[fic] for fic in fics}

    recall, fic_recalls = lea_recall(selected_predicted, selected_gold)
    precision, fic_precisions = lea_precision(selected_predicted, selected_gold)
    f1 = f_score(precision, recall)

    print(f"\t\tPrecision: {precision: .2%}")
    print(f"\t\tRecall: {recall: .2%}")
    print(f"\t\tF-score: {f1: .2%}")


def main():

    # I/O
    predictions_dirpath = '/projects/book-nlp/data/tokens/annotated_10fandom_test/'
    annotations_dirpath = '/data/fanfiction_ao3/annotated_10fandom/test/entity_clusters'
    csv_dirpath = '/data/fanfiction_ao3/annotated_10fandom/test/fics/'
    predicted_entities_outpath = '/projects/book-nlp/tmp/predicted_entities/'

    for fname in sorted(os.listdir(predictions_dirpath)):
    
        print(fname)
        sys.stdout.flush()

        print("\tLoading files...")
        sys.stdout.flush()
        # Load output, CSV file of fic
        booknlp_output = load_booknlp_output(predictions_dirpath, fname)
        fic_csv = load_fic_csv(csv_dirpath, fname)

        print("\tChecking/fixing paragraph breaks...")
        sys.stdout.flush()
        ## Check paragraph breaks
        # Compare number of paragraphs
        n_diff, booknlp_output = check_paragraph_breaks(booknlp_output, fic_csv)
        if n_diff != 0:
            pdb.set_trace()
            fix_paragraph_breaks(booknlp_output, fic_csv, n_diff)

        print("\tChecking/fixing token alignment...")
        sys.stdout.flush()
        ## Make sure token IDs align
        misaligned_rows = get_misaligned_paragraph(booknlp_output, fic_csv)
        if len(misaligned_rows) > 0:
            #print(f"\tFound {len(misaligned_rows)} misaligned rows")
            booknlp_output = fix_token_misalignment(misaligned_rows, booknlp_output, fic_csv)

        ## Renumber BookNLP token IDs
        booknlp_output = renumber_token_ids(booknlp_output)
        #if fname.startswith('sherlock_1296961'):
        #    pdb.set_trace()

        print("\tExtracting predicted entities and clusters...")
        sys.stdout.flush()
        ## Extract entity mention tuples, clusters
        predicted_entities = extract_entity_mentions(booknlp_output, fname, save_path=predicted_entities_outpath)
        gold_entities = extract_gold_entities(annotations_dirpath)

        print("\tCalculating LEA...")
        sys.stdout.flush()
        calculate_lea(predicted_entities, gold_entities)
        print()


if __name__ == '__main__':
    main()
