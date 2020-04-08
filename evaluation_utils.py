""" Utility functions for coref evaluation on different systems """

import pandas as pd
import os
import itertools
import numpy as np


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



def extract_gold_character_spans(annotations_dirpath):
    # Load ground-truth annotated entity mentions or quote spans, attributed to each character

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


def load_fic_csv(csv_dirpath, fname):
    csv_fpath = os.path.join(csv_dirpath, f'{fname.split(".")[0]}.csv')
    return pd.read_csv(csv_fpath)


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


if __name__ == '__main__': main()
