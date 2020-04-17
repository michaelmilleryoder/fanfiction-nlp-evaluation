#!/usr/bin/env python
# coding: utf-8

import os
import pdb
import pickle

import evaluation_utils as utils


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
        
        for cluster_name in entities:
            if not cluster_name in predicted_entities[fic_id]:
                predicted_entities[cluster_name] = set()
            
            for mention in entities[cluster_name]:
                token_id_start = mention[0]
                token_id_end = mention[1]
                predicted_entities[cluster_name].add((chapter_id, para_id, token_id_start, token_id_end))

    if save_path:
        outpath = os.path.join(save_path, f'pipeline_clusters_{fic_id}.pkl')
        with open(os.path.join(outpath), 'wb') as f:
            pickle.dump(predicted_entities, f)

    return predicted_entities


def main():
    pass

if __name__ == '__main__':
    main()
