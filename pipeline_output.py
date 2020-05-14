""" Class for loading, storing and processing output of pipeline """

import os
import pickle
import json
import pandas as pd
import pdb

from fic_representation import FicRepresentation
import evaluation_utils as utils
from quote import Quote
from annotated_span import AnnotatedSpan
from character_mention_parser import CharacterMentionParser


def extract_mention_tags(text):
    """ Returns AnnotatedSpan objects with start and end token IDs annotated with character """
    parser = CharacterMentionParser()
    parser.feed(text)
    return parser.character_mentions


class PipelineOutput(FicRepresentation):
    """ Holds representation for the pipeline-processed output of a fic. """

    def __init__(self, output_dirpath, fandom_fname): 
        super().__init__(fandom_fname)
        self.coref_output_dirpath = os.path.join(output_dirpath, 'char_coref_stories')
        self.quote_output_dirpath = os.path.join(output_dirpath, 'quote_attribution')

    def load_quote_json(self):
        """ Returns pipeline output quotes from json """

        quote_predictions_fpath = os.path.join(self.quote_output_dirpath, f'{self.fandom_fname}.quote.json')

        with open(quote_predictions_fpath) as f:
            predicted_quotes = json.load(f)

        return predicted_quotes

    def load_coref_csv(self):
        """ Returns a pandas DataFrame of CSV with coref <tags> """

        return pd.read_csv(os.path.join(self.coref_output_dirpath, f'{self.fandom_fname}.coref.csv'))

    def extract_quotes(self, save_dirpath=None):
        """ Extracts quotes into AnnotatedSpan objects, saves in self.quotes, also in tmp directory if specified.
        """

        self.quotes = [] # list of AnnotatedSpan objects
        
        predicted_quotes_json = self.load_quote_json()

        for quote_entry in predicted_quotes_json:
            
            character = quote_entry['speaker']
            chap_id = quote_entry['chapter']
            para_id = quote_entry['paragraph']
            quotes = quote_entry['quotes']

            for quote in quotes:
                if len(quote['quote']) > 1:
                    self.quotes.append(AnnotatedSpan(chap_id=chap_id, para_id=para_id, start_token_id=quote['start_paragraph_token_id'], end_token_id=quote['end_paragraph_token_id'], annotation=character, text=quote['quote']))

        if save_dirpath:
            self.pickle_output(save_dirpath, self.quotes)

    def extract_character_mentions(self, save_dirpath=None):
        """ Extracts character mentions, saves in self.character_mentions, also in save_dirpath if specified """
        
        self.character_mentions = []
        coref_fic = self.load_coref_csv()

        for row in list(coref_fic.itertuples()):
            mentions = extract_mention_tags(row.text_tokenized)
            for mention in mentions:
                mention.chap_id = row.chapter_id
                mention.para_id = row.para_id
                self.character_mentions.append(mention)

        if save_dirpath:
            self.pickle_output(save_dirpath, self.character_mentions)

def extract_pipeline_entity_mentions(text):
    """ DEPRECATED """
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
