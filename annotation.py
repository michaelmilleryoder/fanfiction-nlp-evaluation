""" Classes for loading and storing annotations of coreference and quotes """

import os
import pandas as pd

from quote import Quote
from fic_representation import FicRepresentation
import evaluation_utils as utils


class Annotation(FicRepresentation):
    
    def __init__(self, annotations_dirpath, fandom_fname, file_ext='.csv', fic_csv_dirpath=None):
        super().__init__(fandom_fname, fic_csv_dirpath=fic_csv_dirpath)
        self.file_path = os.path.join(annotations_dirpath, f'{fandom_fname}{file_ext}')

    def extract_gold_spans(self):
        # Load ground-truth annotated entity mentions or quote spans, attributed to each character

        gold_entities = {} # cluster_name: {(chapter_id, paragraph_id, token_id_start, token_id_end), ...}

        df = pd.read_csv(self.file_path)
        for colname in df.columns:
            gold_entities[colname] = set()
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
                    
                gold_entities[colname].add((chapter_id, paragraph_id, token_id_start, token_id_end))

        return gold_entities


class QuoteAnnotation(Annotation):
    """ Loads and holds quote annotations for a fic. """
    
    def __init__(self, annotations_dirpath, fandom_fname, fic_csv_dirpath=None):
        """ Loads annotations for a fic """
        super().__init__(annotations_dirpath, fandom_fname, file_ext='_quote_attribution.csv', fic_csv_dirpath=fic_csv_dirpath)
        self.load_quotes() # Stores in self.quotes

    def load_quotes(self):
        """ Saves annotated Quote objects in self.quotes"""

        self.quotes = []
        gold_entities = self.extract_gold_spans()

        if self.fic_csv is None:
            self.load_fic_csv()
        fic_data = self.fic_csv.set_index(['chapter_id', 'para_id'], inplace=False)
        para_tokens = fic_data['text_tokenized'].str.split().to_dict() # (chap_id, para_id): tokens
        
        for speaker, entities in gold_entities.items():
            for chap_id, para_id, start_token, end_token in entities:

                quote_text = ' '.join(para_tokens[(chap_id, para_id)][start_token-1:end_token])
                self.quotes.append(Quote(chap_id, para_id, start_token, end_token, speaker, text=quote_text)) 
