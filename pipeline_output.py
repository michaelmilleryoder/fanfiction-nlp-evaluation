""" Class for loading, storing and processing output of pipeline """

import os
import pickle
import json
import pandas as pd
import re
import pdb

from fic_representation import FicRepresentation
import evaluation_utils as utils
from annotated_span import AnnotatedSpan, group_annotations_para
from character_mention_parser import CharacterMentionParser
from annotation import Annotation


def extract_mention_tags(text):
    """ Returns AnnotatedSpan objects with start and end token IDs annotated with character """
    parser = CharacterMentionParser()
    parser.feed(text)
    return parser.character_mentions


def strip_mention_tags(text):
    """ Returns text without any <tags> """
    parser = CharacterMentionParser()
    parser.feed(text)
    return parser.text


def strip_quotes(text):
    """ Returns text with any quote marks replaced by underscores """
    quote_marks = ['“', '``', '"', '«', '”', "''", '»']
    new_toks = [tok if not tok in quote_marks else '_' for tok in text.split()]
    return ' '.join(new_toks)


def insert_mention_tags(text, mentions):
    """ Returns text without annotation tags inserted around corresponding tokens """
    tokens = text.split()
    for span in mentions:
        if span.start_token_id - 1 >= len(tokens) or span.start_token_id - 1 < 0:
            pdb.set_trace()
        tokens[span.start_token_id-1] = f'<character name="{span.annotation}">' + tokens[span.start_token_id-1]
        tokens[span.end_token_id-1] = tokens[span.end_token_id-1] + "</character>"
    return ' '.join(tokens)


def insert_quote_marks(text, mentions):
    """ Returns text with readable quote marks around annotated spans """
    tokens = text.split()
    for span in mentions:
        start_tok = tokens[span.start_token_id-1]
        tokens[span.start_token_id-1] = '``'
        #if re.search(r'[\w\.]', start_tok): # if has anything other than punctuation
        #    tokens[span.start_token_id] = f'{start_tok}_{tokens[span.start_token_id]}'
        #else:
        #    tokens[span.start_token_id-1] = '``'
        end_tok = tokens[span.end_token_id-1]
        tokens[span.end_token_id-1] = "''"
        #if re.search(r'[\w\.]', end_tok): # if has anything other than punctuation
        #    tokens[span.end_token_id-2] = f'{tokens[span.end_token_id-2]}_{end_tok}'
        #tokens[span.end_token_id-1] = "''"
    return ' '.join(tokens)


class PipelineOutput(FicRepresentation):
    """ Holds representation for the pipeline-processed output of a fic. """

    def __init__(self, output_dirpath, fandom_fname, fic_csv_dirpath=None, modified_suffix=''): 
        super().__init__(fandom_fname)
        self.output_dirpath = output_dirpath
        self.coref_output_dirpath = os.path.join(output_dirpath, 'char_coref_stories' + modified_suffix)
        self.coref_chars_output_dirpath = os.path.join(output_dirpath, 'char_coref_chars' + modified_suffix)
        self.quote_output_dirpath = os.path.join(output_dirpath, 'quote_attribution' + modified_suffix)
        self.fic_csv_dirpath = fic_csv_dirpath

    def load_quote_json(self):
        """ Returns pipeline output quotes from json """

        quote_predictions_fpath = os.path.join(self.quote_output_dirpath, f'{self.fandom_fname}.quote.json')

        with open(quote_predictions_fpath) as f:
            predicted_quotes = json.load(f)

        return predicted_quotes

    def load_coref_csv(self):
        """ Loads a pandas DataFrame of CSV with coref <tags> to self.coref_fic """

        self.coref_fic = pd.read_csv(os.path.join(self.coref_output_dirpath, f'{self.fandom_fname}.coref.csv'))

    def save_coref_csv(self):
        """ Saves a pandas DataFrame of CSV with coref <tags> """

        outpath = os.path.join(self.coref_output_dirpath, f'{self.fandom_fname}.coref.csv')
        if not os.path.exists(self.coref_output_dirpath):
            os.mkdir(self.coref_output_dirpath)
        #print(f"Wrote gold coref csv file to {outpath}")
        self.coref_fic.to_csv(outpath, index=False)

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
        self.load_coref_csv()

        for row in list(self.coref_fic.itertuples()):
            mentions = extract_mention_tags(row.text_tokenized)
            for mention in mentions:
                mention.chap_id = row.chapter_id
                mention.para_id = row.para_id
                self.character_mentions.append(mention)

        if save_dirpath:
            self.pickle_output(save_dirpath, self.character_mentions)

    def save_characters_file(self):
        """ Saves pipeline character coref file out,
            assumes the output dirpath has been modified
        
            Args:
                suffix: suffix for directory path
        """
        characters = set([span.annotation for span in self.character_mentions])
        if not os.path.exists(self.coref_chars_output_dirpath):
            os.mkdir(self.coref_chars_output_dirpath)
        outpath = os.path.join(self.coref_chars_output_dirpath, f'{self.fandom_fname}.chars')
        with open(outpath, 'w') as f:
            for char in characters:
                f.write(f'{char}\n')

    def modify_coref_tags(self, annotations):
        """ Modifies self.coref_fic with <tags>.
            Args:
                annotations: new character coreference annotations
        """

        # Group annotations by (chap_id, para_id)        
        grouped = group_annotations_para(annotations)
        if not hasattr(self, 'coref_fic'):
            self.load_coref_csv()
        new_text_tokenized = []
        for row in list(self.coref_fic.itertuples()):
            # Strip any existing character tags
            text = strip_mention_tags(row.text_tokenized)            
            # Insert new character tags
            if (row.chapter_id, row.para_id) in grouped:
                new_text_tokenized.append(insert_mention_tags(text, grouped[(row.chapter_id, row.para_id)]))
            else:
                new_text_tokenized.append(text)
        self.coref_fic['text_tokenized'] = new_text_tokenized

    def modify_quote_marks(self, annotations):
        """ Modifies self.coref_fic with quote marks around gold quotes.
            Args:
                annotations: new quote spans
        """

        # Group annotations by (chap_id, para_id)        
        grouped = group_annotations_para(annotations)
        if not hasattr(self, 'coref_fic'):
            self.load_coref_csv()
        new_text_tokenized = []
        for row in list(self.coref_fic.itertuples()):
            # Strip any existing quotes (replace with underscores)
            #text = strip_quotes(row.text_tokenized)            
            text = row.text_tokenized
            # Insert readable quote marks
            if (row.chapter_id, row.para_id) in grouped:
                new_text_tokenized.append(insert_quote_marks(text, grouped[(row.chapter_id, row.para_id)]))
            else:
                new_text_tokenized.append(text)
        self.coref_fic['text_tokenized'] = new_text_tokenized

    def modify_quote_spans(self, quote_annotations_dirpath, quote_annotations_ext):
        """ Modifies quote marks so that the pipeline will recognized
            gold quotes as quote spans """
        # Load gold quote extractions
        gold = Annotation(quote_annotations_dirpath, self.fandom_fname, file_ext=quote_annotations_ext, fic_csv_dirpath=self.fic_csv_dirpath)
        gold.extract_annotated_spans()

        # Modify CSV text_tokenized
        self.modify_quote_marks(gold.annotations) # Modifies self.coref_csv

        # Save out
        modify_text = '_gold_quotes'
        self.coref_output_dirpath = self.coref_output_dirpath.rstrip('/') + modify_text
        self.save_coref_csv()

        # Change characters file path, too
        self.coref_chars_output_dirpath = self.coref_chars_output_dirpath.rstrip('/') + modify_text

        return modify_text

    def modify_coref_files(self, coref_annotations_dirpath, coref_annotations_ext):
        """ Changes coref tokens to gold annotations in self.token_data.
            Saves out to {token_output_dirpath}_gold_coref/token_fpath
            Returns the suffix added to dirpaths
        """
        # Load gold mentions, place in self.character_mentions
        gold = Annotation(coref_annotations_dirpath, self.fandom_fname, file_ext=coref_annotations_ext, fic_csv_dirpath=self.fic_csv_dirpath)
        gold.extract_annotated_spans()
        self.character_mentions = gold.annotations

        # Modify coref <tags> in CSV
        self.modify_coref_tags(gold.annotations) # Modifies self.coref_csv

        # Save out
        modify_text = '_gold_coref'
        self.coref_output_dirpath = self.coref_output_dirpath.rstrip('/') + modify_text
        self.save_coref_csv()

        # Modify coref characters file
        self.coref_chars_output_dirpath = self.coref_chars_output_dirpath.rstrip('/') + modify_text
        self.save_characters_file()
        
        return modify_text
