""" Class for loading, storing and processing output of pipeline """

import os
import json
import pandas as pd
import pdb

from fic_representation import FicRepresentation
import evaluation_utils as utils
from quote import Quote


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

    def extract_quotes(self):
        """ Extracts quotes into Quote objects, saves in self.quotes """

        self.quotes = [] # list of Quote objects
        
        predicted_quotes_json = self.load_quote_json()

        for quote_entry in predicted_quotes_json:
            
            character = quote_entry['speaker']
            chap_id = quote_entry['chapter']
            para_id = quote_entry['paragraph']
            quotes = quote_entry['quotes']

            for quote in quotes:
                if len(quote['quote']) > 1:
                    self.quotes.append(Quote(chap_id, para_id, quote['start_paragraph_token_id'], quote['end_paragraph_token_id'], character, text=quote['quote']))
