""" Functions for evaluating pipeline quote attribution """

import os
import json
import pandas as pd
import pdb

import evaluation_utils as utils
from quote import Quote


def load_pipeline_quote_json(quote_predictions_dirpath, fandom_fname):
    quote_predictions_fpath = os.path.join(quote_predictions_dirpath, f'{fandom_fname}.quote.json')

    with open(quote_predictions_fpath) as f:
        predicted_quotes = json.load(f)

    return predicted_quotes


def pipeline_quote_entries(quote_predictions_dirpath, fic_dirpath, fandom_fname):

    #predicted_quote_entries = {} # (chap_id, para_id): [(character, quote_text)]
    predicted_quotes = [] # list of Quote objects
    
    predicted_quotes_json = load_pipeline_quote_json(quote_predictions_dirpath, fandom_fname)

    # Load fic for paragraph alignment check
    #fic_fpath = os.path.join(fic_dirpath, f'{fandom_fname}.csv')
    #fic = pd.read_csv(fic_fpath)

    for quote_entry in predicted_quotes_json:
        
        character = quote_entry['speaker']
        chap_id = quote_entry['chapter']
        para_id = quote_entry['paragraph']
        quotes = quote_entry['quotes']

        #fic_para = fic.loc[fic['para_id']==para_id, 'text_tokenized'].tolist()[0]
        
        for quote in quotes:
            if len(quote['quote']) > 1:
                #if not (chap_id, para_id) in predicted_quote_entries:
                #    predicted_quote_entries[(chap_id, para_id)] = []

                # Check fic paragraph to make sure quote is present
                #if not utils.span_in_paragraph(quote['quote'], fic_para):
                #    pdb.set_trace()

                #predicted_quote_entries[(chap_id, para_id)].append((character, quote['quote']))

                predicted_quotes.append(Quote(chap_id, para_id, quote['start_paragraph_token_id'], quote['end_paragraph_token_id'], character, text=quote['quote']))
    
    return predicted_quotes
