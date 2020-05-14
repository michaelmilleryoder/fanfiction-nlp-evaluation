""" Utility functions for coref evaluation on different systems """

import pandas as pd
import os
import pickle
import itertools
import numpy as np
from string import punctuation
import re
import pdb


def load_pickle(dirpath, fandom_fname):
    fpath = os.path.join(dirpath, f'{fandom_fname}.pkl')
    with open(fpath, 'rb') as f:
        return pickle.load(f)


def sublist_indices(sl,l):
    """ https://stackoverflow.com/questions/17870544/find-starting-and-ending-indices-of-sublist-in-list """
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll))

    return results


def fic_stats(fic_dirpath, fname, gold_entities):
    # Length of fic

    fic = pd.read_csv(os.path.join(fic_dirpath, fname))
    n_words = sum(fic['text_tokenized'].map(lambda x: len(x.split())))
    print(f'{fname}: {n_words}')

    # Number of characters
    if fname in gold_entities:
        print(f'{fname}: {len(gold_entities[fname])}')


def remove_character_tags(text):
    # Remove character parentheses
    modified_para = re.sub(r'\(\$_.*?\)\ ', '', text)

    # Split up character underscore mentions
    modified_para = re.sub(r'([^ ])_([^ ])', r'\1 \2', modified_para)
    modified_para = re.sub(r'([^ ])_([^ ])', r'\1 \2', modified_para) # twice

    return modified_para


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


def span_in_paragraph(span, paragraph):
    """ Measure overlap between tokens in a span and a paragraph
        to make sure the span is contained in the paragraph """

    threshold = .3 # proportion of tokens in the span that have to be present in the paragraph

    processed_span = preprocess_quote(span)
    processed_para = preprocess_quote(paragraph)
    matches = processed_span.intersection(processed_para)

    if len(processed_span) == 0:
        return True

    return len(matches)/len(processed_span) >= threshold


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



if __name__ == '__main__': main()
