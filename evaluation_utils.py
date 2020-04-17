""" Utility functions for coref evaluation on different systems """

import pandas as pd
import os
import itertools
import numpy as np
from string import punctuation
import re
import pdb

from quote import Quote

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
    
    cluster_resolutions = {}
    cluster_sizes = {}
    
    for gold_cluster, gold_mentions in gold_entities.items():
        gold_links = links(gold_mentions)
        
        cluster_resolution = 0
        
        for predicted_cluster, predicted_mentions in predicted_entities.items():
            predicted_links = links(predicted_mentions)
            
            cluster_resolution += len(predicted_links.intersection(gold_links))
            
        cluster_resolution = cluster_resolution/len(gold_links)
        cluster_resolutions[gold_cluster] = cluster_resolution
        cluster_sizes[gold_cluster] = len(gold_mentions)
        
    # take importance (size) of clusters into account
    fic_recall = sum([cluster_sizes[c] * cluster_resolutions[c] for c in gold_entities])/sum(cluster_sizes.values())
        
    # Total recall as mean across fics
    #total_recall = np.mean(list(fic_recalls.values()))

    return fic_recall


def lea_precision(predicted_entities, gold_entities):
    
    cluster_resolutions = {}
    cluster_sizes = {}
    
    for predicted_cluster, predicted_mentions in predicted_entities.items():
        predicted_links = links(predicted_mentions)
        
        cluster_resolution = 0
        
        for gold_cluster, gold_mentions in gold_entities.items():
            gold_links = links(gold_mentions)
            cluster_resolution += len(predicted_links.intersection(gold_links))
        
        cluster_resolution = cluster_resolution/len(predicted_links)
        cluster_resolutions[predicted_cluster] = cluster_resolution
        cluster_sizes[predicted_cluster] = len(predicted_mentions)
        
    # take importance (size) of clusters into account
    fic_precision = sum([cluster_sizes[c] * cluster_resolutions[c] for c in predicted_entities])/sum(cluster_sizes.values())
        
    # Total precision as mean across fics
    #total_precision = np.mean(list(fic_precisions.values()))

    return fic_precision


def f_score(precision, recall):
    return 2 * (precision * recall)/(precision + recall)


def calculate_lea(predicted_entities, gold_entities):
    # # Calculate LEA coreference evaluation

    #fics = set(predicted_entities.keys()).intersection(set(gold_entities.keys()))

    #selected_predicted = {fic: predicted_entities[fic] for fic in fics}
    #selected_gold = {fic: gold_entities[fic] for fic in fics}

    recall = lea_recall(predicted_entities, gold_entities)
    precision = lea_precision(predicted_entities, gold_entities)
    f1 = f_score(precision, recall)

    print(f"\t\tPrecision: {precision: .2%}")
    print(f"\t\tRecall: {recall: .2%}")
    print(f"\t\tF-score: {f1: .2%}")


def extract_gold_character_spans(annotations_dirpath, fname):
    # Load ground-truth annotated entity mentions or quote spans, attributed to each character

    gold_entities = {} # cluster_name: {(chapter_id, paragraph_id, token_id_start, token_id_end), ...}

    df = pd.read_csv(os.path.join(annotations_dirpath, fname))
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


def extract_gold_quotes(gold_entities, fic_dirpath, fandom_fname):

    gold_quotes = {}
    
    # Load actual text
    fic_data = pd.read_csv(os.path.join(fic_dirpath, f'{fandom_fname}.csv'))
    fic_data.set_index(['chapter_id', 'para_id'], inplace=True)
    para_tokens = fic_data['text_tokenized'].str.split().to_dict() # (chap_id, para_id): tokens

    for char, spans in sorted(gold_entities.items()):
        gold_quotes[char] = list()
        
        for span in sorted(spans):
            chap_id, para_id, start_token_id, end_token_id = span # note that token IDs are 0-start, annotations 1-span
            quote_text = ' '.join(para_tokens[(chap_id, para_id)][start_token_id-1:end_token_id])
            gold_quotes[char].append((chap_id, para_id, quote_text))
            
    return gold_quotes


def gold_quotes(annotations_dirpath, fandom_fname):
    """ Return Quote objects for annotations """

    gold_quotes = []
    gold_entities = extract_gold_character_spans(annotations_dirpath, f'{fandom_fname}_quote_attribution.csv')
    
    for speaker, entities in gold_entities.items():
        for chap_id, para_id, start_token, end_token in entities:
            gold_quotes.append(Quote(chap_id, para_id, start_token, end_token, speaker)) 

    return gold_quotes


def match_extracted_quotes(gold_quotes, predicted_quotes):
    """ Match Quote objects just on extracted spans (ignoring speaker assignments) """

    matches = []
    false_positives = []
    false_negatives = []

    for gold_quote in gold_quotes:
        match = None
        for pred_quote in predicted_quotes:
            if gold_quote.extraction_matches(pred_quote):
                match = pred_quote
                break

        if match is not None:
            matches.append(pred_quote)
        else:
            false_negatives.append(gold_quote)

    for pred_quote in predicted_quotes:
        if not pred_quote in matches:
            false_positives.append(pred_quote)

    return matches, false_positives, false_negatives


def match_quote_attributions(gold_quotes, predicted_quotes):
    """ Match Quote objects entirely, including speaker attribution if character names match """

    correct_attributions = []
    incorrect_attributions = []

    # Check extractions
    matched_predicted, incorrect_attributions, _ = match_extracted_quotes(gold_quotes, predicted_quotes)

    # Find matched gold quote
    for pred_quote in matched_predicted:
        for gold_quote in gold_quotes:
            if gold_quote.extraction_matches(pred_quote):
                matched_quote = gold_quote
                break
    
        # Check attribution
        if characters_match(pred_quote.speaker, gold_quote.speaker):
            correct_attributions.append(pred_quote)
        else:
            incorrect_attributions.append((pred_quote, gold_quote))

    return correct_attributions, incorrect_attributions


def gold_quote_entries(annotations_dirpath, fic_dirpath, fandom_fname):
    """ Return gold quote entries of form  (chap_id, para_id): (speaker, quote) """

    gold_entities = extract_gold_character_spans(annotations_dirpath, f'{fandom_fname}_quote_attribution.csv')
    gold_quotes = extract_gold_quotes(gold_entities, fic_dirpath, fandom_fname)

    chap_id = 1 # TODO: handle multiple chapter IDs

    gold_quote_entries = {} # (chap_id, para_id): [(character, quote_text)]

    for character, quote_entries in gold_quotes.items():
        
        for chap_id, para_id, quote in quote_entries:
            
            if not (chap_id, para_id) in gold_quote_entries:
                gold_quote_entries[(chap_id, para_id)] = []
                    
            gold_quote_entries[(chap_id, para_id)].append((character, quote))

    return gold_quote_entries


def characters_match(predicted_char, gold_char):
    """ If any parts of the predicted character matches any part of the gold character (fairly lax) """
    
    predicted_char_parts = predicted_char.lower().split('_')
    gold_char_parts = [re.sub(r'[\(\)]', '', part) for part in gold_char.lower().split(' ')]
    
    match = False
    
    for pred_part in predicted_char_parts:
        for gold_part in gold_char_parts:
            if pred_part == gold_part:
                match = True
                
    return match


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


def preprocess_quote(quote):
    """ Return set of lowercased unique tokens from a quote. """

    # Remove ccc_ business
    #processed_quote = quote.replace('ccc_', '')
    #processed_quote = processed_quote.replace('_ccc', '')
    processed_quote = re.sub(r'ccc_.*?_ccc', '', quote)

    # Remove punctuation, lowercase
    stops = list(punctuation) + ['”', '“']
    processed_quote = ''.join([c for c in processed_quote.lower() if not c in stops])

    # Replace whitespace with spaces
    #processed_quote = re.sub(r'\s+', ' ', processed_quote)
    
    # Extract unique words
    processed_words = set(processed_quote.strip().split())

    return processed_words


def quotes_match(quotes):
    
    processed_quotes = []

    word_match_threshold = .5
    
    for quote in quotes:

        #if 'No' in quote:
        #    pdb.set_trace()

        processed_words = preprocess_quote(quote)
        processed_quotes.append(processed_words)
        
    # Measure unique word overlap
    n_matches = len(processed_quotes[0].intersection(processed_quotes[1]))
    if len(processed_quotes[1]) == 0:
        if len(processed_quotes[0]) < 4: # Probably just the name of a character
            return True
        else:
            return False

    return (n_matches/len(processed_quotes[1])) >= word_match_threshold


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


def compare_quote_entries(predicted, gold):
    
    matched_extracted_quotes = []
    unmatched_extracted_quotes = []
    
    for chap_id, para_id in predicted:
        
        if not (chap_id, para_id) in gold: # didn't extract any quotes in that paragraph
            unmatched_extracted_quotes += [(chap_id, para_id, speaker, quote) for speaker, quote in predicted[(chap_id, para_id)]]
            continue
            
        gold_para_entries = gold[(chap_id, para_id)]
        
        for predicted_character, predicted_quote in predicted[(chap_id, para_id)]:
            match = False
            matched_gold = None
            
            # Search for match
            for gold_character, gold_quote in gold_para_entries:
                if quotes_match((predicted_quote, gold_quote)) and characters_match(predicted_character, gold_character):
                    match = True
                    matched_gold = (chap_id, para_id, gold_character, gold_quote)
                    break
                    
            if match:
                matched_extracted_quotes.append(((chap_id, para_id, predicted_character, predicted_quote),  matched_gold))
            else:
                unmatched_extracted_quotes.append((chap_id, para_id, predicted_character, predicted_quote))
                
    return matched_extracted_quotes, unmatched_extracted_quotes


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
