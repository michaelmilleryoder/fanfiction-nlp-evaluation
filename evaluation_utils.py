""" Utility functions for coref evaluation on different systems """

import pandas as pd
import os
import itertools
import numpy as np
from string import punctuation
import re
import pdb

from quote import Quote

def match_extracted_quotes(predicted_quotes, gold_quotes):
    """ Match Quote objects just on extracted spans (ignoring speaker assignments) """

    matched_gold = []
    matched_predicted = []
    false_positives = []
    false_negatives = []

    # Could perhaps redo with sets, intersections if can specify match functions
    for pred_quote in predicted_quotes:

        match = None
        for gold_quote in gold_quotes:

            if gold_quote.extraction_matches(pred_quote):
                match = gold_quote
                break

        if match is not None:
            matched_gold.append(gold_quote)
            matched_predicted.append(pred_quote)
        else:
            false_positives.append(pred_quote)
        
    for gold_quote in gold_quotes:
        if not gold_quote in matched_gold:
            false_negatives.append(gold_quote)

    assert len(matched_gold) == len(matched_predicted)

    return matched_gold, matched_predicted, false_positives, false_negatives


def match_quote_attributions(predicted_quotes, gold_quotes, matched=False, incorrect_extractions=[]):
    """ Match Quote objects entirely, including speaker attribution if character names match """

    correct_attributions = []

    # Check extractions
    if not matched:
        matched_gold, matched_predicted, incorrect_extractions, _ = self.match_extracted_quotes(predicted_quotes, gold_quotes)
    else:
        matched_gold, matched_predicted = gold_quotes, predicted_quotes

    incorrect_attributions = incorrect_extractions

    # Find matched gold quote
    for pred_quote, gold_quote in zip(matched_predicted, matched_gold):
    
        # Check attribution
        if characters_match(pred_quote.speaker, gold_quote.speaker):
            correct_attributions.append(pred_quote)
        else:
            incorrect_attributions.append((pred_quote, gold_quote))

    return correct_attributions, incorrect_attributions


def print_quote_scores(predicted_quotes, gold_quotes):
    """ Prints quote extraction and attribution scores """

    # Precision, recall of the quote extraction (the markables)
    matched_gold_quotes, matched_pred_quotes, false_positives, false_negatives = match_extracted_quotes(predicted_quotes, gold_quotes)
    if len(predicted_quotes) == 0:
        precision = 0
    else:
        precision = len(matched_pred_quotes)/len(predicted_quotes)
    recall = len(matched_gold_quotes)/len(gold_quotes)
    f1 = f_score(precision, recall)
    print(f'\tExtraction results:')
    print(f'\t\tPrecision: {precision: .2%} ({len(matched_pred_quotes)}/{len(predicted_quotes)})')
    print(f'\t\tRecall: {recall: .2%} ({len(matched_gold_quotes)}/{len(gold_quotes)})')
    print(f'\t\tF-score: {f1: .2%}')

    # Quote attribution accuracy on matched quotes
    correct_attributions, incorrect_attributions = match_quote_attributions(matched_pred_quotes, matched_gold_quotes, matched=True, incorrect_extractions=false_positives)
    if len(matched_pred_quotes) == 0:
        attribution_accuracy_matched = 0
    else:
        attribution_accuracy_matched = len(correct_attributions)/len(matched_pred_quotes)
    print(f'\tAttribution accuracy:')
    print(f'\t\tOn matched quote spans: {attribution_accuracy_matched: .2%} ({len(correct_attributions)}/{len(matched_pred_quotes)})')

    # Quote attribution accuracy on all predicted quotes.
    # If the predicted quote is not a real quote span, is not a match
    if len(predicted_quotes) == 0:
        attribution_accuracy_all = 0
    else:
        attribution_accuracy_all = len(correct_attributions)/len(predicted_quotes)
    print(f'\t\tOn all predicted quote spans: {attribution_accuracy_all: .2%} ({len(correct_attributions)}/{len(predicted_quotes)})')

    print()

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
    if precision + recall == 0:
        return 0
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


#TODO: should go in an Annotation class
def extract_gold_spans(annotations_dirpath, fname):
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


#TODO: should go in an Annotation class
def gold_quotes(annotations_dirpath, fandom_fname):
    """ Return Quote objects for annotations """

    gold_quotes = []
    gold_entities = extract_gold_spans(annotations_dirpath, f'{fandom_fname}_quote_attribution.csv')
    
    for speaker, entities in gold_entities.items():
        for chap_id, para_id, start_token, end_token in entities:
            gold_quotes.append(Quote(chap_id, para_id, start_token, end_token, speaker)) 

    return gold_quotes


def gold_quote_entries(annotations_dirpath, fic_dirpath, fandom_fname):
    """ Return gold quote entries of form  (chap_id, para_id): (speaker, quote) """

    gold_entities = extract_gold_spans(annotations_dirpath, f'{fandom_fname}_quote_attribution.csv')
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



if __name__ == '__main__': main()
