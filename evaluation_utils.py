""" Utility functions for coref evaluation on different systems """

import pandas as pd
import os
import itertools
import numpy as np
from string import punctuation
import re
import pdb

from quote import Quote

def match_extracted_quotes(predicted_quotes, gold_quotes, exact=True):
    """ Match Quote objects just on extracted spans (ignoring speaker assignments).
        Args:
            predicted_quotes: Quote objects predicted
            gold_quotes: Quote objects annotated as gold truth
            exact: whether an exact match on token IDs is necessary.
                For the FanfictionNLP pipeline, this should be the case.
                For baseline systems that might have different tokenization,
                this can be set to False to relax that constraint.
    """

    matched_gold = []
    matched_predicted = []
    false_positives = []
    false_negatives = []

    matched = [(predicted, gold) for predicted, gold in itertools.product(predicted_quotes, gold_quotes) if gold.extraction_matches(predicted, exact=exact)]
    if len(matched) == 0:
        matched_predicted, matched_gold = [], []
    else:
        matched_predicted, matched_gold = list(zip(*matched))

    false_positives = [predicted for predicted in predicted_quotes if not predicted in matched_predicted]
    false_negatives = [gold for gold in gold_quotes if not gold in matched_gold]

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
            correct_attributions.append((pred_quote, gold_quote))
        else:
            incorrect_attributions.append((pred_quote, gold_quote))

    return correct_attributions, incorrect_attributions


def print_quote_scores(predicted_quotes, gold_quotes, exact_match=True):
    """ Prints quote extraction and attribution scores """

    # Precision, recall of the quote extraction (the markables)
    matched_gold_quotes, matched_pred_quotes, false_positives, false_negatives = match_extracted_quotes(predicted_quotes, gold_quotes, exact=exact_match)
    if len(predicted_quotes) == 0:
        precision = 0
    else:
        precision = len(matched_pred_quotes)/len(predicted_quotes)
    recall = len(matched_gold_quotes)/len(gold_quotes)
    f1 = f_score(precision, recall)
    print(f'\tExtraction results:')
    print(f'\t\tF-score: {f1: .2%}')
    print(f'\t\tPrecision: {precision: .2%} ({len(matched_pred_quotes)}/{len(predicted_quotes)})')
    print(f'\t\tRecall: {recall: .2%} ({len(matched_gold_quotes)}/{len(gold_quotes)})')

    # Quote attribution accuracy on matched quotes
    correct_attributions, incorrect_attributions = match_quote_attributions(matched_pred_quotes, matched_gold_quotes, matched=True, incorrect_extractions=false_positives)
    if len(matched_pred_quotes) == 0:
        attribution_accuracy_matched = 0
    else:
        attribution_accuracy_matched = len(correct_attributions)/len(matched_pred_quotes)
    print(f'\tAttribution results:')

    # Quote attribution accuracy on all predicted quotes.
    # If the predicted quote is not a real quote span, is not a match
    if len(predicted_quotes) == 0:
        attribution_precision = 0
    else:
        attribution_precision = len(correct_attributions)/len(predicted_quotes)
    attribution_recall = len(correct_attributions)/len(gold_quotes)
    attribution_f1 = f_score(attribution_precision, attribution_recall)
    print(f'\t\tF-score: {attribution_f1: .2%}')
    print(f'\t\tPrecision: {attribution_precision: .2%} ({len(correct_attributions)}/{len(predicted_quotes)})')
    print(f'\t\tRecall: {attribution_recall: .2%} ({len(correct_attributions)}/{len(gold_quotes)})')

    print(f'\t\tAccuracy on matched quote spans: {attribution_accuracy_matched: .2%} ({len(correct_attributions)}/{len(matched_pred_quotes)})')

    print()

    #pdb.set_trace()

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
