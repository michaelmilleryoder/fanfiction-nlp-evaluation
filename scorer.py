""" Holds scoring functions for coref and quotes """

import pdb
import numpy as np
import itertools
from sklearn.metrics import cohen_kappa_score

from annotated_span import AnnotatedSpan, group_annotations, match_spans, match_annotated_spans, normalize_annotations_to_id


def score_attribution_labels(matched_labels, mismatched_labels, mismatched_extraction_labels, metric='cohen'):
    """ Scores annotation labels for lists of tuples:
            matched_labels: (span1, span2) for spans that match and have the same annotation
            mismatched_labels: (span1, span2) for spans that match but do not have the same annotation
            mismatched_extraction_labels: (span1, span2) for spans that match but one has a NULL label (wasn't extracted)

        Returns score_on_matched_extractions, score_on_all_extractions
    """

    if len(matched_labels) == 0:
        if len(mismatched_labels) == 0 and len(mismatched_extraction_labels) == 0: # nothing predicted
            return (1.0,1.0)
        else:
            return (0.0,0.0)

    # Build corresponding lists of attributions
    matched_labels1 = list(list(zip(*matched_labels))[0])
    matched_labels2 = list(list(zip(*matched_labels))[1])
    if len(mismatched_labels) == 0:
        mismatched_labels1 = []
        mismatched_labels2 = []
    else:
        mismatched_labels1 = list(list(zip(*mismatched_labels))[0])
        mismatched_labels2 = list(list(zip(*mismatched_labels))[1])
    if len(mismatched_extraction_labels) == 0:
        mismatched_ext_labels1 = []
        mismatched_ext_labels2 = []
    else:
        mismatched_ext_labels1 = list(list(zip(*mismatched_extraction_labels))[0])
        mismatched_ext_labels2 = list(list(zip(*mismatched_extraction_labels))[1])

    # Calculate attribution agreement on matched extractions
    ext_labels1 = matched_labels1 + mismatched_labels1
    ext_labels2 = matched_labels2 + mismatched_labels2
    
    # Normalize character names to IDs
    annotations1 = matched_labels1 + mismatched_labels1 + mismatched_ext_labels1
    annotations2 = matched_labels2 + mismatched_labels2 + mismatched_ext_labels2
    char2id = normalize_annotations_to_id(annotations1, annotations2)

    vals1 = [char2id[span.annotation] for span in ext_labels1]
    vals2 = [char2id[span.annotation] for span in ext_labels2]
    score_on_matched_extractions = cohen_kappa_score(vals1, vals2)

    # Calculate attribution agreement on all extractions
    labels1 = matched_labels1 + mismatched_labels1 + mismatched_ext_labels1
    labels2 = matched_labels2 + mismatched_labels2 + mismatched_ext_labels2
    vals1 = [char2id[span.annotation] for span in labels1]
    vals2 = [char2id[span.annotation] for span in labels2]
    score_on_all_extractions = cohen_kappa_score(vals1, vals2)

    return score_on_matched_extractions, score_on_all_extractions


def span_attribution_confusion_matrix(baseline_spans, experimental_spans, gold_spans):
        a = 0 # Both correct
        b = 0 # Baseline correct, experiment incorrect
        c = 0 # Baseline incorrect, experiment correct
        d = 0 # Both incorrect
        for baseline_quote, experimental_quote, gold_quote in zip(self.ordered_predictions['quotes']['baseline'], self.ordered_predictions['quotes']['experimental'], self.ordered_predictions['quotes']['gold']):
            if utils.characters_match(baseline_quote.speaker, gold_quote.speaker) and utils.characters_match(experimental_quote.speaker, gold_quote.speaker):
                a += 1
            elif utils.characters_match(baseline_quote.speaker, gold_quote.speaker) and not utils.characters_match(experimental_quote.speaker, gold_quote.speaker):
                b += 1
            elif not utils.characters_match(baseline_quote.speaker, gold_quote.speaker) and utils.characters_match(experimental_quote.speaker, gold_quote.speaker):
                c += 1
            else:
                d += 1
                
        table = [[a, b],
                 [c, d]]


def links(mention_cluster):
    """ Returns a set of all the links in an entity between lists of AnnotatedSpans 
        Links are tuples (chap_id, start_token_id, end_token_id)
    """
    
    if len(mention_cluster) == 1: # self-link
        links = {mention_cluster[0].get_location(), mention_cluster[0].get_location()}

    else:
        links = set([(mention1.get_location(), mention2.get_location()) for mention1, mention2 in itertools.combinations(mention_cluster, 2)])
        
    return links


def lea_recall(predicted_clusters, gold_clusters):
    """ Calculates LEA recall between predicted mention clusters and gold mention clusters.
        Input clusters are of form {'annotation': [AnnotatedSpan, ...]}
    """
    
    cluster_resolutions = {}
    cluster_sizes = {}
    
    for gold_cluster_name, gold_mentions in gold_clusters.items():
        gold_links = links(gold_mentions)
        
        cluster_resolution = 0
        
        for predicted_cluster, predicted_mentions in predicted_clusters.items():
            predicted_links = links(predicted_mentions)
            
            cluster_resolution += len(predicted_links.intersection(gold_links))
            
        cluster_resolution = cluster_resolution/len(gold_links)
        cluster_resolutions[gold_cluster_name] = cluster_resolution
        cluster_sizes[gold_cluster_name] = len(gold_mentions)
        
    # take importance (size) of clusters into account
    if sum(cluster_sizes.values()) == 0: # no predicted clusters
        fic_recall = 0
    else:
        fic_recall = sum([cluster_sizes[c] * cluster_resolutions[c] for c in gold_clusters])/sum(cluster_sizes.values())

    return fic_recall


def lea_precision(predicted_clusters, gold_clusters):
    
    cluster_resolutions = {}
    cluster_sizes = {}
    
    for predicted_cluster_name, predicted_mentions in predicted_clusters.items():
        predicted_links = links(predicted_mentions)
        
        cluster_resolution = 0
        
        for gold_cluster, gold_mentions in gold_clusters.items():
            gold_links = links(gold_mentions)
            cluster_resolution += len(predicted_links.intersection(gold_links))
        
        cluster_resolution = cluster_resolution/len(predicted_links)
        cluster_resolutions[predicted_cluster_name] = cluster_resolution
        cluster_sizes[predicted_cluster_name] = len(predicted_mentions)
        
    # take importance (size) of clusters into account
    if sum(cluster_sizes.values()) == 0: # no predicted clusters
        fic_precision = 0
    else:
        fic_precision = sum([cluster_sizes[c] * cluster_resolutions[c] for c in predicted_clusters])/sum(cluster_sizes.values())
        
    return fic_precision


def f_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall)/(precision + recall)


def calculate_lea(predicted_mentions, gold_mentions):
    """ Calculate LEA coreference evaluation from gold and predicted AnnotatedSpans """

    predicted_clusters = group_annotations(predicted_mentions)
    gold_clusters = group_annotations(gold_mentions)

    recall = lea_recall(predicted_clusters, gold_clusters)
    precision = lea_precision(predicted_clusters, gold_clusters)
    f1 = f_score(precision, recall)

    return f1, precision, recall


def coref_scores(predicted_mentions, gold_mentions, exact_match=True):
    """ Returns coref scores """

    scores = {}
    scores['lea_f1'], scores['lea_precision'], scores['lea_recall'] = calculate_lea(predicted_mentions, gold_mentions)
    return scores


def quote_scores(predicted_quotes, gold_quotes, exact_match=True):
    """ Returns quote extraction and attribution scores """
    scores = {}

    # Precision, recall of the quote extraction (the markables)
    matched_pred_quotes, matched_gold_quotes, false_positives, false_negatives = match_spans(predicted_quotes, gold_quotes, exact=exact_match)
    if len(predicted_quotes) == 0:
        if len(gold_quotes) == 0:
            extraction_precision = 1
        else:
            extraction_precision = 0
    else:
        extraction_precision = min(len(matched_pred_quotes)/len(predicted_quotes), 1)
    if len(gold_quotes) == 0:
        extraction_recall = 1 # everyone gets perfect recall if there are no quotes
    else:
        extraction_recall = len(matched_gold_quotes)/len(gold_quotes)
    extraction_f1 = f_score(extraction_precision, extraction_recall)
    scores['extraction_f1'] = extraction_f1
    scores['extraction_precision'] = extraction_precision
    scores['extraction_recall'] = extraction_recall

    # Quote attribution accuracy on matched quotes
    correct_attributions, incorrect_attributions = match_annotated_spans(matched_pred_quotes, matched_gold_quotes, matched=True, incorrect_extractions=false_positives)
    if len(matched_pred_quotes) == 0:
        attribution_accuracy_matched = 0
    else:
        attribution_accuracy_matched = len(correct_attributions)/len(matched_pred_quotes)

    # Quote attribution accuracy on all predicted quotes.
    # If the predicted quote is not a real quote span, is not a match
    if len(predicted_quotes) == 0:
        if len(gold_quotes) == 0:
            attribution_precision = 1
        else:
            attribution_precision = 0
    else:
        attribution_precision = len(correct_attributions)/len(predicted_quotes)
    if len(gold_quotes) == 0:
        attribution_recall = 1 # everyone gets perfect recall if no quotes
    else:
        attribution_recall = len(correct_attributions)/len(gold_quotes)
    attribution_f1 = f_score(attribution_precision, attribution_recall)
    scores['attribution_f1'] = attribution_f1
    scores['attribution_precision'] = attribution_precision
    scores['attribution_recall'] = attribution_recall

    #print(f'\t\tAccuracy on matched quote spans: {attribution_accuracy_matched: .2%} ({len(correct_attributions)}/{len(matched_pred_quotes)})')

    return scores
