""" Superclass data structure for holding a predicted or annotate span of text """

import re
import itertools
from string import punctuation
import pdb


def match_spans(predicted_spans, gold_spans, exact=True):
    """ Match AnnotatedSpan objects just on extracted spans (ignoring annotations).
        Args:
            predicted_quotes: AnnotatedSpan objects predicted
            gold_quotes: AnnotatedSpan objects annotated as gold truth
            exact: whether an exact match on token IDs is necessary.
                For the FanfictionNLP pipeline, this should be the case.
                For baseline systems that might have different tokenization,
                this can be set to False to relax that constraint.
    """

    matched_gold = []
    matched_predicted = []
    false_positives = []
    false_negatives = []

    matched = [(predicted, gold) for predicted, gold in itertools.product(predicted_spans, gold_spans) if gold.span_matches(predicted, exact=exact)]
    if len(matched) == 0:
        matched_predicted, matched_gold = [], []
    else:
        matched_predicted, matched_gold = list(zip(*matched))

    false_positives = [predicted for predicted in predicted_spans if not predicted in matched_predicted]
    false_negatives = [gold for gold in gold_spans if not gold in matched_gold]

    return matched_gold, matched_predicted, false_positives, false_negatives


def match_annotated_spans(predicted_spans, gold_spans, matched=False, incorrect_extractions=[]):
    """ Match AnnotatedSpan objects entirely, including annotation.
        If extractions don't match, counts it as a mismatched annotation.
        Returns matched_attributions, mismatched_attributions
    """

    correct_attributions = []

    # Check extractions
    if not matched:
        matched_gold, matched_predicted, incorrect_extractions, _ = self.match_spans(predicted_spans, gold_spans)
    else:
        matched_gold, matched_predicted = gold_spans, predicted_spans

    incorrect_attributions = incorrect_extractions

    # Find matched gold spans
    for pred_span, gold_span in zip(matched_predicted, matched_gold):
    
        # Check attribution
        if characters_match(pred_span.annotation, gold_span.annotation):
            correct_attributions.append((pred_span, gold_span))
        else:
            incorrect_attributions.append((pred_span, gold_span))

    return correct_attributions, incorrect_attributions


def annotation_labels(attributions1, attributions2, matched=False, mismatched_extractions1=[], mismatched_extractions2=[]):
    """ Get lists of labels (including null) from 2 lists of AnnotatedSpans
        Returns matched_attributions, mismatched_attributions, mismatched_extractions where each is a list of tuples (annotated_span1, annotated_span2)
    """
    # TODO: merge with match_annotated_spans?

    matched_attributions = []
    mismatched_attributions = []
    mismatched_extractions = []
    score = None

    # Check extractions
    if not matched:
        matched1, matched2, spans_1not2, spans_2not1 = self.match_spans(attributions1, attributions2)
    else:
        matched1, matched2 = attributions1, attributions2
        spans_1not2 = mismatched_extractions1
        spans_2not1 = mismatched_extractions2

    # Find matched gold spans
    for span1, span2 in zip(matched1, matched2):
    
        # Check attribution
        if characters_match(span1.annotation, span2.annotation):
            matched_attributions.append((span1, span2))
        else:
            mismatched_attributions.append((span1, span2))

    # Total attribution score (including mismatched extractions)
    for span in spans1_not2:
        mismatched_extractions.append((span, span.null_span()))
    for span in spans2_not1:
        mismatched_extractions.append((span.null_span(), span))

    return matched_attributions, mismatched_attributions, mismatched_extractions

def group_annotations(spans):
    """ Group annotations by annotation
        Returns dictionary of {annotation: [AnnotatedSpan, ...]}
    """

    clusters = {}
    for span in spans:
        if not span.annotation in clusters:
            clusters[span.annotation] = []
        clusters[span.annotation].append(span)

    return clusters


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

def get_union_quotes(gold_quotes, baseline_quotes, experimental_quotes):
    """ Returns the union of all quote spans [old, would need to be changed to use] """

    all_quote_spans = gold_quotes
    exclusive_baseline = [baseline_quote for baseline_quote in baseline_quotes if not any([baseline_quote.extraction_matches(quote, exact=False) for quote in all_quote_spans])]
    all_quote_spans += exclusive_baseline
    exclusive_experimental = [experimental_quote for experimental_quote in experimental_quotes if not any([experimental_quote.extraction_matches(quote, exact=False) for quote in all_quote_spans])]
    all_quote_spans += exclusive_experimental

    return all_quote_spans

class AnnotatedSpan():

    def __init__(self, 
            chap_id=None, 
            para_id=None, 
            start_token_id=None, 
            end_token_id=None, 
            annotation=None, # speaker for quotes, or character for character mentions
            text=''):
        self.chap_id = chap_id # starts with 1, just like annotations
        self.para_id = para_id # starts with 1, just like annotations
        self.start_token_id = start_token_id # starts over every paragraph, starts with 1 just like annotations
        self.end_token_id = end_token_id
        self.annotation = annotation
        self.text = text

    def __repr__(self):
        return f"{self.chap_id}.{self.para_id}.{self.start_token_id}-{self.end_token_id},annotation={self.annotation}"

    def null_span(self):
        """ Returns an identical span but with a NULL annotation"""
        return AnnotatedSpan(
            chap_id=self.chap_id,
            para_id=self.para_id, 
            start_token_id=self.start_token_id, 
            end_token_id=self.end_token_id, 
            annotation='NULL')

    def get_location(self):
        return (self.chap_id, self.para_id, self.start_token_id, self.end_token_id)

    def span_matches(self, other_span, exact=True):
        """ Check if the extracted span matches another span.
            Ignores attribution.
            Args:
                exact: whether an exact match on token IDs is necessary.
                    Otherwise matches if is in the same paragraph, 
                    has very similar text and beginning and start points 
                    occur within a small window of the other span.
        """

        if not (self.chap_id == other_span.chap_id and \
            self.para_id == other_span.para_id):
                return False

        if exact:
            return self.span_endpoints_align(other_span, exact=exact)
        else:
            return self.span_text_matches(other_span) and \
                self.span_endpoints_align(other_span, exact=exact)

    def span_endpoints_align(self, other_span, exact=True):
        """ Returns whether span endpoints are within a small window
            of each other (or exact if specified).
        """

        if exact:
            return (self.start_token_id == other_span.start_token_id and \
                self.end_token_id == other_span.end_token_id)
        else: 
            window_size = 3
            return abs(self.start_token_id - other_span.start_token_id) <= window_size and abs(self.end_token_id - other_span.end_token_id) <= window_size
    
    def span_text_matches(self, other_span):
        
        processed_spans = []
        word_match_threshold = .5

        if not hasattr(self, 'text_tokens'):
            self.preprocess_text()
        if not hasattr(other_span, 'text_tokens'):
            other_span.preprocess_text()
        
        # Measure unique word overlap
        n_matches = len(self.text_tokens.intersection(other_span.text_tokens))

        # Check for edge cases
        if len(other_span.text_tokens) == 0:
            if len(self.text_tokens) < 4: # Probably just the name of a character
                return True
            else:
                return False

        return (n_matches/len(other_span.text_tokens)) >= word_match_threshold

    def preprocess_text(self):
        """ Creates a set of lowercased unique tokens from a quote's text.
            Saves to self.text_tokens
        """

        # Remove ccc_ business
        processed_quote = re.sub(r'ccc_.*?_ccc', '', self.text)

        # Remove punctuation, lowercase
        stops = list(punctuation) + ['”', '“']
        processed_quote = ''.join([c for c in processed_quote.lower() if not c in stops])

        # Replace whitespace with spaces
        #processed_quote = re.sub(r'\s+', ' ', processed_quote)
        
        # Extract unique words
        processed_words = set(processed_quote.strip().split())

        self.text_tokens = processed_words
