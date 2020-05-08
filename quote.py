""" Data structure for holding quote representations for evaluation """

import re
import pdb
from string import punctuation


class Quote():

    def __init__(self, 
            chap_id=None, 
            para_id=None, 
            start_token_id=None, 
            end_token_id=None, 
            speaker=None,
            text=None):
        self.chap_id = chap_id # starts with 1, just like annotations
        self.para_id = para_id # starts with 1, just like annotations
        self.start_token_id = start_token_id # starts over every paragraph, starts with 1 just like annotations
        self.end_token_id = end_token_id
        self.text = text
        self.speaker = speaker

    def __repr__(self):
        return f"{self.chap_id}.{self.para_id}.{self.start_token_id}-{self.end_token_id},speaker={self.speaker}"

    def extraction_matches(self, other_quote, exact=True):
        """ Check if the extracted quote matches another quote.
            Ignores speaker attribution.
            Args:
                exact: whether an exact match on token IDs is necessary.
                    Otherwise matches if has is in the same paragraph, 
                    has very similar text and beginning and start points 
                    occur within a small window of the other quote.
        """

        if not (self.chap_id == other_quote.chap_id and \
            self.para_id == other_quote.para_id):
                return False

        if exact:
            return self.quote_endpoints_align(other_quote, exact=exact)
        else:
            return self.quote_text_matches(other_quote) and \
                self.quote_endpoints_align(other_quote, exact=exact)

    def quote_endpoints_align(self, other_quote, exact=True):
        """ Returns whether quote endpoints are within a small window
            of each other (or exact if specified).
        """

        if exact:
            return (self.start_token_id == other_quote.start_token_id and \
                self.end_token_id == other_quote.end_token_id)
        else: 
            window_size = 3
            return abs(self.start_token_id - other_quote.start_token_id) <= window_size and abs(self.end_token_id - other_quote.end_token_id) <= window_size
    
    def quote_text_matches(self, other_quote):
        
        processed_quotes = []
        word_match_threshold = .5

        if not hasattr(self, 'text_tokens'):
            self.preprocess_quote_text()
        if not hasattr(other_quote, 'text_tokens'):
            other_quote.preprocess_quote_text()
        
        # Measure unique word overlap
        n_matches = len(self.text_tokens.intersection(other_quote.text_tokens))

        # Check for edge cases
        if len(other_quote.text_tokens) == 0:
            if len(self.text_tokens) < 4: # Probably just the name of a character
                return True
            else:
                return False

        return (n_matches/len(other_quote.text_tokens)) >= word_match_threshold

    def preprocess_quote_text(self):
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
