""" Parser for character mentions output by the FanfictionNLP pipeline """

from html.parser import HTMLParser
import pdb

from annotated_span import AnnotatedSpan


class CharacterMentionParser(HTMLParser):
    
    def __init__(self):
        HTMLParser.__init__(self)
        self.character_mentions = [] # ordered list of AnnotatedSpan objects
        self.text = ''
        self._current_token_id = 1 # starts with 1
        self._current_characters = [] # stack for holding current characters
        self._current_tokens = [] # stack for holding tokens in a mention
        self._start_token_ids = [] # stack for holding tag start token IDs
    
    def handle_starttag(self, tag, attrs):
        self._current_characters.append(attrs[0][1])
        self._start_token_ids.append(self._current_token_id)
        
    def handle_endtag(self, tag):
        start_id = self._start_token_ids.pop()
        exclusive_end_id = self._current_token_id
        inclusive_end_id = self._current_token_id - 1
        character = self._current_characters.pop()
        tokens = self._current_tokens.pop()
        
        # Add to character_mentions
        self.character_mentions.append(AnnotatedSpan(
            start_token_id = start_id, 
            end_token_id = inclusive_end_id,
            annotation = character,
            text = ' '.join(tokens)
        ))
        
    def handle_data(self, data):
        words = data.split()
        self.text += data + ' '
        self._current_token_id += len(words)
        self._current_tokens.append(words)
        
    def print_character_mentions(self):
        for mention in self.character_mentions:
            print(f"{mention}\n")
