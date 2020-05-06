""" Data structure for holding quote representations for evaluation """

class Quote():

    def __init__(self, chap_id, para_id, start_token_id, end_token_id, speaker, text=''):
        self.chap_id = int(chap_id) # starts with 1, just like annotations
        self.para_id = int(para_id) # starts with 1, just like annotations
        self.start_token_id = int(start_token_id) # starts over every paragraph, starts with 1 just like annotations
        self.end_token_id = int(end_token_id)
        self.text = text
        self.speaker = speaker

    def __repr__(self):
        return f"{self.chap_id}.{self.para_id}.{self.start_token_id}-{self.end_token_id},speaker={self.speaker}"

    def extraction_matches(self, other_quote):
        return (self.chap_id == other_quote.chap_id and \
            self.para_id == other_quote.para_id and \
            self.start_token_id == other_quote.start_token_id and \
            self.end_token_id == other_quote.end_token_id)
