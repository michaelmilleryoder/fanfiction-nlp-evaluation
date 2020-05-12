""" Superclass data structure for holding a predicted or annotate span of text """

class AnnotatedSpan():

    def __init__(self, 
            chap_id=None, 
            para_id=None, 
            start_token_id=None, 
            end_token_id=None, 
            annotation=None, # speaker for quotes, or character for character mentions
            text=None):
        self.chap_id = chap_id # starts with 1, just like annotations
        self.para_id = para_id # starts with 1, just like annotations
        self.start_token_id = start_token_id # starts over every paragraph, starts with 1 just like annotations
        self.end_token_id = end_token_id
        self.annotation = annotation
        self.text = text
