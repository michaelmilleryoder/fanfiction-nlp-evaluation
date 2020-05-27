""" Wrapper to run BookNLP """
import os
from subprocess import call
import pdb

class BookNLPWrapper():

    def __init__(self, token_fpath, json_fpath, quote_attribution_only=True):
        self.token_fpath = token_fpath
        self.json_fpath = json_fpath
        self.quote_attribution_only = quote_attribution_only
        if self.quote_attribution_only:
            self.booknlp_dirpath = '/projects/book-nlp-quote-attribution'
    
    def run(self):
        print("\tRunning BookNLP quote attribution...")
        os.chdir(self.booknlp_dirpath)
        cmd = ['./runjava', 'novels/BookNLP', 
                '-p', self.json_fpath,
                '-tok', self.token_fpath,
                '-onlyQuotes']
        call(cmd)
