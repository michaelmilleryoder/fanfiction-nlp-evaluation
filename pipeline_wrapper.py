""" Wrapper to run fanfiction NLP wrapper """
import os
from subprocess import call
import pdb

class PipelineWrapper():

    def __init__(self, coref_stories_dirpath, coref_chars_dirpath, quote_dirpath, parts=['quotes']):
        self.parts = parts
        self.coref_stories_dirpath = coref_stories_dirpath
        self.coref_chars_dirpath = coref_chars_dirpath
        self.quote_dirpath = quote_dirpath
        self.pipeline_dirpath = '/projects/fanfiction-nlp'
    
    def run(self):
        if 'quotes' in self.parts:
            print("\tRunning pipeline quote attribution...")
            os.chdir(os.path.join(self.pipeline_dirpath, 'quote_attribution'))
            cmd = ['python3', 'run.py', 'predict',
                    '--story-path', self.coref_stories_dirpath,
                    '--char-path', self.coref_chars_dirpath,
                    '--output-path', self.quote_dirpath,
                    #'--features', 'disttoutter', 'spkappcnt', 'nameinuttr', 'spkcntpar', 'neighboring',
                    '--features', 'disttoutter', 'spkappcnt', 'nameinuttr', 'spkcntpar',
                    #'--features', 'disttoutter', 'spkappcnt', 'nameinuttr',
                    '--model-path', 'austen_4.model',
                    '--svmrank', '/usr0/home/mamille2/svm_rank']
            call(cmd)
