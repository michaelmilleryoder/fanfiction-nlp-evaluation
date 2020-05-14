""" Classes for loading and storing annotations of coreference and quotes """

import os
import pandas as pd

from fic_representation import FicRepresentation
import evaluation_utils as utils

from annotated_span import AnnotatedSpan


class Annotation(FicRepresentation):
    
    def __init__(self, annotations_dirpath, fandom_fname, file_ext='.csv', fic_csv_dirpath=None):
        super().__init__(fandom_fname, fic_csv_dirpath=fic_csv_dirpath)
        self.file_path = os.path.join(annotations_dirpath, f'{fandom_fname}{file_ext}')
        self.extract_annotated_spans()

    def extract_annotated_spans(self):
        """ Load gold fic annotations, match text to mentions
            Saves to self.annotations
         """
        # Load ground-truth annotated entity mentions or quote spans, attributed to each character

        self.annotations = [] # list of AnnotatedSpans

        df = pd.read_csv(self.file_path)
        for colname in df.columns:
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
                    
                self.annotations.append(AnnotatedSpan(
                    chap_id=chapter_id,
                    para_id=paragraph_id,
                    start_token_id=token_id_start,
                    end_token_id=token_id_end,
                    annotation=colname
                ))

        # Match text to mentions
        if self.fic_csv is None:
            self.load_fic_csv()
        fic_data = self.fic_csv.set_index(['chapter_id', 'para_id'], inplace=False)
        para_tokens = fic_data['text_tokenized'].str.split().to_dict() # (chap_id, para_id): tokens
        
        for mention in self.annotations:
            mention.text = ' '.join(para_tokens[(mention.chap_id, mention.para_id)][mention.start_token_id-1:mention.end_token_id])
