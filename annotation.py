""" Classes for loading and storing annotations of coreference and quotes """

import os
import pandas as pd
from copy import deepcopy
import pdb

from fic_representation import FicRepresentation
import evaluation_utils as utils

from annotated_span import AnnotatedSpan, all_characters, group_annotations, normalize_annotations_to_name


class Annotation(FicRepresentation):
    
    def __init__(self, annotations_dirpath, fandom_fname, file_ext='.csv', fic_csv_dirpath=None):
        super().__init__(fandom_fname, fic_csv_dirpath=fic_csv_dirpath)
        self.file_path = os.path.join(annotations_dirpath, f'{fandom_fname}{file_ext}')

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
            if not (mention.chap_id, mention.para_id) in para_tokens:
                raise ValueError(f"Chapter ID and paragraph ID in {mention} not present in fic {self.file_path}")
            mention.text = ' '.join(para_tokens[(mention.chap_id, mention.para_id)][mention.start_token_id-1:mention.end_token_id])

    def save_annotated_spans(self, annotations):
        """ Normalize annotatations, save in format of one column/character """
        normalized_annotations = normalize_annotations_to_name(annotations)
        grouped = group_annotations(normalized_annotations)
        output = {}
        longest_column = max([len(spans) for spans in grouped.values()])
        for char, spans in grouped.items():
            output[char] = [span.readable_span() for span in spans] + [''] * (longest_column - len(spans))
        df = pd.DataFrame(output)
        df.to_csv(self.file_path, index=False)
        print(f"Annotated spans saved to {self.file_path}")

    def annotation_bio(self):
        """ Returns a sequence of token-level BIO annotations """

        # Get token lengths
        self.fic_csv['token_count'] = self.fic_csv['text_tokenized'].map(lambda x: len(x.split()))
        fic = self.fic_csv.set_index(['chapter_id', 'para_id'], inplace=False)
        fic['original_bio'] = fic['token_count'].map(lambda x: ['O'] * x)
        annotation_bio = deepcopy(fic['original_bio'].to_dict())

        # Mark with BIO
        for span in self.annotations:
            bio = annotation_bio[(span.chap_id, span.para_id)]
            if span.start_token_id-1 >= len(bio):
                raise ValueError(f"Span {span} out of range in fic {self.file_path}")
            bio[span.start_token_id-1] = 'B' # 1-start instead of 0
            bio[span.start_token_id:span.end_token_id] = ['I'] * (span.end_token_id-span.start_token_id)
            annotation_bio[(span.chap_id, span.para_id)] = bio

        ordered_bio = [vals for _,vals in sorted(annotation_bio.items())] 
        bio_labels = [val for vals in ordered_bio for val in vals] # flatten
        return bio_labels
