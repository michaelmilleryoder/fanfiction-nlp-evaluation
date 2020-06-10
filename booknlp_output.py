import os
import sys
import json
import pickle
import pdb
import csv
import re
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

from fic_representation import FicRepresentation
from booknlp_wrapper import BookNLPWrapper
from annotated_span import AnnotatedSpan
from annotation import Annotation
import evaluation_utils as utils


def modify_paragraph_id(para_id, trouble_line):
    
    if para_id == trouble_line: # trouble line
        new_para_id = trouble_line + 2 # add 1 anyway for the index-0 issue
        
    if para_id >= trouble_line + 1:
        new_para_id = para_id
    
    else:
        new_para_id = para_id + 1 # add 1 since BookNLP starts with 0

    return new_para_id


def load_tokens_file(token_fpath):
    """ Load a BookNLP tokens file, return a pandas DataFrame """ 
    return pd.read_csv(token_fpath, sep='\t', quoting=csv.QUOTE_NONE)


def save_tokens_file(token_data, token_fpath):
    token_data.to_csv(token_fpath, sep='\t', quoting=csv.QUOTE_NONE, index=False)


def token_matches(original_tok, whitespace_tok, quotes=False):
    transformations = {')': '-RRB-', # whitespace_tok: original_tok
                       '(': '-LRB-',
                       '?-': '-',
                        '…': '...',
                        '—': '--',
                        '—And': 'And',
                        'name—': 'name',
                        'make—': 'make',
                      }
    transformations_quotes = {**transformations, **{
                       '"': "``",
                       '“': "``",
                       "'": "`",
                       '"': "''",
                       '”': "''",
                      }}
    if original_tok == whitespace_tok:
        return True
    if quotes:
        if whitespace_tok in transformations_quotes and transformations_quotes[whitespace_tok] == original_tok:
            return True
        else:
            return False
    else:
        if whitespace_tok in transformations and transformations[whitespace_tok] == original_tok:
            return True
        else:
            return False

def match_quotes(source, target):
    """ Match quote styles in the source DataFrame to the target DataFrame.
        Source assumed to be original tokenization, target is whitespace-tokenized.
    """
    new_tokens = []
    offset = 0 # how many tokens whitespace appears to be off from original
    desired_quote_chars = ['``', '`', "''", "'"]
    quote_chars = ['“', '``', '"', '«', '”', "''", '"', '»', "'"]
    for i in range(len(target)):
        if i+offset >= len(source):
            pdb.set_trace()
        original_tok = source.loc[i+offset, 'normalizedWord']
        whitespace_tok = target.loc[i, 'normalizedWord']
        tok_to_add = whitespace_tok
        
        if not token_matches(original_tok, whitespace_tok, quotes=False):
            if original_tok in quote_chars:
                # Add the original quote token
                tok_to_add = original_tok
            else:
                next_whitespace_tok = target.loc[i+1, 'normalizedWord']
                # Find offset using the next non-quote character
                for j in range(1,5):
                    next_original_tok = source.loc[i+offset+j, 'normalizedWord']
                    if token_matches(next_original_tok, next_whitespace_tok, quotes=True):
                        offset += j-1
                        break
                else:
                    pdb.set_trace()
        new_tokens.append(tok_to_add)
        
    assert len(new_tokens) == len(target)
    target['normalizedWord'] = new_tokens
    target['lemma'] = new_tokens
    return target


class BookNLPOutput(FicRepresentation):
    """ Holds representation for the BookNLP processed output of a fic. """

    def __init__(self, token_output_dirpath, fandom_fname, json_output_dirpath=None, fic_csv_dirpath=None, token_file_ext='.tokens', original_tokenization_dirpath=None): 
        """ 
        Args:
            csv_dirpath: path to directory with corresponding original fic CSV
            token_file_ext: file extension after fandom_fname for token output files
        """
        super().__init__(fandom_fname, fic_csv_dirpath=fic_csv_dirpath)
        self.original_token_output_dirpath = token_output_dirpath
        self.modified_token_output_dirpath = token_output_dirpath
        self.original_token_fpath = os.path.join(token_output_dirpath, f'{fandom_fname}{token_file_ext}')
        self.modified_token_fpath = self.original_token_fpath
        self.original_token_data = load_tokens_file(self.original_token_fpath)
        self.token_data = self.original_token_data.copy()
        self.original_json_dirpath = json_output_dirpath
        self.modified_json_dirpath = json_output_dirpath
        self.token_file_ext = token_file_ext
        self.original_tokenization_dirpath = original_tokenization_dirpath

    def load_json_output(self):
        json_fpath = os.path.join(self.modified_json_dirpath, self.fandom_fname, 'book.id.book')
        with open(json_fpath, 'r') as f:
            self.json_data = json.load(f)

    def align_with_annotations(self):
        """ Align token IDs, paragraph breaks with annotated fics. 
            Modified/aligned data in self.token_data
        """

        # Load original fic CSV
        self.load_fic_csv()

        print("\tChecking/fixing paragraph breaks...")
        sys.stdout.flush()
        ## Check paragraph breaks
        # Compare number of paragraphs
        n_diff = self.check_paragraph_breaks()
        if n_diff != 0:
            self.fix_paragraph_breaks(n_diff)

        print("\tChecking/fixing token alignment...")
        sys.stdout.flush()
        ## Make sure token IDs align
        misaligned_rows = self.get_misaligned_paragraph()
        if len(misaligned_rows) > 0:
            #print(f"\tFound {len(misaligned_rows)} misaligned rows")
            self.fix_token_misalignment(misaligned_rows)

        ## Renumber BookNLP token IDs
        self.get_paragraph_token_ids()

    def check_paragraph_breaks(self):
        """ Checks for the same number of paragraphs between BookNLP output and annotations. """

        assert hasattr(self, 'token_data') and self.token_data is not None and \
                hasattr(self, 'fic_csv') and self.fic_csv is not None

        # In case it wasn't created in modify_paragraph_ids, create a "full_paragraphID" column that runs for the whole story
        if not 'full_paragraphId' in self.token_data.columns:
            self.token_data['full_paragraphId'] = self.token_data['paragraphId'] + 1

        n_para_diff = len(self.token_data['full_paragraphId'].unique()) - len(set(zip(self.fic_csv['chapter_id'], self.fic_csv['para_id'])))

        # Add in chapter ID column, restart modified_paragraphId every chapter
        para2chap = {} # modified_paragraphId: (chapter_id, para_id)
        para_offset = 0
        if n_para_diff == 0:
            chap_paras = list(zip(self.fic_csv['chapter_id'], self.fic_csv['para_id']))
            all_paras = sorted(self.token_data['full_paragraphId'].unique().tolist())
            
            for i, para_num in enumerate(all_paras):
                para2chap[para_num] = chap_paras[i]

            self.token_data['chapterId'] = self.token_data['full_paragraphId'].map(lambda x: para2chap[x][0])
            self.token_data['modified_paragraphId'] = self.token_data['full_paragraphId'].map(lambda x: para2chap[x][1])

        return n_para_diff

    def fix_paragraph_breaks(self, n_diff):
        """ Fix paragraph break issues """
        trouble_line = self.find_mismatched_paragraphs(n_diff)
        self.modify_paragraph_ids(trouble_line)

        # Confirm that fixed it
        n_para_diff = self.check_paragraph_breaks()
        if n_para_diff != 0:
            pdb.set_trace()

    def modify_paragraph_ids(self, trouble_line):
        
        self.token_data['full_paragraphId'] = [modify_paragraph_id(para_id, trouble_line) for para_id in self.token_data['paragraphId']]

    def find_mismatched_paragraphs(self, n_diff):
        """ Right now only handles 1 mismatch """

        if n_diff != 1:
            pdb.set_trace()

        trouble_line = -1

        booknlp_paras = self.token_data.groupby('paragraphId').agg({'originalWord': lambda x: ' '.join(x.tolist())})['originalWord']

        for i, (booknlp_para, fic_para) in enumerate(zip(booknlp_paras, self.fic_csv['text_tokenized'])):
            # There will be tokenization differences, so look for dramatic differences
            if abs(len(booknlp_para.split()) - len(fic_para.split())) > 10:
                trouble_line = i
                break

        return trouble_line

    def get_misaligned_paragraph(self):
        # Get token counts for paragraphs from BookNLP, make sure they match the original fic token counts

        self.fic_csv['booknlp_para_length'] = self.token_data.groupby('full_paragraphId').size().tolist() # Paragraphs should be in the same order (already checked this)
        self.fic_csv['token_count'] = self.fic_csv['text_tokenized'].map(lambda x: len(x.split()))
        misaligned_rows = self.fic_csv.loc[self.fic_csv['token_count'] != self.fic_csv['booknlp_para_length'], ['chapter_id', 'para_id', 'token_count', 'booknlp_para_length']]

        return misaligned_rows

    def fix_token_misalignment(self, misaligned_rows):
        # Fix token misalignment issues
        for selected_chap_id, selected_para_id in zip(misaligned_rows['chapter_id'], misaligned_rows['para_id']):

            gold_tokens = self.fic_csv.loc[(self.fic_csv['chapter_id']==selected_chap_id) & (self.fic_csv['para_id']==selected_para_id), 'text_tokenized'].tolist()[0].split()
            booknlp_tokens = self.token_data.loc[(self.token_data['chapterId']==selected_chap_id) & (self.token_data['modified_paragraphId']==selected_para_id), 'originalWord'].tolist()

            total_offset = 0
            trouble_offsets = {} # line_number: offset
            #modify_words = {} # index: new word
            #remove_words = {} # indices
            first_tokenId = self.token_data.loc[(self.token_data['chapterId']==selected_chap_id) & (self.token_data['modified_paragraphId']==selected_para_id), 'tokenId'].tolist()[0]

#            if selected_para_id == 29:
#                pdb.set_trace()

            for i, gold_tok in enumerate(gold_tokens):

                current_booknlp_token = booknlp_tokens[i + total_offset]
                if not gold_tok == current_booknlp_token:
        
                    # Try detecting an ellipsis
            #        if len(current_booknlp_token) > 1 and current_booknlp_token.endswith('.') and len(booknlp_tokens) < i+total_offset+2 and booknlp_tokens[i + total_offset + 1] == '.' and booknlp_tokens[i + total_offset + 2] == '.':
            #            total_offset += 2
            #            continue

                    # Try adding tokens
                    added = current_booknlp_token
                    for offset in range(1, 4):
                        if i + total_offset + offset >= len(booknlp_tokens):
                            break
                        added += booknlp_tokens[i + total_offset + offset]
                        if added == gold_tok:
                            total_offset += offset
                            trouble_offsets[first_tokenId + i] = offset
                            break

                    else:
                        print(gold_tok)
                        print(booknlp_tokens[i])
                        pdb.set_trace()

            # Modify BookNLP output
            for line, offset in trouble_offsets.items():
                row_filter = (self.token_data['chapterId']==selected_chap_id) & (self.token_data['modified_paragraphId']==selected_para_id) & (self.token_data['tokenId'].isin(range(line, line+offset+1)))

                # Modify offset word
                new_word = ''.join(self.token_data.loc[row_filter, 'originalWord'].tolist())
                modified_row_filter = (self.token_data['chapterId']==selected_chap_id) & (self.token_data['modified_paragraphId']==selected_para_id) & (self.token_data['tokenId']==line)
                self.token_data.loc[modified_row_filter, 'originalWord'] = new_word

                # Delete offset words
                delete_row_filter = (self.token_data['chapterId']==selected_chap_id) & (self.token_data['modified_paragraphId']==selected_para_id) & (self.token_data['tokenId'].isin(range(line+1, line+offset+1)))
                delete_index = self.token_data.loc[delete_row_filter].index
                self.token_data.drop(index=delete_index, inplace=True)

        # Confirm token length match
        misaligned_rows = self.get_misaligned_paragraph()
        if len(misaligned_rows) > 0:
            pdb.set_trace()

    def get_paragraph_token_ids(self):
        para_token_lengths = self.token_data.groupby('full_paragraphId').size().tolist()
        new_tokenIds = sum([list(range(1, para_length+1)) for para_length in para_token_lengths], [])
        self.token_data['modified_tokenId'] = new_tokenIds

    def renumber_token_ids(self):

        if max(self.token_data['tokenId']) == len(self.token_data):
            return

        self.token_data['tokenId'] = range(len(self.token_data))

    def extract_bio_quotes(self):
        """ Extracts Quote objects (unattributed) from BookNLP output token data.
            Saves to self.quotes
            Not used anymore since hardly any quotes are saved in the token files.
        """

        selected_columns = ['chapterId', 'modified_paragraphId', 'modified_tokenId', 'originalWord', 'inQuotation']
        quote_token_data = self.token_data.loc[self.token_data['inQuotation']!='O', selected_columns]

        current_chapter_id = 0
        current_para_id = 0
        current_quote_start = 0
        current_quote_tokens = []
        prev_token_id = 0
        self.quotes = []

        for row in list(quote_token_data.itertuples()):
            if row.inQuotation == 'B-QUOTE': # Start of quote
                if len(current_quote_tokens) != 0: 
                    # Store past quote
                    self.quotes.append(Quote(current_chapter_id, current_para_id, current_quote_start, prev_token_id, text=' '.join(current_quote_tokens)))
                quote_token_id_start = row.modified_tokenId
                current_chapter_id = row.chapterId
                current_para_id = row.modified_paragraphId

            current_quote_tokens.append(row.originalWord)
            prev_token_id = row.modified_tokenId

    def build_character_id_name_map(self):
        """ Builds a dict {character_id: character_name}.
            Save to self.character_id2name 
        """

        self.character_id2name = {}

        if not hasattr(self, 'json_data'):
            self.load_json_output()

        for char in self.json_data['characters']:
            char_id = char['id']
            char_name = char['names'][0]['n'] # take first name as name
            self.character_id2name[char_id] = char_name

    def extract_character_mentions(self, save_dirpath=None):
        """ Extracts character mentions, saves in self.character_mentions, also in save_dirpath if specified """

        self.character_mentions = []
        self.build_character_id_name_map()

        selected_cols = ['chapterId', 'modified_paragraphId', 'modified_tokenId', 'characterId', 'originalWord']
        char_values = self.token_data['characterId'].unique()
        if len(char_values) == 1 and char_values[0] == -1: # no character mentions
            if save_dirpath:
                self.pickle_output(save_dirpath, self.character_mentions)
            return
            
        mentions = self.token_data[self.token_data['characterId']>-1].loc[:, selected_cols]

        # Calculate end tokens for any entity mentions
        mentions['next_entity_tokenId'] = mentions['modified_tokenId'].tolist()[1:] + [0]
        mentions['next_entity_paragraphId'] = mentions['modified_paragraphId'].tolist()[1:] + [0]
        mentions['next_entity_characterId'] = mentions['characterId'].tolist()[1:] + [0]
        mentions['sequential'] = [(next_entity_tokenId == modified_tokenId + 1) and                               (next_entity_paragraphId == modified_paragraphId) and                               (next_entity_characterId == characterId) 
                                for next_entity_tokenId, modified_tokenId, next_entity_paragraphId, modified_paragraphId, next_entity_characterId, characterId in \
                                  zip(mentions['next_entity_tokenId'], mentions['modified_tokenId'], mentions['next_entity_paragraphId'], \
                                      mentions['modified_paragraphId'], mentions['next_entity_characterId'], mentions['characterId'])
                                     ]

        prev_was_sequential = False
        prev_token_id_start = 0
        mention_tokens = []

        for row in list(mentions.itertuples()):
            chapter_id = row.chapterId
            para_id = row.modified_paragraphId
            character_id = row.characterId
            token_id_start = row.modified_tokenId
            mention_tokens.append(str(row.originalWord))

            if row.sequential: # Store last token ID
                if not prev_was_sequential: # not in the middle of an entity mention
                    prev_was_sequential = True
                    prev_token_id_start = token_id_start

            else:
                # Save character mention
                if prev_was_sequential:
                    token_id_start = prev_token_id_start

                token_id_end = row.modified_tokenId
                if np.isnan(character_id):
                    pdb.set_trace()
                character_name = self.character_id2name[character_id]
                self.character_mentions.append(AnnotatedSpan(chap_id=chapter_id, para_id=para_id, start_token_id=token_id_start, end_token_id=token_id_end, annotation=character_name, text=' '.join(mention_tokens)))
                prev_was_sequential = row.sequential
                mention_tokens = []

        if save_dirpath:
            self.pickle_output(save_dirpath, self.character_mentions)

    def extract_quotes(self, save_dirpath=None, coref_from='system'):
        """ Extract AnnotatedSpan objects from BookNLP output representations. 
            Saves to self.quotes
            Args:
                save_dirpath: where to save the extracted quotes, pickled
                gold_coref: whether to apply some fixes to character names to match
                    annotated gold names
        """

        self.quotes = []

        # Load BookNLP JSON
        self.load_json_output()

        # Fixes for character names to match gold (kind of a hack)
        if coref_from == 'gold':
            name_transform = {
                    'Bilbo': 'Male Bilbo',
                    'Thorin': 'Male Thorin',
                    'Gandalf': 'Male Gandalf',
                    'me': 'Clara',
                }

        # Get AnnotatedSpan objects from all characters from BookNLP JSON
        for char in self.json_data['characters']:
            if len(char['names']) == 0:
                pdb.set_trace()
            char_name = char['names'][0]['n'] # take first name as name
            if coref_from == 'gold':
                char_name = name_transform.get(char_name, char_name)
            for utterance in char['speaking']:
                text = utterance['w']
                quote_length = len(text.split())
                matching_token_data = self.token_data.loc[self.token_data['tokenId'].isin(range(utterance['i'], utterance['i'] + quote_length))]

                # Check if the tokens in the matching token data match the quote. If not try to search for the text
                matching_tokens = matching_token_data['originalWord'].tolist()
                if not AnnotatedSpan(text=' '.join(matching_tokens)).span_matches(AnnotatedSpan(text=text)):
                    found_quote_indices = utils.sublist_indices(text.split()[1:-1], self.token_data['normalizedWord'].tolist())
                    if len(found_quote_indices) == 1:
                        matching_token_data = self.token_data.iloc[found_quote_indices[0][0]-1:found_quote_indices[0][1]+1]

                    else: # quote not found or multiple matches found
                        continue


                chap_id = matching_token_data['chapterId'].values[0]
                para_id = Counter(matching_token_data['modified_paragraphId'].tolist()).most_common(1)[0][0]

                # In case wraps to the next paragraph                
                matching_token_data = matching_token_data[matching_token_data['modified_paragraphId']==para_id]

                modified_token_range = matching_token_data['modified_tokenId'].tolist()
                token_start = modified_token_range[0]
                token_end = modified_token_range[-1]
                assert token_start < token_end

                self.quotes.append(AnnotatedSpan(chap_id=chap_id, para_id=para_id, start_token_id=token_start, end_token_id=token_end, annotation=char_name, text=text))

        if save_dirpath is not None:
            self.pickle_output(save_dirpath, self.quotes)

    def modify_quote_tokens(self, original_tokenization_dirpath=None, quote_annotations_dirpath=None, quote_annotations_ext=None, change_to='gold'):
        """ Changes quote tokens so BookNLP will recognize them in certain ways.
            Args:
                change_to:
                    'gold': Change to gold quote extractions
                    'match': Replace quotes with smart quotes to match a tokens file done without whitespace tokenization
                    'strict': Change existing BookNLP quotes using a dictionary. Single quotes to ` and ', double quotes to `` and ''
        """
        if change_to == 'gold':
            # Load gold quote extractions
            gold = Annotation(quote_annotations_dirpath, self.fandom_fname, file_ext=quote_annotations_ext, fic_csv_dirpath=self.fic_csv_dirpath)
            gold.extract_annotated_spans()

            # Clear existing quotes, since might have been modified after whitespace tokenization
            self.clear_quotes()

            # Add gold quote spans in
            for span in gold.annotations:
                self.add_quote_span(span)
            
            # Change output dirpath for later saving (after replace gold coref)
            self.modified_token_output_dirpath = self.modified_token_output_dirpath.rstrip('/') + '_gold_quotes'

        elif change_to == 'match':
            original_tokens = load_tokens_file(os.path.join(self.original_tokenization_dirpath, self.fandom_fname + self.token_file_ext))
            self.token_data = match_quotes(original_tokens, self.token_data)

            # Save out
            save_tokens_file(self.token_data, self.modified_token_fpath)

        elif change_to == 'strict':
            quote_changes = {
                "“": "``",
                "”": "''",
            }
            self.token_data['normalizedWord'] = self.token_data['normalizedWord'].map(lambda x: quote_changes.get(x, x))
            self.token_data['lemma'] = self.token_data['lemma'].map(lambda x: quote_changes.get(x, x))

            # Save out
            pdb.set_trace()
            self.token_data.to_csv(self.modified_token_fpath, sep='\t', quoting=csv.QUOTE_NONE, index=False)

    def modify_coref_tokens(self, coref_annotations_dirpath, coref_annotations_ext):
        """ Changes coref tokens to gold annotations in self.token_data.
            Saves out to {token_output_dirpath}_gold_coref/token_fpath
        """
        # Load gold mentions
        gold = Annotation(coref_annotations_dirpath, self.fandom_fname, file_ext=coref_annotations_ext, fic_csv_dirpath=self.fic_csv_dirpath)
        gold.extract_annotated_spans()

        # Build character name to id dictionary for gold characters (arbitrary)
        self.char_name2id = defaultdict(lambda: len(self.char_name2id))
        #self.char_name2id = {charname: len(self.char_name2id) for charname in sorted(gold.annotations_set)}

        # Clear existing character coref annotations
        self.token_data['characterId'] = -1

        # Modify original tokens file
        for span in gold.annotations:
            self.modify_coref_span(span)

        # Renumber BookNLP's own token IDs for re-running on modified output
        self.renumber_token_ids()
        
        # Save out
        self.modified_token_output_dirpath = self.modified_token_output_dirpath.rstrip('/') + '_gold_coref'
        if not os.path.exists(self.modified_token_output_dirpath):
            os.mkdir(self.modified_token_output_dirpath)
        self.modified_token_fpath = os.path.join(self.modified_token_output_dirpath, f'{self.fandom_fname}{self.token_file_ext}')
        self.token_data.to_csv(self.modified_token_fpath, sep='\t', quoting=csv.QUOTE_NONE, index=False)
        #print(f"Wrote gold coref token file to {self.modified_token_fpath}")

    def modify_coref_span(self, span):
        """ Modify token data to match gold span """
        for i in range(span.start_token_id, span.end_token_id + 1):
            self.token_data.loc[(self.token_data['chapterId']==span.chap_id) & (self.token_data['modified_paragraphId']==span.para_id) & (self.token_data['modified_tokenId']==i), 'characterId'] = self.char_name2id[span.annotation]

    def clear_quotes(self):
        """ Clear quote marks that are recognized by BookNLP into smart quotes,
            not recognized 
        """
        transformations = {
            "``": '“',
            "''": '”',
            "`": "'",
        }

        for colname in ['normalizedWord', 'lemma']:
            self.token_data[colname] = self.token_data[colname].map(lambda x: transformations.get(x, x))

    def add_quote_span(self, span):
        """ Add quote marks so that BookNLP recognizes a span in self.token_data"""
        start_span_filter = (self.token_data['chapterId']==span.chap_id) & (self.token_data['modified_paragraphId']==span.para_id) & (self.token_data['modified_tokenId']==span.start_token_id)
        end_span_filter = (self.token_data['chapterId']==span.chap_id) & (self.token_data['modified_paragraphId']==span.para_id) & (self.token_data['modified_tokenId']==span.end_token_id)
        start_span_token = self.token_data.loc[start_span_filter, 'normalizedWord'].values[0]
        end_span_token = self.token_data.loc[end_span_filter, 'normalizedWord'].values[0]

        # Check for alphanumeric characters or periods
        #pattern = re.compile(r'[A-Za-z0-9\.]')
        #if re.search(pattern, start_span_token):
        #    pdb.set_trace()
        #if re.search(pattern, end_span_token):
        #    pdb.set_trace()

        # Set tokens to recognizable quotes
        self.token_data.loc[start_span_filter, 'normalizedWord'] = '``'
        self.token_data.loc[start_span_filter, 'lemma'] = '``'
        self.token_data.loc[end_span_filter, 'normalizedWord'] = "''"
        self.token_data.loc[end_span_filter, 'lemma'] = "''"
        

    def run_booknlp_quote_attribution(self):
        """ Run booknlp-quote-attribution on modified token file.
            Saves to spot in modified_json_dirpath, which is read in evaluate_quotes()
        """ 
        # Run BookNLP on modified token file
        added_to_token_fpath = self.modified_token_output_dirpath.replace(self.original_token_output_dirpath.rstrip('/'), '')
        self.modified_json_dirpath = self.original_json_dirpath.rstrip('/') + added_to_token_fpath
        if not os.path.exists(self.modified_json_dirpath):
            os.mkdir(self.modified_json_dirpath)
        wrapper = BookNLPWrapper(self.modified_token_fpath, os.path.join(self.modified_json_dirpath, self.fandom_fname))
        wrapper.run()
