import os
import sys
import json
import pdb
import csv
import pandas as pd

from quote import Quote

def modify_paragraph_id(para_id, trouble_line):
    
    if para_id == trouble_line: # trouble line
        new_para_id = trouble_line + 2 # add 1 anyway for the index-0 issue
        
    if para_id >= trouble_line + 1:
        new_para_id = para_id
    
    else:
        new_para_id = para_id + 1 # add 1 since BookNLP starts with 0

    return new_para_id


class BookNLPOutput():
    """ Holds representation for the BookNLP processed output of a fic. """

    def __init__(self, token_output_dirpath, json_output_dirpath, fandom_fname, fic_csv_dirpath=None): 
        """ 
        Args:
            csv_dirpath: path to directory with corresponding original fic CSV
        
        """
        self.fandom_fname = fandom_fname
        self.original_token_data = pd.read_csv(os.path.join(token_output_dirpath, f'{fandom_fname}.txt.tokens'), sep='\t', quoting=csv.QUOTE_NONE)
        self.token_data = self.original_token_data.copy()
        self.json_fpath = os.path.join(json_output_dirpath, self.fandom_fname, 'book.id.book')
        if fic_csv_dirpath is not None:
            self.fic_csvpath = os.path.join(fic_csv_dirpath, f'{fandom_fname}.csv')
        self.fic_csv = None

    def load_fic_csv(self):
        self.fic_csv = pd.read_csv(self.fic_csvpath)

    def load_json_output(self):
        with open(self.json_fpath, 'r') as f:
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
        self.renumber_token_ids()

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
            first_tokenId = self.token_data.loc[(self.token_data['chapterId']==selected_chap_id) & (self.token_data['modified_paragraphId']==selected_para_id), 'tokenId'].tolist()[0]

            for i, gold_tok in enumerate(gold_tokens):

                current_booknlp_token = booknlp_tokens[i + total_offset]
                if not gold_tok == current_booknlp_token:

                    # Try adding tokens
                    added = current_booknlp_token
                    for offset in range(1, 4):
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

    def renumber_token_ids(self):
        para_token_lengths = self.token_data.groupby('full_paragraphId').size().tolist()
        new_tokenIds = sum([list(range(1, para_length+1)) for para_length in para_token_lengths], [])
        self.token_data['modified_tokenId'] = new_tokenIds

    def extract_entity_mentions(self, save_path=None):
        """ Extract character mentions from BookNLP output """
        selected_cols = ['chapterId', 'modified_paragraphId', 'modified_tokenId', 'characterId', 'originalWord']
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

        predicted_entities = {}

        prev_was_sequential = False
        prev_token_id_start = 0

        fic_id = int(fname.split('.')[0].split('_')[1])

        for row in list(mentions.itertuples()):
            chapter_id = row.chapterId
            para_id = row.modified_paragraphId
            character_id = row.characterId
            token_id_start = row.modified_tokenId

            if row.sequential: # Store last token ID
                if prev_was_sequential: # in the middle of an entity mention
                    continue
                else:
                    prev_was_sequential = True
                    prev_token_id_start = token_id_start
                    continue

            # Save entity mention
            if not fic_id in predicted_entities:
                predicted_entities[fic_id] = {}

            if not character_id in predicted_entities[fic_id]:
                predicted_entities[fic_id][character_id] = set()

            if prev_was_sequential:
                token_id_start = prev_token_id_start

            token_id_end = row.modified_tokenId

            predicted_entities[fic_id][character_id].add((chapter_id, para_id, token_id_start, token_id_end))

            prev_was_sequential = row.sequential

        if save_path:
            outpath = os.path.join(save_path, f'booknlp_{fic_id}.pkl')
            with open(os.path.join(outpath), 'wb') as f:
                pickle.dump(predicted_entities, f)

        return predicted_entities

    def extract_quotes(self):
        """ Extract Quote objects from BookNLP output representations. """

        self.quotes = []

        # Load BookNLP JSON
        self.load_json_output()

        # Get Quote objects from all characters from BookNLP JSON
        for char in self.json_data['characters']:
            char_name = char['names'][0]['n'] # take first name as name
            for utterance in char['speaking']:
                text = utterance['w']
                quote_length = len(text.split())
                matching_token_data = self.token_data.loc[self.token_data['tokenId'].isin(range(utterance['i'], utterance['i'] + quote_length))]
                modified_token_range = matching_token_data['modified_tokenId'].tolist()
                chap_id = matching_token_data['chapterId'].values[0]
                para_id = matching_token_data['modified_paragraphId'].values[0]
                token_start = modified_token_range[0]
                token_end = modified_token_range[-1]

                self.quotes.append(Quote(chap_id, para_id, token_start, token_end, char_name, text))
