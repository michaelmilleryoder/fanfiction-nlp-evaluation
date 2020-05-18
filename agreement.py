""" Calculate inter-annotator agreement from span annotations """

from collections import namedtuple
from configparser import ConfigParser
import argparse
import numpy as np
import os
from sklearn.metrics import cohen_kappa_score
import pdb

from annotation import Annotation
from annotated_span import annotation_labels, spans_union
import scorer


class Agreement():
    """ Holds settings, data and methods for calculating agreement between annotators on spans in multiple fics """

    def __init__(self, annotator_names, annotations_dirpath, fic_csv_dirpath=None, span_type=None):
        """ Args:
                annotator_names: list of the names of annotators. 
                    Should correspond to the suffixes of files in the annotations_dirpath as _name.csv.
                annotations_dirpath: dirpath to the annotations
                fic_csv_dirpath: dirpath to the text CSVs, used to match the spans
        """
    
        self.annotators = annotator_names
        self.annotations_dirpath = annotations_dirpath
        self.fic_csv_dirpath = fic_csv_dirpath
        self.span_type = span_type
        self.fic_annotations = {} # {fandom_fname: {annotator: Annotation}}
        self.get_fnames()

    def get_annotator_fnames(self, annotator):
        """ Returns all fandom prefixes of file names with the annotator as a suffix _<name>.csv
            in the annotation dir
        """
        fandom_fnames = ['_'.join(fname.split('_')[:2]) for fname in os.listdir(self.annotations_dirpath) if fname.endswith(f'_{annotator}.csv')]
        return set(fandom_fnames)

    def get_fnames(self):
        """ Saves intersection of fandom_fnames annotated to self.fandom_fnames
        """
        self.fandom_fnames = self.get_annotator_fnames(self.annotators[0])
        for annotator in self.annotators[1:]:
            self.fandom_fnames = self.fandom_fnames.intersection(self.get_annotator_fnames(annotator))

    def calculate_agreement(self, span_type, extraction=True, attribution=True, metric='cohen'):
        """ Calculate average agreement on annotations on fics
            Args:
                span_type: {coref, quotes}
        """

        agreements = [] # list of namedtuples (extraction, attribution)

        print(f'Agreement between {" and ".join(self.annotators)}:')
        for fandom_fname in sorted(self.fandom_fnames):
            agreements.append(self.fic_agreement(fandom_fname, span_type, extraction=extraction, attribution=attribution, metric=metric))
            print(f"\t{fandom_fname}:") 
            for name, value in agreements[-1]._asdict().items():
                print(f'\t\t{name}: {value}')
        if extraction:
            avg_extraction_agreement = np.mean([scores.extraction for scores in agreements])
            # Print average agreement
            print()
            print(f'\tAvg extraction {metric}: {avg_extraction_agreement}')
        if attribution:
            avg_matched_attribution_agreement = np.mean([scores.matched_attribution for scores in agreements])
            avg_total_attribution_agreement = np.mean([scores.total_attribution for scores in agreements])
            print(f'\tAvg attribution on total extractions {metric}: {avg_total_attribution_agreement}')
            print(f'\tAvg attribution on matched extractions {metric}: {avg_matched_attribution_agreement}')

    def fic_agreement(self, fandom_fname, span_type, extraction=True, attribution=True, metric='cohen'):
        """ Calculate annotation on a fic. 
            Returns namedtuple (extraction, attribution) for that fic.
        """

        # Load annotations
        if span_type == 'coref':
            file_ext = 'entity_clusters'
        elif span_type == 'quotes':
            file_ext = 'quote_attribution'
        self.fic_annotations[fandom_fname] = {}
        for annotator in self.annotators:
            self.fic_annotations[fandom_fname][annotator] = Annotation(self.annotations_dirpath, fandom_fname, file_ext=f'_{file_ext}_{annotator}.csv', fic_csv_dirpath=self.fic_csv_dirpath)
            self.fic_annotations[fandom_fname][annotator].extract_annotated_spans()

        # Extraction agreement
        if extraction:
            extraction_agreement = self.bio_token_agreement(fandom_fname, metric=metric)
        else:
            extraction_agreement = None

        # Attribution agreement
        if attribution:
            total_attribution_agreement, matched_attribution_agreement = self.attribution_agreement(fandom_fname, metric=metric)
        else:
            total_attribution_agreement, matched_attribution_agreement = None, None

        AgreementScores = namedtuple('AgreementScores', ['extraction', 'total_attribution', 'matched_attribution'])
        scores = AgreementScores(extraction=extraction_agreement, total_attribution=total_attribution_agreement, matched_attribution=matched_attribution_agreement)
        return scores

    def bio_token_agreement(self, fandom_fname, metric='cohen'):
        """ Calculate BIO token-level span extraction agreement on a fic """
        bio = {} # annotator: [token_bio]
        for annotator in self.annotators:
            bio[annotator] = self.fic_annotations[fandom_fname][annotator].annotation_bio()
        score = 0
        if metric == 'cohen':
            assert len(bio) == 2
            annotator1_bio = bio[list(bio.keys())[0]]
            annotator2_bio = bio[list(bio.keys())[1]]
            if annotator1_bio == annotator2_bio:
                score = 1
            else:
                score = cohen_kappa_score(annotator1_bio, annotator2_bio)
        return score

    def attribution_agreement(self, fandom_fname, metric='cohen'):      
        """ Calculate agreement on character annotations of spans 
            Returns total_attribution_agreement, matched_attribution_agreement
        """
        assert len(self.annotators) == 2 # right now can't handle more than 2 annotators
        annotations1 = self.fic_annotations[fandom_fname][self.annotators[0]]
        annotations2 = self.fic_annotations[fandom_fname][self.annotators[1]]
        #annotations2 = fic_annotations[list(fic_annotations.keys())[1]]
        matched_attributions, mismatched_attributions, mismatched_extractions = annotation_labels(annotations1.annotations, annotations2.annotations)
        matched_attribution_agreement, total_attribution_agreement = scorer.score_attribution_labels(matched_attributions, mismatched_attributions, mismatched_extractions, metric=metric)
        return total_attribution_agreement, matched_attribution_agreement

    def build_gold_annotations(self):
        """ Merge annotations, save as gold annotations.
            Take union of all extractions, discard mismatched attributions
        """

        # Merge annotations from annotators
        self.gold_fic_annotations = {}
        for fandom_fname in sorted(self.fandom_fnames):
            self.build_fic_gold_annotations(fandom_fname) # saves to self.gold_fic_annotations[fandom_fname]

        # Save out
        for fandom_fname, annotations in sorted(self.gold_fic_annotations.items()):
            if self.span_type == 'coref':
                gold_annotations = Annotation(self.annotations_dirpath, fandom_fname, file_ext='_entity_clusters.csv')
            elif self.span_type == 'quotes':
                gold_annotations = Annotation(self.annotations_dirpath, fandom_fname, file_ext='_quote_attribution.csv')
            gold_annotations.save_annotated_spans(annotations)
            

    def build_fic_gold_annotations(self, fandom_fname):
        """ Merge annotations
        """
        all_annotations = [annotation.annotations for annotation in self.fic_annotations[fandom_fname].values()]
        all_extractions = spans_union(all_annotations, exact=True)
        self.gold_fic_annotations[fandom_fname] = all_extractions


def main():
    parser = argparse.ArgumentParser(description='Get inter-annotator agreement between annotations')
    parser.add_argument('config_fpath', nargs='?', help='File path of config file with annotations dirpath, fic csv dirpath, and coref/quote settings')
    parser.add_argument('--annotators', nargs='+', dest='annotators')
    parser.add_argument('--coref', dest='evaluate_coref', action='store_true')
    parser.set_defaults(evaluate_coref=False)
    parser.add_argument('--quotes', dest='evaluate_quotes', action='store_true')
    parser.set_defaults(evaluate_quotes=False)
    parser.add_argument('--build-gold', dest='build_gold', action='store_true', help="Build gold annotations from annotation agreements")
    parser.set_defaults(build_gold=False)
    args = parser.parse_args()

    config = ConfigParser(allow_no_value=False)
    config.read(args.config_fpath)
    fic_csv_dirpath = config.get('Filepaths', 'fic_csv_dirpath')
    if args.evaluate_coref:
        annotations_dirpath = config.get('Filepaths', 'coref_annotations_dirpath')
        agreement = Agreement(args.annotators, annotations_dirpath, fic_csv_dirpath=fic_csv_dirpath, span_type='coref')
        agreement.calculate_agreement('coref')
        if args.build_gold:
            agreement.build_gold_annotations()
    if args.evaluate_quotes:
        annotations_dirpath = config.get('Filepaths', 'quote_annotations_dirpath')
        agreement = Agreement(args.annotators, annotations_dirpath, fic_csv_dirpath=fic_csv_dirpath, span_type='quotes')
        agreement.calculate_agreement('quotes')
        if args.build_gold:
            agreement.build_gold_annotations()


if __name__ == '__main__':
    main()
