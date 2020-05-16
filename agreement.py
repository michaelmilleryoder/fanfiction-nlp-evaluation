""" Calculate inter-annotator agreement from span annotations """

from collections import namedtuple
from configparser import ConfigParser
import argparse
import numpy as np
import os
from sklearn.metrics import cohen_kappa_score
import pdb

from annotation import Annotation
from annotated_span import annotation_labels
import scorer


class Agreement():
    """ Holds settings, data and methods for calculating agreement between annotators on spans in multiple fics """

    def __init__(self, annotator_names, annotations_dirpath, fic_csv_dirpath=None):
        """ Args:
                annotator_names: list of the names of annotators. 
                    Should correspond to the suffixes of files in the annotations_dirpath as _name.csv.
                annotations_dirpath: dirpath to the annotations
                fic_csv_dirpath: dirpath to the text CSVs, used to match the spans
        """
    
        self.annotators = annotator_names
        self.annotations_dirpath = annotations_dirpath
        self.fic_csv_dirpath = fic_csv_dirpath

    def get_fnames(self, annotator):
        """ Returns all fandom prefixes of file names with the annotator as a suffix _<name>.csv
            in the annotation dir
        """
        fandom_fnames = ['_'.join(fname.split('_')[:2]) for fname in os.listdir(self.annotations_dirpath) if fname.endswith(f'_{annotator}.csv')]
        return set(fandom_fnames)

    def calculate_agreement(self, span_type, extraction=True, attribution=True, metric='cohen'):
        """ Calculate average agreement on annotations on fics
            Args:
                span_type: {coref, quotes}
        """

        fandom_fnames = self.get_fnames(self.annotators[0])
        agreements = [] # list of namedtuples (extraction, attribution)

        for annotator in self.annotators[1:]:
            fandom_fnames = fandom_fnames.intersection(self.get_fnames(annotator))
        print(f'Agreement between {" and ".join(self.annotators)}:')
        for fandom_fname in sorted(fandom_fnames):
            agreements.append(self.fic_agreement(fandom_fname, span_type, extraction=extraction, attribution=attribution, metric=metric))
            print(f"\tAgreement on {fandom_fname}: \n{agreements[-1]}")
        if extraction:
            avg_extraction_agreement = np.mean([scores.extraction for scores in agreements])
            # Print average agreement
            print()
            print(f'\Avg extraction {metric}: {avg_extraction_agreement}')
        if attribution:
            avg_matched_attribution_agreement = np.mean([scores.matched_attribution for scores in agreements])
            avg_total_attribution_agreement = np.mean([scores.total_attribution for scores in agreements])
            print(f'\tAvg attribution on matched extractions {metric}: {avg_matched_attribution_agreement}')
            print(f'\tAvg attribution on total extractions {metric}: {avg_total_attribution_agreement}')

    def fic_agreement(self, fandom_fname, span_type, extraction=True, attribution=True, metric='cohen'):
        """ Calculate annotation on a fic. 
            Returns namedtuple (extraction, attribution) for that fic.
        """

        # Load annotations
        fic_annotations = {}
        if span_type == 'coref':
            file_ext = 'entity_clusters'
        elif span_type == 'quotes':
            file_ext = 'quote_attribution'
        for annotator in self.annotators:
            fic_annotations[annotator] = Annotation(self.annotations_dirpath, fandom_fname, file_ext=f'_{file_ext}_{annotator}.csv', fic_csv_dirpath=self.fic_csv_dirpath)

        # Extraction agreement
        if extraction:
            extraction_agreement = self.bio_token_agreement(fic_annotations, metric=metric)
        else:
            extraction_agreement = None

        # Attribution agreement
        if attribution:
            total_attribution_agreement, matched_attribution_agreement = self.attribution_agreement(fic_annotations, metric=metric)
        else:
            total_attribution_agreement, matched_attribution_agreement = None, None

        AgreementScores = namedtuple('AgreementScores', ['extraction', 'total_attribution', 'matched_attribution'])
        scores = AgreementScores(extraction=extraction_agreement, total_attribution=total_attribution_agreement, matched_attribution=matched_attribution_agreement)
        return scores

    def bio_token_agreement(self, fic_annotations, metric='cohen'):
        """ Calculate BIO token-level span extraction agreement on a fic """
        bio = {} # annotator: [token_bio]
        for annotator in self.annotators:
            bio[annotator] = fic_annotations[annotator].annotation_bio()
        score = 0
        if metric == 'cohen':
            assert len(bio) == 2
            score = cohen_kappa_score(bio[list(bio.keys())[0]], bio[list(bio.keys())[1]])
        return score

    def attribution_agreement(self, fic_annotations, metric='cohen'):      
        """ Calculate agreement on character annotations of spans 
            Returns total_attribution_agreement, matched_attribution_agreement
        """
        assert len(self.annotators) == 2 # right now can't handle more than 2 annotators
        annotations1 = fic_annotations[list(fic_annotations.keys())[0]]
        annotations2 = fic_annotations[list(fic_annotations.keys())[1]]
        matched_attributions, mismatched_attributions, mismatched_extractions = annotation_labels(annotations1.annotations, annotations2.annotations)
        matched_attribution_agreement, total_attribution_agreement = scorer.score_attribution_labels(matched_attributions, mismatched_attributions, mismatched_extractions, metric=metric)
        return total_attribution_agreement, matched_attribution_agreement


def main():
    parser = argparse.ArgumentParser(description='Get inter-annotator agreement between annotations')
    parser.add_argument('config_fpath', nargs='?', help='File path of config file with annotations dirpath, fic csv dirpath, and coref/quote settings')
    parser.add_argument('--annotators', nargs='+', dest='annotators')
    parser.add_argument('--coref', dest='evaluate_coref', action='store_true')
    parser.set_defaults(evaluate_coref=False)
    parser.add_argument('--quotes', dest='evaluate_quotes', action='store_true')
    parser.set_defaults(evaluate_quotes=False)
    args = parser.parse_args()

    config = ConfigParser(allow_no_value=False)
    config.read(args.config_fpath)
    fic_csv_dirpath = config.get('Filepaths', 'fic_csv_dirpath')
    if args.evaluate_coref:
        annotations_dirpath = config.get('Filepaths', 'coref_annotations_dirpath')
        agreement = Agreement(args.annotators, annotations_dirpath, fic_csv_dirpath=fic_csv_dirpath)
        agreement.calculate_agreement('coref')
    if args.evaluate_quotes:
        annotations_dirpath = config.get('Filepaths', 'quote_annotations_dirpath')
        agreement = Agreement(args.annotators, annotations_dirpath, fic_csv_dirpath=fic_csv_dirpath)
        agreement.calculate_agreement('quotes')


if __name__ == '__main__':
    main()
