""" Compare models on character coreference, quote attribution.
    Loads predictions from those saved out after running 
    evaluate_{pipeline,booknlp}.py.
    Does a significance test 
"""

import os
from configparser import ConfigParser
import argparse
import pdb
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import ttest_rel, ttest_ind

from annotation import Annotation
from pipeline_output import PipelineOutput
from booknlp_output import BookNLPOutput
import evaluation_utils as utils
from annotated_span import characters_match, spans_union


class ModelComparer():

    def __init__(self, dataset_name, baseline_name, experimental_name, fic_csv_dirpath, 
                baseline_coref_dirpath='', baseline_quotes_dirpath='',
                experimental_coref_dirpath='', experimental_quotes_dirpath='',
                coref=False, quotes=False,
                coref_annotations_dirpath='', quote_annotations_dirpath='',
                coref_annotations_ext='_entity_clusters.csv',
                quote_annotations_ext='_quote_attribution.csv'
                ):
        """ Args:
                dataset_name: name of the dataset.
                baseline_name: name of the baseline model
                experimental_name: name of the model to be compared to the baseline
                baseline_config_fpath: path to config file for baseline
                experimental_config_fpath: path to config file for experimental
                fic_csv_dirpath: path to actual fic text CSVs
                coref: whether to compare coreference for models
                quotes: whether to compare quote attribution for models
                coref_annotations_dirpath: path to directory with coref annotations
                quote_annotations_dirpath: path to directory with quote annotations
        """

        self.dataset = dataset_name
        self.baseline_name = baseline_name
        self.experimental_name = experimental_name
        self.fic_csv_dirpath = fic_csv_dirpath
        self.evaluate_coref = coref
        self.evaluate_quotes = quotes
        self.baseline_coref_dirpath = baseline_coref_dirpath
        self.baseline_quotes_dirpath = baseline_quotes_dirpath
        self.experimental_coref_dirpath = experimental_coref_dirpath
        self.experimental_quotes_dirpath = experimental_quotes_dirpath
        self.coref_annotations_dirpath = coref_annotations_dirpath
        self.quote_annotations_dirpath = quote_annotations_dirpath
        self.coref_annotations_ext = coref_annotations_ext
        self.quote_annotations_ext = quote_annotations_ext
        self.ordered_predictions = {} # in the order to run significance tests
        if self.evaluate_coref:
            self.ordered_predictions.update({'coref': {'baseline': [], 'experimental': [], 'gold': []}})
        if self.evaluate_quotes:
            self.ordered_predictions.update({'quotes': {'baseline': [], 'experimental': [], 'gold': []}})

    def compare(self):
        """ Runs significance test on models. """
        print(f"Comparing {self.baseline_name} to {self.experimental_name} on {self.dataset}")

        # Build ordered predictions for gold, baseline, experimental
        for fname in sorted(os.listdir(self.fic_csv_dirpath)):
            fandom_fname = fname.split('.')[0]
            self.compare_fic(fandom_fname)

        # Run significance test
        if self.evaluate_coref:
            self.ttest('coref')

        if self.evaluate_quotes:
            self.ttest('quotes')

    def compare_fic(self, fandom_fname):
        """ Extract gold, baseline and experimental predictions, save to self.predictions """

        if self.evaluate_coref:
            # Load coref predictions
            gold_spans, baseline_spans, experimental_spans = self.load_fic_spans(fandom_fname, self.coref_annotations_dirpath, self.baseline_coref_dirpath, self.experimental_coref_dirpath, self.coref_annotations_ext)

            # Build list of annotated spans, 'null' if don't extract the span
            self.build_ordered_predictions(gold_spans, baseline_spans, experimental_spans, span_type='coref')

        if self.evaluate_quotes:
            # Load quote attributions
            gold_quotes, baseline_quotes, experimental_quotes = self.load_fic_spans(fandom_fname, self.quote_annotations_dirpath, self.baseline_quotes_dirpath, self.experimental_quotes_dirpath, self.quote_annotations_ext)

            # Build list of annotated spans, 'null' if don't extract the span
            self.build_ordered_predictions(gold_quotes, baseline_quotes, experimental_quotes, over='all', span_type='quotes')

    def build_ordered_predictions(self, gold_spans, baseline_spans, experimental_spans, over='all', span_type='coref'):
        """ Build ordered predictions for quotes or coref, append to self.ordered_quote_predictions.
            Args:
                over: What set of annotations each instance is drawn from, {'all', 'gold'}. All is the union of all quotes extracted by any system, in which case the gold prediction is NULL for the speaker
                span_type: {'coref', 'quotes'} just to identify in the output data structure
        """
        if over == 'gold':
            spans = gold_spans
        elif over == 'all':
            spans = spans_union([gold_spans, baseline_spans, experimental_spans], attribution_conflicts='ignore')

        for span in spans:
            # Gold classification
            if over == 'gold':
                self.ordered_predictions[span_type]['gold'].append(span)
            elif over == 'all':
                matching_gold = [gold_span for gold_span in gold_spans if span.span_matches(gold_span, exact=True)]
                if len(matching_gold) == 0:
                    self.ordered_predictions[span_type]['gold'].append(span.null_span())
                else:
                    self.ordered_predictions[span_type]['gold'].append(matching_gold[0])

            # Baseline classification
            matching_baseline = [baseline_span for baseline_span in baseline_spans if span.span_matches(baseline_span, exact=True)]
            if len(matching_baseline) == 0:
                self.ordered_predictions[span_type]['baseline'].append(span.null_span())
            else:
                self.ordered_predictions[span_type]['baseline'].append(matching_baseline[0])

            # Experimental classification
            matching_experimental = [experimental_span for experimental_span in experimental_spans if span.span_matches(experimental_span, exact=True)]
            if len(matching_experimental) == 0:
                self.ordered_predictions[span_type]['experimental'].append(span.null_span())
            else:
                self.ordered_predictions[span_type]['experimental'].append(matching_experimental[0])

    def load_fic_spans(self, fandom_fname, gold_dirpath, baseline_dirpath, experimental_dirpath, gold_annotations_ext):
        """ Load quote or coref predictions and gold spans for a fic.
            Returns gold_spans, baseline_spans, experimental_spans
        """
        gold_annotation = Annotation(gold_dirpath, fandom_fname, file_ext=gold_annotations_ext, fic_csv_dirpath=self.fic_csv_dirpath)
        gold_annotation.extract_annotated_spans()
        gold_spans = gold_annotation.annotations
        baseline_spans = utils.load_pickle(baseline_dirpath, fandom_fname)
        experimental_spans = utils.load_pickle(experimental_dirpath, fandom_fname)

        return gold_spans, baseline_spans, experimental_spans

    def ttest(self, span_type):
        """ Prints related t-test between error distributions of 2 classifiers.
            Args:
                span_type: What key in self.ordered_predictions to use.
                            Will compare baseline and experimental predictions
        """
        baseline_correct = [int(characters_match(pred_span.annotation, gold_span.annotation)) for pred_span, gold_span in zip(self.ordered_predictions[span_type]['baseline'], self.ordered_predictions[span_type]['gold'])]
        experimental_correct = [int(characters_match(pred_span.annotation, gold_span.annotation)) for pred_span, gold_span in zip(self.ordered_predictions[span_type]['experimental'], self.ordered_predictions[span_type]['gold'])]
        print(f'\tBaseline ({self.baseline_name}) correct: {sum(baseline_correct)} / {len(baseline_correct)}')
        print(f'\tExperimental ({self.experimental_name}) correct: {sum(experimental_correct)} / {len(experimental_correct)}')

        result = ttest_rel(baseline_correct, experimental_correct)
        #result = ttest_ind(baseline_correct, experimental_correct)
        print('t-test statistic=%.3f, p-value=%.10f' % (result.statistic, result.pvalue))

    def mcnemar_quotes(self):
        """ Run McNemar test on quotes. """

        a = 0 # Both correct
        b = 0 # Baseline correct, experiment incorrect
        c = 0 # Baseline incorrect, experiment correct
        d = 0 # Both incorrect
        for baseline_quote, experimental_quote, gold_quote in zip(self.ordered_predictions['quotes']['baseline'], self.ordered_predictions['quotes']['experimental'], self.ordered_predictions['quotes']['gold']):
            if utils.characters_match(baseline_quote.speaker, gold_quote.speaker) and utils.characters_match(experimental_quote.speaker, gold_quote.speaker):
                a += 1
            elif utils.characters_match(baseline_quote.speaker, gold_quote.speaker) and not utils.characters_match(experimental_quote.speaker, gold_quote.speaker):
                b += 1
            elif not utils.characters_match(baseline_quote.speaker, gold_quote.speaker) and utils.characters_match(experimental_quote.speaker, gold_quote.speaker):
                c += 1
            else:
                d += 1
                
        table = [[a, b],
                 [c, d]]

        # Example of calculating the mcnemar test
        # calculate mcnemar test
        result = mcnemar(table, correction=False)
        # summarize the finding
        print('statistic=%.3f, p-value=%.6f' % (result.statistic, result.pvalue))
        # interpret the p-value
        alpha = 0.05
        if result.pvalue > alpha:
                print('Same proportions of errors (fail to reject H0)')
        else:
                print('Different proportions of errors (reject H0)')
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Run significance tests between model predictions')
    parser.add_argument('dataset_name', nargs='?', help='Name of dataset')
    parser.add_argument('baseline_model', nargs='?', help='Name of baseline model in {pipeline, booknlp}')
    parser.add_argument('experimental_model', nargs='?', help='Name of experimental model')
    parser.add_argument('baseline_config_fpath', nargs='?', help='File path of config file for baseline model')
    parser.add_argument('experimental_config_fpath', nargs='?', help='File path of config file for experimental (other) model')
    parser.add_argument('--coref', dest='evaluate_coref', action='store_true')
    parser.set_defaults(evaluate_coref=False)
    parser.add_argument('--quotes', dest='evaluate_quotes', action='store_true')
    parser.set_defaults(evaluate_quotes=False)
    args = parser.parse_args()

    baseline_config = ConfigParser(allow_no_value=False)
    baseline_config.read(args.baseline_config_fpath)

    experimental_config = ConfigParser(allow_no_value=False)
    experimental_config.read(args.experimental_config_fpath)

    fic_csv_dirpath = baseline_config.get('Filepaths', 'fic_csv_dirpath')
    coref_annotations_dirpath = baseline_config.get('Filepaths', 'coref_annotations_dirpath')
    quote_annotations_dirpath = baseline_config.get('Filepaths', 'quote_annotations_dirpath')

    baseline_coref_dirpath = baseline_config.get('Filepaths', 'predicted_coref_outpath')
    baseline_quotes_dirpath = baseline_config.get('Filepaths', 'predicted_quotes_outpath')
    experimental_coref_dirpath = experimental_config.get('Filepaths', 'predicted_coref_outpath')
    experimental_quotes_dirpath = experimental_config.get('Filepaths', 'predicted_quotes_outpath')

    comparer = ModelComparer(args.dataset_name, args.baseline_model, args.experimental_model, fic_csv_dirpath, 
                baseline_coref_dirpath=baseline_coref_dirpath,
                baseline_quotes_dirpath=baseline_quotes_dirpath,
                experimental_coref_dirpath=experimental_coref_dirpath,
                experimental_quotes_dirpath=experimental_quotes_dirpath,
                coref=args.evaluate_coref, quotes=args.evaluate_quotes,
                coref_annotations_dirpath=coref_annotations_dirpath,
                quote_annotations_dirpath=quote_annotations_dirpath
                )

    comparer.compare()
    

if __name__ == '__main__':
    main()
