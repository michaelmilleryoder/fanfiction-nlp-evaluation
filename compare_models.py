""" Compare models on character coreference, quote attribution.
    Does a significance test """

import os
from configparser import ConfigParser
import argparse
import pdb
from statsmodels.stats.contingency_tables import mcnemar

from annotation import QuoteAnnotation
from pipeline_output import PipelineOutput
from booknlp_output import BookNLPOutput
from quote import Quote
import evaluation_utils as utils


class ModelComparer():

    def __init__(self, dataset_name, baseline_name, experimental_name, fic_csv_dirpath, 
                baseline_coref_dirpath='', baseline_quotes_dirpath='',
                experimental_coref_dirpath='', experimental_quotes_dirpath='',
                coref=False, quotes=False,
                coref_annotations_dirpath='', quote_annotations_dirpath=''):
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
        self.ordered_predictions = {} # in the order to run significance tests
        if self.evaluate_coref:
            self.ordered_predictions.update({'coref': {'baseline': [], 'experimental': [], 'gold': []}})
        if self.evaluate_quotes:
            self.ordered_predictions.update({'quotes': {'baseline': [], 'experimental': [], 'gold': []}})

    def compare(self):
        """ Runs significance test on models. """
        print(f"Comparing {self.baseline_name} to {self.experimental_name} on {self.dataset}")

        # Build ordered predictions for gold, baseline, experimental
        for fname in sorted(os.listdir(self.baseline_quotes_dirpath)):
            fandom_fname = fname.split('.')[0]
            self.compare_fic(fandom_fname)

        # Run significance test
        if self.evaluate_quotes:
            self.mcnemar_quotes()

    def compare_fic(self, fandom_fname):
        """ Extract gold, baseline and experimental predictions, save to self.predictions """

        if self.evaluate_quotes:
            # Quote extraction evaluation
            gold_quotes, baseline_quotes, experimental_quotes = self.load_fic_quotes(fandom_fname)

            # Take union of all marked quote spans, annotated with 'null' if don't attribute a character
            all_quote_spans = Quote.get_union_quotes(gold_quotes, baseline_quotes, experimental_quotes) # This is the order of the output table
            self.build_ordered_quote_predictions(all_quote_spans, gold_quotes, baseline_quotes, experimental_quotes)

    def build_ordered_quote_predictions(self, all_quote_spans, gold_quotes, baseline_quotes, experimental_quotes):
        """ Build ordered quote predictions, append to self.ordered_quote_predictions """
        for quote in all_quote_spans:
            matching_gold = [gold_quote for gold_quote in gold_quotes if quote.extraction_matches(gold_quote, exact=False)]
            if len(matching_gold) == 0:
                self.ordered_predictions['quotes']['gold'].append(quote.null_character_quote())
            else:
                self.ordered_predictions['quotes']['gold'].append(matching_gold[0])

            matching_baseline = [baseline_quote for baseline_quote in baseline_quotes if quote.extraction_matches(baseline_quote, exact=False)]
            if len(matching_baseline) == 0:
                self.ordered_predictions['quotes']['baseline'].append(quote.null_character_quote())
            else:
                self.ordered_predictions['quotes']['baseline'].append(matching_baseline[0])

            matching_experimental = [experimental_quote for experimental_quote in experimental_quotes if quote.extraction_matches(experimental_quote, exact=False)]
            if len(matching_experimental) == 0:
                self.ordered_predictions['quotes']['experimental'].append(quote.null_character_quote())
            else:
                self.ordered_predictions['quotes']['experimental'].append(matching_experimental[0])

    def load_fic_quotes(self, fandom_fname):
        """ Load quote predictions and gold quotes for a fic.
            Returns gold_quotes, baseline_quotes, experimental_quotes
        """
        gold_quotes = QuoteAnnotation(self.quote_annotations_dirpath, fandom_fname, fic_csv_dirpath=self.fic_csv_dirpath).quotes
        baseline_quotes = utils.load_pickle(self.baseline_quotes_dirpath, fandom_fname)
        experimental_quotes = utils.load_pickle(self.experimental_quotes_dirpath, fandom_fname)

        return gold_quotes, baseline_quotes, experimental_quotes

    def get_quotes(self, model_name, fandom_fname, model_args, model_kwargs):
        """ NOT USED """
        """ Returns a list of quote objects from the FicRepresentation subclass corresponding to the model.
            Args:
                model_name: which model output out of {pipeline, booknlp}
                fandom_fname: fandom_ficid name of fic
                *args: positional arguments specific to the model FicRepresentation constructor (excluding fandom_fname)

                **kwargs: arguments specific to the model FicRepresentation constructor
        """

        if model_name == 'pipeline':
            output = PipelineOutput(*model_args, fandom_fname)
            output.extract_quotes()

        elif model_name == 'gold':
            output = QuoteAnnotation(*model_args, fandom_fname, **model_kwargs)

        elif model_name == 'booknlp':
            output = BookNLPOutput(*model_args, fandom_fname, **model_kwargs)

        return output.quotes

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
                pdb.set_trace()
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
   #         'pipeline': {'quotes': '/projects/fanfiction-nlp/tmp/predicted_quotes'},
   #         'booknlp': {'quotes': '/projects/book-nlp/tmp/predicted_quotes'}

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

    baseline_coref_dirpath = baseline_config.get('Filepaths', 'predicted_entities_outpath')
    baseline_quotes_dirpath = baseline_config.get('Filepaths', 'predicted_quotes_outpath')
    experimental_coref_dirpath = experimental_config.get('Filepaths', 'predicted_entities_outpath')
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
