import scorer
import os
import pdb
import pandas as pd

from annotation import Annotation


class Evaluator():
    """ Superclass for evaluation, holding evaluation methods common to BookNLP,
        FanfictionNLP, and any other systems.
    """
    
    def __init__(self, fic_csv_dirpath,
                    evaluate_coref=False, evaluate_quotes=False,
                    coref_from='pipeline', quotes_from='pipeline',
                    run_quote_attribution=False,
                    coref_annotations_dirpath=None,
                    quote_annotations_dirpath=None,
                    predicted_coref_outpath=None,
                    predicted_quotes_outpath=None,
                    scores_outpath=None,
                    dataset_name=None,
                    coref_annotations_ext='_entity_clusters.csv',
                    quote_annotations_ext='_quote_attribution.csv'):
        
        self.fic_csv_dirpath = fic_csv_dirpath
        self.whether_evaluate_coref = evaluate_coref
        self.whether_evaluate_quotes = evaluate_quotes
        self.coref_from = coref_from
        self.quotes_from = quotes_from
        self.run_quote_attribution = run_quote_attribution
        self.coref_annotations_dirpath = coref_annotations_dirpath
        self.quote_annotations_dirpath = quote_annotations_dirpath
        self.predicted_coref_outpath = predicted_coref_outpath
        self.predicted_quotes_outpath = predicted_quotes_outpath
        self.scores_outpath = scores_outpath
        self.dataset_name = dataset_name
        self.coref_annotations_ext = coref_annotations_ext
        self.quote_annotations_ext = quote_annotations_ext

    def evaluate_coref(self, fandom_fname, fic_representation, save=True):
        """ Evaluate coref for a fic. 
            Args:
                save: save AnnotatedSpan objects in a pickled file in a tmp directory
        """
        # Load gold mentions
        gold = Annotation(self.coref_annotations_dirpath, fandom_fname, file_ext=self.coref_annotations_ext, fic_csv_dirpath=self.fic_csv_dirpath)
        gold.extract_annotated_spans()

        # Load predicted mentions
        fic_representation.extract_character_mentions(save_dirpath=self.predicted_coref_outpath)

        # Get scores
        coref_scores = scorer.coref_scores(fic_representation.character_mentions, gold.annotations, exact_match=True)
        print('\tCoref results:')
        for key in ['lea_f1', 'lea_precision', 'lea_recall']:
            print(f'\t\t{key}: {coref_scores[key]: .2%}')
        print()
        return coref_scores

    def evaluate_quotes(self, fandom_fname, fic_representation, save=True, exact_match=True, coref_from='system', quotes_from='system'):
        """ Evaluate quotes for a fic. 
            Args:
                save: save AnnotatedSpan quote objects in a pickled file in a tmp directory
        """
        # Quote extraction evaluation
        # Load gold quote spans
        gold = Annotation(self.quote_annotations_dirpath, fandom_fname, file_ext=self.quote_annotations_ext, fic_csv_dirpath=self.fic_csv_dirpath)
        gold.extract_annotated_spans()

        # Load predicted quote spans (from BookNLP output to AnnotatedSpan objects)
        fic_representation.extract_quotes(save_dirpath=self.predicted_quotes_outpath)

        # Get scores
        quote_scores = scorer.quote_scores(fic_representation.quotes, gold.annotations, exact_match=exact_match)
        print('\tQuote extraction results:')
        for key in ['extraction_f1', 'extraction_precision', 'extraction_recall']:
            print(f'\t\t{key}: {quote_scores[key]: .2%}')
        print('\tQuote attribution results:')
        for key in ['attribution_f1', 'attribution_precision', 'attribution_recall']:
            print(f'\t\t{key}: {quote_scores[key]: .2%}')
        print()
        return quote_scores

    def save_scores(self, scores, system_name, params):
        """ Save scores to a CSV in self.output_dirpath
            Args:
                scores: scores as a list of dicts
                params: list of parameters to add to self.dataset_name in output filepath
        """
        outpath = os.path.join(self.scores_outpath, self.dataset_name, system_name, f'{"_".join(params)}_scores.csv')
        if not os.path.exists(os.path.join(self.scores_outpath, self.dataset_name)):
            os.mkdir(os.path.join(self.scores_outpath, self.dataset_name))
        if not os.path.exists(os.path.join(self.scores_outpath, self.dataset_name, system_name)):
            os.mkdir(os.path.join(self.scores_outpath, self.dataset_name, system_name))
        pd.DataFrame(scores).to_csv(outpath, index=False)
        print(f"Saved scores to {outpath}")
