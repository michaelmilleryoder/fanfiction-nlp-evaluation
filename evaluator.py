from quote import Quote
import scorer
from annotation import Annotation


class Evaluator():
    """ Superclass for evaluation, holding evaluation methods common to BookNLP,
        FanfictionNLP, and any other systems.
    """
    
    def __init__(self, fic_csv_dirpath,
                    evaluate_coref=False, evaluate_quotes=False,
                    coref_annotations_dirpath=None,
                    quote_annotations_dirpath=None,
                    predicted_coref_outpath=None,
                    predicted_quotes_outpath=None,
                    coref_annotations_ext='_entity_clusters.csv',
                    quote_annotations_ext='_quote_attribution.csv'):
        
        self.fic_csv_dirpath = fic_csv_dirpath
        self.whether_evaluate_coref = evaluate_coref
        self.whether_evaluate_quotes = evaluate_quotes
        self.coref_annotations_dirpath = coref_annotations_dirpath
        self.quote_annotations_dirpath = quote_annotations_dirpath
        self.predicted_coref_outpath = predicted_coref_outpath
        self.predicted_quotes_outpath = predicted_quotes_outpath
        self.coref_annotations_ext = coref_annotations_ext
        self.quote_annotations_ext = quote_annotations_ext

    def evaluate_coref(self, fandom_fname, fic_representation, save=True):
        """ Evaluate coref for a fic. 
            Args:
                save: save AnnotatedSpan objects in a pickled file in a tmp directory
        """
        
        # Load gold mentions
        gold = Annotation(self.coref_annotations_dirpath, fandom_fname, file_ext=self.coref_annotations_ext, fic_csv_dirpath=self.fic_csv_dirpath)

        # Load predicted mentions
        fic_representation.extract_character_mentions(save_dirpath=self.predicted_coref_outpath)

        # Print scores
        scorer.print_coref_scores(fic_representation.character_mentions, gold.annotations, exact_match=True)

    def evaluate_quotes(self, fandom_fname, fic_representation, save=True, exact_match=True):
        """ Evaluate quotes for a fic. 
            Args:
                save: save Quote objects in a pickled file in a tmp directory
        """
        
        # Quote extraction evaluation
        # Load gold quote spans
        gold = Annotation(self.quote_annotations_dirpath, fandom_fname, file_ext=self.quote_annotations_ext, fic_csv_dirpath=self.fic_csv_dirpath)

        # Load predicted quote spans (from BookNLP output to Quote objects)
        fic_representation.extract_quotes(save_dirpath=self.predicted_quotes_outpath)

        # Print scores
        scorer.print_quote_scores(fic_representation.quotes, gold.annotations, exact_match=exact_match)

