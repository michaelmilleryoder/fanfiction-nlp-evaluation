from quote import Quote

class Evaluator():
    """ Superclass for evaluation, holding evaluation methods common to BookNLP,
        FanfictionNLP, and any other systems.
    """
    
    def __init__(self, fic_csv_dirpath,
                    evaluate_coref=False, evaluate_quotes=False,
                    coref_annotations_dirpath=None,
                    quote_annotations_dirpath=None,
                    predicted_entities_outpath=None,
                    predicted_quotes_outpath=None):
        
        self.fic_csv_dirpath = fic_csv_dirpath
        self.whether_evaluate_coref = evaluate_coref
        self.whether_evaluate_quotes = evaluate_quotes
        self.coref_annotations_dirpath = coref_annotations_dirpath
        self.quote_annotations_dirpath = quote_annotations_dirpath
        self.predicted_entities_outpath = predicted_entities_outpath
        self.predicted_quotes_outpath = predicted_quotes_outpath
