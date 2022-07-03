# Formarea Modelului de Limba cu ajutorul spacy
# Sursa: "https://github.com/explosion/spaCy/tree/master/spacy/lang/ro"

from spacy.lang.ro.tokenizer_exceptions  import TOKENIZER_EXCEPTIONS
from spacy.lang.ro.stop_words import STOP_WORDS
from spacy.lang.ro. punctuation import TOKENIZER_PREFIXES, TOKENIZER_INFIXES, TOKENIZER_SUFFIXES
from spacy.lang.ro.lex_attrs import LEX_ATTRS
from spacy.language import Language, BaseDefaults

class LbRomana(BaseDefaults):
    tokenizer_exceptii = TOKENIZER_EXCEPTIONS
    prefie = TOKENIZER_PREFIXES
    sufixe = TOKENIZER_SUFFIXES
    infixes = TOKENIZER_INFIXES
    lex_attr_getters = LEX_ATTRS
    cuv_stop = STOP_WORDS


class Romana(Language):
    lang = "ro" 
    Defaults = LbRomana

__all__ = ["Romana"] 