from functools import partial
from typing import Iterable, List, Optional

import attrs
import pandas as pd
import spacy

from .base import Transformer


@attrs.define
class DummySentenceTokenizer(Transformer):
    """
    Initialize Dummy Sentence Tokenizer.

    Attributes
    ----------
    input_column : str
        Name of column with target text to split into sentences.
    output_column : str, default to "sentences"
        Name of column to hold the sentences.
    explode : bool
        Whether to explode dataframe to have one sentence per row or not.
    original_index_column : str, default to "invoice_file_id"
        Name of column to store old indices after exploding dataframe.
    """

    input_column: str
    output_column: str = attrs.field(default="sentences")
    explode: bool = attrs.field(default=True)
    original_index_column: str = attrs.field(default="invoice_file_id")
    split_token: str = attrs.field(default="\n")

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Split text into sentences."""
        X[self.output_column] = X[self.input_column].apply(lambda t: t.split(self.split_token))

        if self.explode:
            X = X.explode(self.output_column)
            X.reset_index(names=self.original_index_column, inplace=True)

        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop column with sentences."""
        X.drop(columns=self.output_column, inplace=True)
        return X


@attrs.define
class SpacyTokenizer(Transformer):
    r"""
    Tokenize text with Spacy.

    Attributes
    ----------
    input_column : str
        Name of column with input text.
    output_column : str, default to "tokens"
        Name of column to hold the output from Spacy's pipeline.
    nlp_pipeline : `spacy.language.Language`
        Spacy's pipeline.
    remove_stopwords : bool
        Whether to remove stopwords or not.
    extra_stopwords : list of str
        Extra stop words.
    """

    input_column: str
    output_column: str = attrs.field(default="tokens")
    nlp_pipeline: spacy.language.Language = attrs.field(
        factory=partial(spacy.load, "en_core_web_trf")
    )
    clean_tokens: bool = attrs.field(default=True)
    remove_stopwords: bool = attrs.field(default=True)
    extra_stopwords: List[str] = attrs.field(factory=list)

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply tokenization."""
        X[self.output_column] = X[self.input_column].apply(self._tokenize)
        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop column with tokenization result."""
        X.drop(columns=self.output_column, inplace=True)
        return X

    def _tokenize(self, text) -> List[str]:
        """Tokenize text."""
        tokens = []
        for tok in self.nlp_pipeline(
            text,
            disable=[
                "tagger",
                "parser",
                "attribute_ruler",
                "ner",
            ],
        ):
            clean_token = tok.text
            if self.clean_tokens:
                clean_token = tok.lemma_.lower()
            if not self.remove_stopwords or (
                not tok.is_stop and clean_token not in self.extra_stopwords
            ):
                tokens.append(clean_token)

        return tokens


@attrs.define
class SpacySentenceTokenizer(SpacyTokenizer):
    r"""
    Spacy sentence tokenizer.

    Attributes
    ----------
    input_column : str
        Name of column with input text.
    output_column : str, default to "tokens"
        Name of column to hold the output from Spacy's pipeline.
    nlp_pipeline : `spacy.language.Language`
        Spacy's pipeline.
    remove_stopwords : bool
        Whether to remove stopwords or not.
    extra_stopwords : list of str
        Extra stop words.
    """

    def __attrs_post_init__(self):
        self.nlp_pipeline.add_pipe("sentencizer")

    def _tokenize(self, text) -> List[str]:
        """Tokenize text."""
        doc = self.nlp_pipeline(
            text,
            disable=[
                "tagger",
                "parser",
                "attribute_ruler",
                "ner",
            ],
        )

        sentences = []
        for sent in doc.sents:
            tokens = []
            for tok in sent:
                clean_token = tok.text
                if self.clean_tokens:
                    clean_token = tok.lemma_.lower()
                if not self.remove_stopwords or (
                    not tok.is_stop and clean_token not in self.extra_stopwords
                ):
                    tokens.append(clean_token)
            sentences.append(tokens)

        return sentences
