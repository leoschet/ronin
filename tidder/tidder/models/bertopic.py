"""Builder and director pattern for BERTopic model."""

from enum import Enum

import attrs
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import (
    BaseRepresentation,
    KeyBERTInspired,
    MaximalMarginalRelevance,
    TextGeneration,
)
from typing import Self
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
    _VectorizerMixin,
)
from transformers import pipeline

from tidder.models.clustering import ClusteringBuilder


# enum for topic representation aspects
class TopicRepresentationAspect(str, Enum):
    MAIN = "Main"
    REDUCED_KEYWORDS = "ReducedKeywords"
    TEXTUAL = "Textual"


class BERTopicDirector:
    """Define blueprints for building BERTopic models."""

    @classmethod
    def build_bertopic(
        cls,
        data: pd.DataFrame,
        target_column: str,
        random_state: int,
        k_range: range = range(2, 20),
        fit: bool = True,
        plot: bool = False,
    ) -> BERTopic:
        """Build BERTopic model."""
        builder = BERTopicBuilder(random_state=random_state, plot=plot, fit=False)
        builder.produce_sentence_transformer("all-MiniLM-L6-v2")

        embedded_data = builder.embedding_model.encode(
            data[target_column].values,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        builder.produce_identity_dimensionality_reduction().produce_kmeans(
            embedded_data=embedded_data, k_range=k_range, plot=plot
        ).produce_count_vectorizer(
            stop_words="english", ngram_range=(1, 3)
        ).produce_class_tfidf(
            reduce_frequent_words=True
        ).produce_keybert_representation(
            representation_id=TopicRepresentationAspect.REDUCED_KEYWORDS.value,
            top_n_words=50,
        ).produce_maximal_marginal_relevance_representation(
            representation_id=TopicRepresentationAspect.REDUCED_KEYWORDS.value,
            append=True,
            diversity=0.5,
        ).produce_keybert_representation(
            representation_id=TopicRepresentationAspect.TEXTUAL.value,
            top_n_words=50,
        ).produce_maximal_marginal_relevance_representation(
            representation_id=TopicRepresentationAspect.TEXTUAL.value,
            append=True,
            diversity=0.5,
        ).produce_text2text_representation(
            representation_id=TopicRepresentationAspect.TEXTUAL.value,
            append=True,
            model="google/flan-t5-large",
        ).build()

        if fit:
            print(f"{len(data[target_column].values)=}")
            print(f"{len(embedded_data.numpy())=}")
            builder.bertopic.fit(documents=data[target_column].tolist(), embeddings=embedded_data.numpy())

        return builder.bertopic

    @classmethod
    def build_soft_bertopic(
        cls,
        data: pd.DataFrame,
        target_column: str,
        random_state: int,
        fit: bool = True,
    ) -> BERTopic:
        """Build BERTopic model with soft clustering.

        Soft clustering allows a document to be part of multiple documents.
        It takes out the "decision" aspect in hard clustering and returns the probability of a
        document belonging to a topic.
        """
        builder = BERTopicBuilder(random_state=random_state)
        builder.produce_sentence_transformer("all-MiniLM-L6-v2")

        embedded_data = builder.embedding_model.encode(
            data[target_column].values,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        builder.produce_umap_dimensionality_reduction(
            n_components=50
        ).produce_count_vectorizer(
            stop_words="english", ngram_range=(1, 3)
        ).produce_class_tfidf(
            reduce_frequent_words=True
        ).produce_keybert_representation(
            representation_id=TopicRepresentationAspect.REDUCED_KEYWORDS.value,
            top_n_words=50,
        ).produce_maximal_marginal_relevance_representation(
            representation_id=TopicRepresentationAspect.REDUCED_KEYWORDS.value,
            append=True,
            diversity=0.5,
        ).produce_keybert_representation(
            representation_id=TopicRepresentationAspect.TEXTUAL.value, top_n_words=50
        ).produce_maximal_marginal_relevance_representation(
            representation_id=TopicRepresentationAspect.TEXTUAL.value,
            append=True,
            diversity=0.5,
        ).produce_text2text_representation(
            representation_id=TopicRepresentationAspect.TEXTUAL.value,
            append=True,
            model="google/flan-t5-large",
        ).build()

        if fit:
            builder.bertopic.fit(data[target_column].values, embedded_data.numpy())

        return builder.bertopic


@attrs.define
class BERTopicBuilder(ClusteringBuilder):
    """Build BERTopic class using the builder pattern.

    If any of the components are not produced, the defaults from BERTopic are used.
    """

    bertopic: BERTopic | None = attrs.field(default=None, init=False)

    vectorizer_model: _VectorizerMixin | None = attrs.field(default=None, init=False)
    ctfidf_model: ClassTfidfTransformer | None = attrs.field(default=None, init=False)
    representation_model: dict[
        str, BaseRepresentation | list[BaseRepresentation]
    ] | None = attrs.field(default=None, init=False)

    def build(self) -> BERTopic:
        self.bertopic = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.dimensionality_reduction_model,
            hdbscan_model=self.clustering_model,
            vectorizer_model=self.vectorizer_model,
            ctfidf_model=self.ctfidf_model,
            representation_model=self.representation_model,
        )

        return self.bertopic

    def produce_count_vectorizer(self, docs: list[str] = None, **kwargs) -> Self:
        """Use CountVectorizer for vectorization."""
        return self._produce_vectorizer(CountVectorizer(**kwargs), docs=docs)

    def produce_tfidf_vectorizer(self, docs: list[str] = None, **kwargs) -> Self:
        """Use TfidfVectorizer for vectorization."""
        return self._produce_vectorizer(TfidfVectorizer(**kwargs), docs=docs)

    def produce_hashing_vectorizer(self, docs: list[str] = None, **kwargs) -> Self:
        """Use HashingVectorizer for vectorization."""
        return self._produce_vectorizer(HashingVectorizer(**kwargs), docs=docs)

    def produce_class_tfidf(self, docs: list[str] = None, **kwargs) -> Self:
        """Use ClassTfidfTransformer for vectorization."""
        self.ctfidf_model = ClassTfidfTransformer(**kwargs)

        if self.bertopic is not None and docs is not None:
            self.bertopic.update_topics(docs, ctfidf_model=self.ctfidf_model)

        return self

    def produce_keybert_representation(
        self,
        docs: list[str] = None,
        representation_id: str = TopicRepresentationAspect.MAIN.value,
        append: bool = False,
        **kwargs,
    ) -> Self:
        """Use KeyBERTInspired for representation.

        Parameters
        ----------
        docs : list of str, optional
            Documents to use for updating the topics.
            If not None, we try to update the topics of the already built bertopic model.
        append : bool, default False
            Whether to chain representation models.
            Refer to: https://maartengr.github.io/BERTopic/getting_started/representation/representation.html#chain-models
        """
        return self._produce_representation(
            KeyBERTInspired(**kwargs),
            representation_id=representation_id,
            docs=docs,
            append=append,
        )

    def produce_maximal_marginal_relevance_representation(
        self,
        docs: list[str] = None,
        representation_id: str = TopicRepresentationAspect.MAIN.value,
        append: bool = False,
        **kwargs,
    ) -> Self:
        """Use MaximalMarginalRelevance for representation.

        Parameters
        ----------
        docs : list of str, optional
            Documents to use for updating the topics.
            If not None, we try to update the topics of the already built bertopic model.
        append : bool, default False
            Whether to chain representation models.
            Refer to: https://maartengr.github.io/BERTopic/getting_started/representation/representation.html#chain-models
        """
        return self._produce_representation(
            MaximalMarginalRelevance(**kwargs),
            representation_id=representation_id,
            docs=docs,
            append=append,
        )

    def produce_text2text_representation(
        self,
        docs: list[str] = None,
        representation_id: str = TopicRepresentationAspect.MAIN.value,
        append: bool = False,
        **kwargs,
    ) -> Self:
        """Use Text2Text for representation.

        Parameters
        ----------
        docs : list of str, optional
            Documents to use for updating the topics.
            If not None, we try to update the topics of the already built bertopic model.
        append : bool, default False
            Whether to chain representation models.
            Refer to: https://maartengr.github.io/BERTopic/getting_started/representation/representation.html#chain-models
        """
        # remove task from kwargs
        kwargs.pop("task", None)
        generation_pipeline = pipeline("text2text-generation", **kwargs)
        return self._produce_representation(
            TextGeneration(generation_pipeline),
            representation_id=representation_id,
            docs=docs,
            append=append,
        )

    def _produce_vectorizer(
        self, vectorizer: _VectorizerMixin, docs: list[str] = None
    ) -> Self:
        """Use CountVectorizer for vectorization."""
        self.vectorizer_model = vectorizer

        if self.bertopic is not None and docs is not None:
            self.bertopic.update_topics(docs, vectorizer_model=self.vectorizer_model)

        return self

    def _produce_representation(
        self,
        representation: BaseRepresentation,
        docs: list[str] = None,
        representation_id: str = TopicRepresentationAspect.MAIN.value,
        append: bool = False,
    ) -> Self:
        """Use CountVectorizer for vectorization.

        Keep in mind that errors may appear if the MAIN representation is a list.
        """
        if self.representation_model is None:
            self.representation_model = {}

        if representation_id not in self.representation_model:
            self.representation_model[representation_id] = []

        if append:
            if not isinstance(self.representation_model[representation_id], list):
                self.representation_model[representation_id] = [
                    self.representation_model[representation_id]
                ]

            self.representation_model[representation_id].append(representation)
        else:
            self.representation_model[representation_id] = representation

        if self.bertopic is not None and docs is not None:
            self.bertopic.update_topics(
                docs, representation_model=self.representation_model
            )

        return self
