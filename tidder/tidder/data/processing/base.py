"""Base utilities for document processing.

This module serves as the interface between different processing strategies
(i.e.: Form Recognizer and Unstructured).
"""


import json
import urllib.request
from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, Callable

import attrs

from tidder.generic import InstanceableGeneric, T
from tidder.data.post_processing.base import PostProcessor


@attrs.define
class ProcessingStatus:
    successful: bool
    error: Exception = None


@attrs.define
class ProcessingResult(Generic[T]):
    processing_status: ProcessingStatus
    value: T | None = None


class EProcessingStrategy(str, Enum):
    """A processing strategy.

    Attributes
    ----------
    FORM_RECOGNIZER : str
        The Form Recognizer processing strategy.
    UNSTRUCTURED : str
        The Unstructured processing strategy.
    """

    FORM_RECOGNIZER = "Form Recognizer"
    UNSTRUCTURED = "Unstructured"


@attrs.define
class Processor(InstanceableGeneric[T], ABC):
    """Abstract class for processing strategies."""

    document_post_processors: list[PostProcessor[T]] = attrs.field(factory=list)

    def __attrs_post_init__(self) -> None:
        self._validate_post_processors()

    def process_local_document(self, file_path: str) -> ProcessingResult[T]:
        """Process a local document.

        Parameters
        ----------
        file_path : str
            Path to the local file.

        Returns
        -------
        ProcessingResult[T]
            Processed document.
        """
        return self._process(file_path, self._process_local_document)

    def process_remote_document(self, file_url: str) -> ProcessingResult[T]:
        """Process a remote document.

        Parameters
        ----------
        file_url : str
            URL to the file.

        Returns
        -------
        ProcessingResult[T]
            Processed document.
        """
        return self._process(file_url, self._process_remote_document)

    def post_process(self, processed_document: T) -> T:
        """Post-process a T instance.

        Parameters
        ----------
        processed_document : T
            Document to be post processed.

        Returns
        -------
        T
            Post-processed document.
        """
        for post_processor in self.document_post_processors:
            processed_document = post_processor.apply(processed_document)
        return processed_document

    @abstractmethod
    def _process_local_document(self, file_path: str) -> T:
        """Process a document.

        Parameters
        ----------
        file_path : str
            Path to the local file.

        Returns
        -------
        T
            Processed document.
        """
        raise NotImplementedError

    @abstractmethod
    def _process_remote_document(self, file_url: str) -> T:
        """Process a remote document.

        Parameters
        ----------
        file_url : str
            URL to the file.

        Returns
        -------
        T
            Processed document.
        """
        raise NotImplementedError

    def _validate_post_processors(self) -> None:
        """Validate post processors."""
        cumulative_sets = set()
        for post_processor in self.document_post_processors:
            # Check if post_processor.requires was set by any previous post processor
            assert post_processor.requires.issubset(cumulative_sets), (
                f"Post processor {post_processor.__class__.__name__} requires "
                f"{post_processor.requires} but previous post processors only set {cumulative_sets}"
            )
            cumulative_sets.update(post_processor.sets)

    def _process(
        self, file_location: str, process_function: Callable[[str], T]
    ) -> ProcessingResult[T]:
        """Process a document.

        Parameters
        ----------
        file_location : str
            Location of file.

        Returns
        -------
        ProcessingResult[T]
            Processed document.
        """
        processed_document: T
        try:
            processed_document = process_function(file_location)
            processed_document = self.post_process(processed_document)
        except Exception as err:
            return ProcessingResult(
                processing_status=ProcessingStatus(successful=False, error=err)
            )
        return ProcessingResult(
            value=processed_document,
            processing_status=ProcessingStatus(successful=True),
        )


class IdentityProcessor(Processor[T]):
    """Load the json dump of a processed document"""

    def _process_remote_document(self, file_url: str) -> T:
        return self._process(file_url, urllib.request.urlopen)

    def _process_local_document(self, file_path: str) -> T:
        return self._process(file_path, open)

    def _process(self, file_location: str, file_open: Callable) -> T:
        with file_open(file_location) as f:
            data = json.load(f)
            assert hasattr(self.GenericType, "from_json"), (
                f"Cannot initialize {self.GenericType} from json data. "
                "Make sure the class has a `from_json` method."
            )
            return self.GenericType.from_json(data)
