import json

import attrs
import polars as pl

from tidder.data.captions import Captions
from tidder.typing_mixin import TimeBasedInfo


@attrs.define
class Video:
    """An YouTube video.

    Attributes
    ----------
    title : str
        The title of the video.
    fulltitle : str
        The full title of the video.
    description : str
        The description of the video.
    duration : int
        The duration of the video in seconds.
    categories : list[str]
        The categories of the video.
    tags : list[str]
        The tags associated with the video.
    chapters : list[dict[str, Union[float, str]]]
        The chapters in the video.
    heatmap : list[dict[str, Union[float, str]]]
        The heatmap data for the video.
    playlist_title : str
        The title of the playlist the video belongs to.
    language : str
        The language of the video.
    captions : pandas.DataFrame
        The captions for the video.
    """

    title: str
    fulltitle: str
    description: str
    duration: int

    categories: list[str]
    tags: list[str]

    chapters: list[TimeBasedInfo]
    heatmap: list[TimeBasedInfo]

    playlist_title: str
    language: str

    captions: pl.DataFrame

    @classmethod
    def from_files(
        cls, video_info_file: str, captions_file: str, captions_kwargs: dict
    ) -> "Video":
        """Creates a Video object from video info and captions files."""

        with open(video_info_file) as f:
            video_info = json.load(f)

        captions = Captions.from_file(captions_file, **captions_kwargs)
        cls.augment_captions(
            captions,
            video_chapters=video_info["chapters"],
            video_heatmap=video_info["heatmap"],
        )

        return cls(
            title=video_info["title"],
            fulltitle=video_info["fulltitle"],
            description=video_info["description"],
            duration=video_info["duration"],
            categories=video_info["categories"],
            tags=video_info["tags"],
            chapters=video_info["chapters"],
            heatmap=video_info["heatmap"],
            playlist_title=video_info["playlist_title"],
            language=video_info["language"],
            captions=captions,
        )

    @staticmethod
    def augment_captions(
        captions: Captions,
        video_chapters: list[TimeBasedInfo],
        video_heatmap: list[TimeBasedInfo],
    ) -> None:
        """Augment captions with extra information."""

        captions.add_time_based_info(
            info=video_chapters,
            info_value_column="title",
            taget_column="chapters",
        )

        captions.add_time_based_info(
            video_heatmap,
            info_value_column="value",
            taget_column="heatmap",
            default_value=0,
        )

    def get_chapters_content(self, concat_symbol: str = " ") -> str:
        """Get chapters content.

        Parameters
        ----------
        concat_symbol : str, optional, default " "
            The symbol to be used to concatenate the chapters.
        """
        return {
            chapter_name: captions.get_content(concat_symbol=concat_symbol)
            for chapter_name, captions in self.captions.groupby(by="chapters")
        }
