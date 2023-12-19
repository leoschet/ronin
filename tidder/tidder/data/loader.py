from typing import cast

import attrs
import polars as pl

from tidder.dependencies import webvtt


def to_seconds(time: str) -> float:
    """Convert time string into seconds."""
    seconds = 0.0
    for part in time.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


@attrs.define
class CaptionsReader:
    """Process captions files.

    This class has the primary responsibility of converting .vtt files into a DataFrame.
    At the current state, it is an overkill to have such class. As of now, the main
    benefit is being able to abstract the columns' names avoiding hardcoded strings.
    In the long run, it may pay off if we ever need to support other formats or
    transformations.
    """

    start_column: str = "start"
    end_column: str = "end"
    text_column: str = "text"

    def read_captions(self, captions_file: str) -> pl.DataFrame:
        """Read captions from a .vtt file."""
        captions: webvtt.WebVTT = webvtt.read(captions_file)

        return pl.DataFrame(
            [
                {
                    self.start_column: to_seconds(caption.start),
                    self.end_column: to_seconds(caption.end),
                    self.text_column: caption.text.strip(),
                }
                for caption in cast(webvtt.Caption, captions)
            ]
        )

    def clean_captions(
        self,
        captions_df: pl.DataFrame,
        caption_splitter: str = "\n",
        temporary_group_column: str = "group",
    ) -> pl.DataFrame:
        """Cleans captions by removing duplicates and splitting them into multiple rows.

        Parameters
        ----------
        captions_df : pl.DataFrame
            The captions DataFrame.
        start_column : str, optional, default "start"
            The name of the column containing the start time of the caption.
        end_column : str, optional, default "end"
            The name of the column containing the end time of the caption.
        text_column : str, optional, default "text"
            The name of the column containing the text of the caption.
        caption_splitter : str, optional, default "\n"
            The string used to split the captions into multiple rows.
        temporary_group_column : str, optional, default "group"
            The name of the temporary column used to group the captions.

        Returns
        -------
        pl.DataFrame
            The cleaned captions DataFrame.
        """
        return (
            captions_df.with_columns(
                pl.col(self.text_column).str.split(caption_splitter)
            )
            .explode(self.text_column)
            # Add a temporary column where its value increments whenever the text changes.
            # https://github.com/pola-rs/polars/issues/9328#issue-1750954001
            # .with_columns(
            #     (pl.col(self.text_column) != pl.col(self.text_column).shift())
            #     .backward_fill()
            #     .cumsum()
            #     .alias("group")
            # )
            # https://stackoverflow.com/a/75405310/7454638
            .with_columns(
                pl.col(self.text_column).rle_id().alias(temporary_group_column)
            )
            .group_by(temporary_group_column)
            .agg(
                pl.col(self.start_column).min(),
                pl.col(self.end_column).max(),
                pl.col(self.text_column).first(),
            )
            .sort(temporary_group_column)
            .drop(temporary_group_column)
        )
