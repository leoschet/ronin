import attrs
import polars as pl

import tidder.polars as tpl
from tidder.data.loader import CaptionsReader
from tidder.typing_mixin import TimeBasedInfo


@attrs.define
class Captions:
    """Captions for a YouTube video.

    Attributes
    ----------
    auto : pandas.DataFrame
        The automatically generated captions.
    en_us : pandas.DataFrame
        The manually generated captions.
    """

    df: pl.DataFrame

    start_column: str
    end_column: str
    text_column: str

    @classmethod
    def from_file(
        cls, captions_file: str, reader: CaptionsReader = CaptionsReader()
    ) -> "Captions":
        """Creates a Captions object from captions files."""
        captions_df = reader.read_captions(captions_file)
        captions_df = reader.clean_captions(captions_df)

        return cls(
            df=captions_df,
            start_column=reader.start_column,
            end_column=reader.end_column,
            text_column=reader.text_column,
        )

    def get_content(self, concat_symbol: str = " ") -> str:
        """Get captions content.

        Parameters
        ----------
        concat_symbol : str, optional, default " "
            The symbol to be used to concatenate the captions.
        """
        return concat_symbol.join(
            self.df.sort(by=self.start_column)[self.text_column].to_list()
        )

    def groupby(self, by: str) -> Generator[tuple[str, "Captions"], None, None]:
        """Groups captions by a column."""
        for group_id, group_df in self.df.groupby(by, maintain_order=True):
            yield (
                group_id,
                Captions(
                    df=group_df,
                    start_column=self.start_column,
                    end_column=self.end_column,
                    text_column=self.text_column,
                ),
            )

    def add_time_based_info(
        self,
        info_dicts: list[TimeBasedInfo],
        taget_column: str,
        info_value_column: str,
        default_value: str | None = None,
        info_start_column: str = "start_time",
        info_end_column: str = "end_time",
        override: bool = False,
    ) -> "Captions":
        """Adds time based information to captions.

        Parameters
        ----------
        info_dicts : list[TimeBasedInfo]
            A list of dictionaries containing time based information.
        taget_column : str
            The name of the column to be created.
        info_value_column : str
            The name of the column containing the information value.
        default_value : str, optional, default None
            The default value to be used if no information is found
            for a time span.
        info_start_column : str, optional, default "start_time"
            The name of the column containing the start time of the information.
        info_end_column : str, optional, default "end_time"
            The name of the column containing the end time of the information.
        override : bool, optional, default False
            Whether to override the current target column.

        Returns
        -------
        Captions
            The captions with chapters information.
        """
        if taget_column in self.df.columns and not override:
            raise ValueError(f"Column '{taget_column}' already exists.")

        self.df = self.df.with_columns(
            tpl.time_based_replace(
                info_dicts=info_dicts,
                taget_column=taget_column,
                info_value_column=info_value_column,
                default_value=default_value,
                df_start_column=self.start_column,
                df_end_column=self.end_column,
                info_start_column=info_start_column,
                info_end_column=info_end_column,
            )
        )

        return self
