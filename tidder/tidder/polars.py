"""Util functions for interacting with polars."""

import polars as pl

from tidder.typing_mixin import TimeBasedInfo


def time_based_replace(
    info_dicts: list[TimeBasedInfo],
    taget_column: str,
    info_value_column: str,
    default_value: str | None = None,
    df_start_column: str = "start",
    df_end_column: str = "end",
    info_start_column: str = "start_time",
    info_end_column: str = "end_time",
    fill_strategy: str | None = None,
) -> pl.Expr:
    """Map start and end times to chapter titles.

    Challenge here is to conditionally replace (or map) values in a column based on
    a defined list of dictionaries. Built in `replace` method does not support this.

    This function is inspired by: https://stackoverflow.com/a/70974264/7454638

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

    Returns
    -------
        A polars expression that performs conditional replacement into a new
        column.
    """
    # Hack beginning of expression
    expr = pl
    for info in info_dicts:
        expr = expr.when(
            (pl.col(df_start_column) >= info[info_start_column])
            & (pl.col(df_end_column) <= info[info_end_column])
        ).then(pl.lit(info[info_value_column]))

    if default_value is not None:
        expr = expr.otherwise(default_value)

    return expr.fill_null(strategy=fill_strategy).alias(taget_column)
