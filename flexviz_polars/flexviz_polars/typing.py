from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeAlias

    import polars as pl
    from polars.datatypes import DataType, DataTypeClass

    IntoExprColumn: TypeAlias = pl.Expr | str | pl.Series
    PolarsDataType: TypeAlias = DataType | DataTypeClass
