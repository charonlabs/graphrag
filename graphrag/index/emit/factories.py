# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Table Emitter Factories."""

from graphrag.index.storage import PipelineStorage
from graphrag.index.typing import ErrorHandlerFn

from .csv_table_emitter import CSVTableEmitter
from .json_table_emitter import JsonTableEmitter
from .parquet_table_emitter import ParquetTableEmitter
from .supabase_emitter import SupabaseEmitter
from .table_emitter import TableEmitter
from .types import TableEmitterType

from sqlalchemy.orm import DeclarativeBase
from typing import TypeVar

Table = TypeVar("Table", bound=DeclarativeBase)

def create_table_emitter(
    emitter_type: TableEmitterType, storage: PipelineStorage, on_error: ErrorHandlerFn, table_model: Table | None = None
) -> TableEmitter:
    """Create a table emitter based on the specified type."""
    match emitter_type:
        case TableEmitterType.Json:
            return JsonTableEmitter(storage)
        case TableEmitterType.Parquet:
            return ParquetTableEmitter(storage, on_error)
        case TableEmitterType.CSV:
            return CSVTableEmitter(storage)
        case TableEmitterType.Supabase:
            if table_model is None:
                raise ValueError("Table model is required for Supabase emitter")
            return SupabaseEmitter(table_model)
        case _:
            msg = f"Unsupported table emitter type: {emitter_type}"
            raise ValueError(msg)


def create_table_emitters(
    emitter_types: list[TableEmitterType],
    storage: PipelineStorage,
    on_error: ErrorHandlerFn,
    table_model: Table | None = None,
) -> list[TableEmitter]:
    """Create a list of table emitters based on the specified types."""
    emitters = []
    for emitter_type in emitter_types:
        if emitter_type == TableEmitterType.Supabase:
            if table_model is None:
                raise ValueError("Table model is required for Supabase emitter")
            emitters.append(SupabaseEmitter(table_model))
        else:
            emitters.append(create_table_emitter(emitter_type, storage, on_error))
    return emitters
