"""SupabaseEmitter module."""

import logging
import traceback
import json
from datetime import datetime

import pandas as pd

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import DeclarativeBase, selectinload
from typing import TypeVar
from io import StringIO

from .table_emitter import TableEmitter

logger = logging.getLogger(__name__)

Table = TypeVar("Table", bound=DeclarativeBase)

class SupabaseEmitter(TableEmitter):
    """SupabaseEmitter class."""
    
    def __init__(
        self,
        table_model: Table # type: ignore
    ):
        """Create a new Supabase Emitter."""
        self.table_model = table_model

    async def emit(self, name: str, graph_index: DeclarativeBase, data: pd.DataFrame, session: AsyncSession) -> None:
        """Emit data to the Supabase database."""
        table = self.table_model(
            name=name,
            data=json.loads(data.to_json()),
            created_at=datetime.now(),
        ) # type: ignore
        logger.info(f"Emiting {name} to Supabase")
        try:
            (await graph_index.awaitable_attrs.rows).append(table) # type: ignore
            logger.info(f"Emitted {name} to Supabase")
        except Exception as e:
            logger.error(f"Error emitting {name} to Supabase: {e}")
            traceback.print_exc()
            
    async def load_table(self, name: str, graph_index: DeclarativeBase, session: AsyncSession) -> pd.DataFrame:
        """Load table from Supabase."""
        rows = await graph_index.awaitable_attrs.rows # type: ignore
        for row in rows:
            if row.name == name:
                return pd.DataFrame(row.data)
        raise ValueError(f"No data found for name '{name}'")