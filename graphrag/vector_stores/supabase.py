"""The Supabase vector store implementation package."""

from graphrag.model.types import TextEmbedder

from typing import Any, TypeVar
import json

from sklearn.metrics.pairwise import cosine_distances

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, selectinload
from sqlalchemy import select, delete

from .base import (
    BaseVectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)

import logging
import traceback

logger = logging.getLogger(__name__)

VectorTable = TypeVar("VectorTable", bound=DeclarativeBase)

class SupabaseVectorStore(BaseVectorStore):
    """The Supabase vector store implementation."""
    
    def connect(self, **kwargs: Any) -> Any:
        """Connect to the vector store."""
        pass
    
    async def load_documents(
        self, documents: list[VectorStoreDocument], session: AsyncSession, graph_index: DeclarativeBase, vector_table_model: VectorTable # type: ignore
    ) -> None:
        """Load documents into vector storage."""
        data = [vector_table_model(
            id=document.id,
            text=document.text,
            vector=document.vector,
            attributes=json.dumps(document.attributes),
            index=graph_index  # Add this line to set the relationship
        ) for document in documents if document.vector is not None] # type: ignore

        if data:
            try:
                for vector in (await graph_index.awaitable_attrs.vectors):
                     await session.delete(vector)
            except Exception as e:
                logger.error(f"Error adding vectors to session: {e}")
                traceback.print_exc()
            
    def filter_by_id(self, include_ids: list[str] | list[int]) -> Any:
            """Build a query filter to filter documents by id."""
            if len(include_ids) == 0:
                self.query_filter = None
            else:
                self.query_filter = include_ids
            return self.query_filter

    async def similarity_search_by_vector(
            self, query_embedding: list[float], session: AsyncSession, graph_index: DeclarativeBase, vector_table_model: VectorTable, k: int = 10, **kwargs: Any # type: ignore
        ) -> list[VectorStoreSearchResult]:
            """Perform a vector-based similarity search."""
            if self.query_filter:
                docs = (
                    await session.scalars(select(vector_table_model).options(selectinload(vector_table_model.index)).where(vector_table_model.index == graph_index, vector_table_model.id in self.query_filter).order_by(vector_table_model.vector.cosine_distance(query_embedding)).limit(k)) # type: ignore
                )
            else:
                docs = (
                    await session.scalars(select(vector_table_model).options(selectinload(vector_table_model.index)).where(vector_table_model.index == graph_index).order_by(vector_table_model.vector.cosine_distance(query_embedding)).limit(k)) # type: ignore
                )
            return [
                VectorStoreSearchResult(
                    document=VectorStoreDocument(
                        id=doc.id,
                        text=doc.text,
                        vector=doc.vector,
                        attributes=json.loads(doc.attributes),
                    ),
                    score=1 - abs(float(cosine_distances([query_embedding], [doc.vector]))), # type: ignore
                )
                for doc in docs
            ]

    async def similarity_search_by_text(
        self, text: str, text_embedder: TextEmbedder, session: AsyncSession, graph_index: DeclarativeBase, vector_table_model: VectorTable, k: int = 10, **kwargs: Any # type: ignore
    ) -> list[VectorStoreSearchResult]:
            """Perform a similarity search using a given input text."""
            query_embedding = text_embedder(text)
            if query_embedding:
                return await self.similarity_search_by_vector(query_embedding, session, graph_index, vector_table_model, k)
            return []