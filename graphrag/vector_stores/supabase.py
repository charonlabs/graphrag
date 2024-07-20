"""The Supabase vector store implementation package."""

from graphrag.model.types import TextEmbedder

from typing import Any, TypeVar
import json

from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import SQLModel, select

from .base import (
    BaseVectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)

Table = TypeVar("Table", bound=SQLModel)

class SupabaseVectorStore(BaseVectorStore):
    """The Supabase vector store implementation."""
    
    def connect(self) -> Any:
        """Connect to the vector store."""
        pass
    
    async def load_documents(
        self, documents: list[VectorStoreDocument], session: AsyncSession, entity_id: int, table_model: Table # type: ignore
    ) -> None:
        """Load documents into vector storage."""
        data = [table_model(
            id=document.id,
            text=document.text,
            vector=document.vector,
            attributes=json.dumps(document.attributes),
            entity_id=entity_id
        ) for document in documents if document.vector is not None] # type: ignore

        if len(data) == 0:
            data = None
            
        if data:
            session.add_all(data) # type: ignore
            await session.commit()
            
    def filter_by_id(self, include_ids: list[str] | list[int]) -> Any:
            """Build a query filter to filter documents by id."""
            if len(include_ids) == 0:
                self.query_filter = None
            else:
                self.query_filter = include_ids
            return self.query_filter

    async def similarity_search_by_vector(
            self, query_embedding: list[float], session: AsyncSession, entity_id: int, table_model: Table, k: int = 10, **kwargs: Any # type: ignore
        ) -> list[VectorStoreSearchResult]:
            """Perform a vector-based similarity search."""
            if self.query_filter:
                docs = (
                    await session.exec(select(table_model).where(table_model.entity_id == entity_id, table_model.id in self.query_filter).order_by(table_model.vector.cosine_distance(query_embedding)).limit(k)) # type: ignore
                )
            else:
                docs = (
                    await session.exec(select(table_model).where(table_model.entity_id == entity_id).order_by(table_model.vector.cosine_distance(query_embedding)).limit(k)) # type: ignore
                )
            return [
                VectorStoreSearchResult(
                    document=VectorStoreDocument(
                        id=doc["id"],
                        text=doc["text"],
                        vector=doc["vector"],
                        attributes=json.loads(doc["attributes"]),
                    ),
                    score=1 - abs(float(doc["_distance"])),
                )
                for doc in docs
            ]

    async def similarity_search_by_text(
        self, text: str, text_embedder: TextEmbedder, session: AsyncSession, entity_id: int, table_model: Table, k: int = 10, **kwargs: Any # type: ignore
    ) -> list[VectorStoreSearchResult]:
            """Perform a similarity search using a given input text."""
            query_embedding = text_embedder(text)
            if query_embedding:
                return await self.similarity_search_by_vector(query_embedding, session, entity_id, table_model, k)
            return []