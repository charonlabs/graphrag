# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LocalSearch implementation."""

import logging
import time
from typing import Any, TypeVar

import tiktoken

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

from graphrag.query.context_builder.builders import LocalContextBuilder
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
from graphrag.query.llm.base import BaseLLM, BaseLLMCallback
from graphrag.query.llm.text_utils import num_tokens
from graphrag.query.structured_search.base import BaseSearch, SearchResult
from graphrag.query.structured_search.local_search.system_prompt import (
    LOCAL_SEARCH_SYSTEM_PROMPT,
)
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.vector_stores.supabase import SupabaseVectorStore

DEFAULT_LLM_PARAMS = {
    "max_tokens": 1500,
    "temperature": 0.0,
}

log = logging.getLogger(__name__)


VectorTable = TypeVar("VectorTable", bound=DeclarativeBase)


class LocalSearch(BaseSearch):
    """Search orchestration for local search mode."""

    def __init__(
        self,
        llm: BaseLLM,
        context_builder: LocalContextBuilder,
        token_encoder: tiktoken.Encoding | None = None,
        system_prompt: str = LOCAL_SEARCH_SYSTEM_PROMPT,
        response_type: str = "multiple paragraphs",
        callbacks: list[BaseLLMCallback] | None = None,
        llm_params: dict[str, Any] = DEFAULT_LLM_PARAMS,
        context_builder_params: dict | None = None,
    ):
        super().__init__(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            llm_params=llm_params,
            context_builder_params=context_builder_params or {},
        )
        self.system_prompt = system_prompt
        self.callbacks = callbacks
        self.response_type = response_type

    async def asearch(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        session: AsyncSession | None = None,
        index_id: int | None = None,
        vector_table_model: VectorTable | None = None, # type: ignore
        **kwargs,
    ) -> SearchResult:
        """Build local search context that fits a single context window and generate answer for the user query."""
        start_time = time.time()
        search_prompt = ""

        if isinstance(self.context_builder, LocalSearchMixedContext):
            if isinstance(self.context_builder.entity_text_embeddings, SupabaseVectorStore):
                assert session is not None, "Session is required for SupabaseVectorStore search."
                assert index_id is not None, "Index ID is required for SupabaseVectorStore search."
                assert vector_table_model is not None, "Vector table model is required for SupabaseVectorStore search."
                context_text, context_records = await self.context_builder.build_context(
                    query=query,
                    conversation_history=conversation_history,
                    session=session,
                    index_id=index_id,
                    vector_table_model=vector_table_model,
                    **kwargs,
                    **self.context_builder_params,
                )
            else:
                context_text, context_records = await self.context_builder.build_context(
                    query=query,
                    conversation_history=conversation_history,
                    **kwargs,
                    **self.context_builder_params,
                )
        else:
            context_text, context_records = self.context_builder.build_context(
                query=query,
                conversation_history=conversation_history,
                **kwargs,
                **self.context_builder_params,
        )
        log.info("GENERATE ANSWER: %s. QUERY: %s", start_time, query)
        try:
            search_prompt = self.system_prompt.format(
                context_data=context_text, response_type=self.response_type
            )
            search_messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": query},
            ]

            response = await self.llm.agenerate(
                messages=search_messages,
                streaming=True,
                callbacks=self.callbacks,
                **self.llm_params,
            )

            return SearchResult(
                response=response,
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )

        except Exception:
            log.exception("Exception in _asearch")
            return SearchResult(
                response="",
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )

    def search(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> SearchResult:
        """Build local search context that fits a single context window and generate answer for the user question."""
        start_time = time.time()
        search_prompt = ""
        context_text, context_records = self.context_builder.build_context(
            query=query,
            conversation_history=conversation_history,
            **kwargs,
            **self.context_builder_params,
        )
        log.info("GENERATE ANSWER: %d. QUERY: %s", start_time, query)
        try:
            search_prompt = self.system_prompt.format(
                context_data=context_text, response_type=self.response_type
            )
            search_messages = [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": query},
            ]

            response = self.llm.generate(
                messages=search_messages,
                streaming=True,
                callbacks=self.callbacks,
                **self.llm_params,
            )

            return SearchResult(
                response=response,
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )

        except Exception:
            log.exception("Exception in _map_response_single_batch")
            return SearchResult(
                response="",
                context_data=context_records,
                context_text=context_text,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(search_prompt, self.token_encoder),
            )
