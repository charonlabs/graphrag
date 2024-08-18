# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Command line interface for the query module."""

import os
from pathlib import Path
from typing import cast, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase
import pandas as pd

from graphrag.config import (
    GraphRagConfig,
    create_graphrag_config,
)
from graphrag.index.progress import PrintProgressReporter
from graphrag.model.entity import Entity
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.vector_stores import VectorStoreFactory, VectorStoreType
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.vector_stores.supabase import SupabaseVectorStore

from .factories import get_global_search_engine, get_local_search_engine
from .indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from ..index.emit.supabase_emitter import SupabaseEmitter

reporter = PrintProgressReporter("")

Table = TypeVar("Table", bound=DeclarativeBase)
VectorTable = TypeVar("VectorTable", bound=DeclarativeBase)

async def __get_embedding_description_store(
    entities: list[Entity],
    vector_store_type: str = VectorStoreType.LanceDB,
    config_args: dict | None = None,
    session: AsyncSession | None = None,
    index_id: int | None = None,
    vector_table_model: VectorTable | None = None # type: ignore
):
    """Get the embedding description store."""
    if not config_args:
        config_args = {}

    collection_name = config_args.get(
        "query_collection_name", "entity_description_embeddings"
    )
    config_args.update({"collection_name": collection_name})
    description_embedding_store = VectorStoreFactory.get_vector_store(
        vector_store_type=vector_store_type, kwargs=config_args
    )

    description_embedding_store.connect(**config_args)

    if config_args.get("overwrite", True):
        # this step assumps the embeddings where originally stored in a file rather
        # than a vector database

        # dump embeddings from the entities list to the description_embedding_store
        await store_entity_semantic_embeddings(
            entities=entities, vectorstore=description_embedding_store, session=session, index_id=index_id, vector_table_model=vector_table_model # type: ignore
        )
    else:
        # load description embeddings to an in-memory lancedb vectorstore
        # to connect to a remote db, specify url and port values.
        description_embedding_store = LanceDBVectorStore(
            collection_name=collection_name
        )
        description_embedding_store.connect(
            db_uri=config_args.get("db_uri", "./lancedb")
        )

        # load data from an existing table
        description_embedding_store.document_collection = (
            description_embedding_store.db_connection.open_table(
                description_embedding_store.collection_name
            )
        )

    return description_embedding_store


async def run_global_search(
    data_dir: str | None,
    root_dir: str,
    community_level: int,
    response_type: str,
    query: str,
    use_db: bool = False,
    session: AsyncSession | None = None,
    index_id: int | None = None,
    table_model: Table | None = None # type: ignore
):
    """Run a global search with the given query."""
    data_dir, root_dir, config = _configure_paths_and_settings(data_dir, root_dir, use_db)
    
    if use_db:
        assert session is not None, "Session is required when using the database."
        assert index_id is not None, "Index ID is required when using the database."
        assert table_model is not None, "Table model is required when using the database."
    
    emitter = SupabaseEmitter(table_model=table_model) # type: ignore
    
    if not use_db:
        assert data_dir is not None
        data_path = Path(data_dir)

        final_nodes: pd.DataFrame = pd.read_parquet(
            data_path / "create_final_nodes.parquet"
        )
        final_entities: pd.DataFrame = pd.read_parquet(
            data_path / "create_final_entities.parquet"
        )
        final_community_reports: pd.DataFrame = pd.read_parquet(
            data_path / "create_final_community_reports.parquet"
        )
    else:
        final_nodes = await emitter.load_table(name="create_final_nodes", index_id=index_id, session=session) # type: ignore
        final_entities = await emitter.load_table(name="create_final_entities", index_id=index_id, session=session) # type: ignore
        final_community_reports = await emitter.load_table(name="create_final_community_reports", index_id=index_id, session=session) # type: ignore

    reports = read_indexer_reports(
        final_community_reports, final_nodes, community_level
    )
    entities = read_indexer_entities(final_nodes, final_entities, community_level)
    search_engine = get_global_search_engine(
        config,
        reports=reports,
        entities=entities,
        response_type=response_type,
    )

    result = await search_engine.asearch(query=query)

    reporter.success(f"Global Search Response: {result.response}")
    return result.response


async def run_local_search(
    data_dir: str | None,
    root_dir: str,
    community_level: int,
    response_type: str,
    query: str,
    use_db: bool = False,
    session: AsyncSession | None = None,
    index_id: int | None = None,
    table_model: Table | None = None, # type: ignore
    vector_table_model: VectorTable | None = None # type: ignore
):
    """Run a local search with the given query."""
    data_dir, root_dir, config = _configure_paths_and_settings(data_dir, root_dir, use_db)
    
    if use_db:
        assert session is not None, "Session is required when using the database."
        assert index_id is not None, "Index ID is required when using the database."
        assert table_model is not None, "Table model is required when using the database."
    
    emitter = SupabaseEmitter(table_model=table_model) # type: ignore
    
    if not use_db:
        assert data_dir is not None
        data_path = Path(data_dir)

        final_nodes = pd.read_parquet(data_path / "create_final_nodes.parquet")
        final_community_reports = pd.read_parquet(
            data_path / "create_final_community_reports.parquet"
        )
        final_text_units = pd.read_parquet(data_path / "create_final_text_units.parquet")
        final_relationships = pd.read_parquet(
            data_path / "create_final_relationships.parquet"
        )
        final_entities = pd.read_parquet(data_path / "create_final_entities.parquet")
        final_covariates_path = data_path / "create_final_covariates.parquet"
        final_covariates = (
            pd.read_parquet(final_covariates_path)
            if final_covariates_path.exists()
            else None
        )
    else:
        final_nodes = await emitter.load_table(name="create_final_nodes", index_id=index_id, session=session) # type: ignore
        final_community_reports = await emitter.load_table(name="create_final_community_reports", index_id=index_id, session=session) # type: ignore
        final_text_units = await emitter.load_table(name="create_final_text_units", index_id=index_id, session=session) # type: ignore
        final_relationships = await emitter.load_table(name="create_final_relationships", index_id=index_id, session=session) # type: ignore
        final_entities = await emitter.load_table(name="create_final_entities", index_id=index_id, session=session) # type: ignore
        try:
            final_covariates = await emitter.load_table(name="create_final_covariates", index_id=index_id, session=session) # type: ignore
        except:
            final_covariates = None

    vector_store_args = (
        config.embeddings.vector_store if config.embeddings.vector_store else {}
    )

    reporter.info(f"Vector Store Args: {vector_store_args}")
    vector_store_type = vector_store_args.get("type", VectorStoreType.Supabase)

    entities = read_indexer_entities(final_nodes, final_entities, community_level)
    description_embedding_store = await __get_embedding_description_store(
        entities=entities,
        vector_store_type=vector_store_type,
        config_args=vector_store_args,
        session=session,
        index_id=index_id,
        vector_table_model=vector_table_model # type: ignore
    )
    entities = read_indexer_entities(final_nodes, final_entities, community_level)
    if isinstance(description_embedding_store, SupabaseVectorStore):
        await store_entity_semantic_embeddings(
            entities=entities, vectorstore=description_embedding_store, session=session, index_id=index_id, vector_table_model=vector_table_model # type: ignore
        )
    else:
        await store_entity_semantic_embeddings(
            entities=entities, vectorstore=description_embedding_store
    )
    covariates = (
        read_indexer_covariates(final_covariates)
        if final_covariates is not None
        else []
    )

    search_engine = get_local_search_engine(
        config,
        reports=read_indexer_reports(
            final_community_reports, final_nodes, community_level
        ),
        text_units=read_indexer_text_units(final_text_units),
        entities=entities,
        relationships=read_indexer_relationships(final_relationships),
        covariates={"claims": covariates},
        description_embedding_store=description_embedding_store,
        response_type=response_type,
    )

    if isinstance(description_embedding_store, SupabaseVectorStore):
        result = await search_engine.asearch(query=query, session=session, index_id=index_id, vector_table_model=vector_table_model) # type: ignore
    else:
        result = await search_engine.asearch(query=query)
    reporter.success(f"Local Search Response: {result.response}")
    return result.response


def _configure_paths_and_settings(
    data_dir: str | None, root_dir: str, use_db: bool = False
) -> tuple[str | None, str, GraphRagConfig]:
    if data_dir is None and root_dir is None and not use_db:
        msg = "Either data_dir or root_dir must be provided."
        raise ValueError(msg)
    if data_dir is None and not use_db:
        data_dir = _infer_data_dir(cast(str, root_dir))
    config = _create_graphrag_config(root_dir, data_dir)
    if use_db:
        data_dir = None
    return data_dir, root_dir, config


def _infer_data_dir(root: str) -> str:
    output = Path(root) / "output"
    # use the latest data-run folder
    if output.exists():
        folders = sorted(output.iterdir(), key=os.path.getmtime, reverse=True)
        if len(folders) > 0:
            folder = folders[0]
            return str((folder / "artifacts").absolute())
    msg = f"Could not infer data directory from root={root}"
    raise ValueError(msg)


def _create_graphrag_config(
    root: str | None,
    config_dir: str | None,
) -> GraphRagConfig:
    """Create a GraphRag configuration."""
    return _read_config_parameters(root or "./", config_dir)


def _read_config_parameters(root: str, config: str | None):
    _root = Path(root)
    settings_yaml = (
        Path(config)
        if config and Path(config).suffix in [".yaml", ".yml"]
        else _root / "settings.yaml"
    )
    if not settings_yaml.exists():
        settings_yaml = _root / "settings.yml"

    if settings_yaml.exists():
        reporter.info(f"Reading settings from {settings_yaml}")
        with settings_yaml.open(
            "rb",
        ) as file:
            import yaml

            data = yaml.safe_load(file.read().decode(encoding="utf-8", errors="strict"))
            return create_graphrag_config(data, root)

    settings_json = (
        Path(config)
        if config and Path(config).suffix == ".json"
        else _root / "settings.json"
    )
    if settings_json.exists():
        reporter.info(f"Reading settings from {settings_json}")
        with settings_json.open("rb") as file:
            import json

            data = json.loads(file.read().decode(encoding="utf-8", errors="strict"))
            return create_graphrag_config(data, root)

    reporter.info("Reading settings from environment variables")
    return create_graphrag_config(root_dir=root)
