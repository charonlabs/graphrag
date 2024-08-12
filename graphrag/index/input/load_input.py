# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing load_input method definition."""

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast, TypeVar

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel

from graphrag.config import InputConfig, InputType
from graphrag.index.config import PipelineInputConfig
from graphrag.index.progress import NullProgressReporter, ProgressReporter
from graphrag.index.storage import (
    BlobPipelineStorage,
    FilePipelineStorage,
)

from .csv import input_type as csv
from .csv import load as load_csv
from .text import input_type as text
from .text import load as load_text
from .supabase import input_type as supabase
from .supabase import load as load_supabase

log = logging.getLogger(__name__)
loaders: dict[str, Callable[..., Awaitable[pd.DataFrame]]] = {
    text: load_text,
    csv: load_csv,
    supabase: load_supabase,
}

Episode = TypeVar("Episode", bound=SQLModel)

async def load_input(
    config: PipelineInputConfig | InputConfig,
    session: AsyncSession | None = None,
    episode: Episode | None = None,
    promptify: Callable[[AsyncSession, Episode, bool], str] | None = None,
    progress_reporter: ProgressReporter | None = None,
    root_dir: str | None = None,
    entity_id: int | None = None,
) -> pd.DataFrame:
    """Load the input data for a pipeline."""
    root_dir = root_dir or ""
    log.info("loading input from root_dir=%s", config.base_dir)
    progress_reporter = progress_reporter or NullProgressReporter()

    if config is None:
        msg = "No input specified!"
        raise ValueError(msg)

    match config.type:
        case InputType.blob:
            log.info("using blob storage input")
            if config.container_name is None:
                msg = "Container name required for blob storage"
                raise ValueError(msg)
            if (
                config.connection_string is None
                and config.storage_account_blob_url is None
            ):
                msg = "Connection string or storage account blob url required for blob storage"
                raise ValueError(msg)
            storage = BlobPipelineStorage(
                connection_string=config.connection_string,
                storage_account_blob_url=config.storage_account_blob_url,
                container_name=config.container_name,
                path_prefix=config.base_dir,
            )
        case InputType.file:
            log.info("using file storage for input")
            storage = FilePipelineStorage(
                root_dir=str(Path(root_dir) / (config.base_dir or ""))
            )
        case InputType.supabase:
            log.info("using supabase storage for input")
            storage = None
        case _:
            log.info("using file storage for input")
            storage = FilePipelineStorage(
                root_dir=str(Path(root_dir) / (config.base_dir or ""))
            )

    if config.file_type in loaders:
        progress = progress_reporter.child(
            f"Loading Input ({config.file_type})", transient=False
        )
        loader = loaders[config.file_type]
        if config.file_type == InputType.supabase:
            if session is None or episode is None or promptify is None:
                msg = "Session, episode, and promptify must be provided for supabase input"
                raise ValueError(msg)
            results = await loader(config, session, entity_id, episode, promptify, progress)
        else:
            results = await loader(config, progress, storage)
        return cast(pd.DataFrame, results)

    msg = f"Unknown input type {config.file_type}"
    raise ValueError(msg)
