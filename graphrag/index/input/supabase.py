"""A module containing load method definition."""

import logging
from typing import Any, TypeVar, Callable, Any, Awaitable

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, selectinload
from sqlalchemy import select

from graphrag.index.config import PipelineInputConfig
from graphrag.index.progress import ProgressReporter
from graphrag.index.storage import PipelineStorage
from graphrag.index.utils import gen_md5_hash

input_type = "supabase"
logger = logging.getLogger(__name__)

Episode = TypeVar("Episode", bound=DeclarativeBase)

async def load(
    config: PipelineInputConfig,
    session: AsyncSession,
    entity_id: int,
    episode: Episode,
    promptify: Callable[[AsyncSession, Episode], Awaitable[str]],
    progress: ProgressReporter | None,
) -> pd.DataFrame:
    """Load the input data for a pipeline."""
    episodes = (await session.scalars(select(episode).options(selectinload(episode.entity)).where(episode.entity.has(id=entity_id)))).all() # type: ignore
    formatted_episodes = []
    for episode in episodes:
        text = await promptify(session, episode, use_xml=False) # type: ignore
        logger.debug(f"Loaded episode {episode.id} with text {text}") # type: ignore
        formatted_episode = {"text": text} # type: ignore
        formatted_episode["id"] = episode.id # type: ignore
        formatted_episode["title"] = f"Episode {episode.id}" # type: ignore
        formatted_episodes.append(formatted_episode)
    return pd.DataFrame(formatted_episodes)