"""
title: Morphik Pipeline
author: Arnav Agrawal
date: 2025-04-10
version: 1.0
license: MIT
description: A pipeline for retrieving multimodal information from the Morphik knowledge base.
requirements: morphik
"""

from typing import Generator, Iterator, List, Union
# from pydantic import BaseModel, Field
import os

class Pipeline:
    # class Valves(BaseModel):
    #     # OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    #     RATE_LIMIT: int = Field(default=5, description="Rate limit for the pipeline")
    #     WORD_LIMIT: int = Field(default=300, description="Word limit when getting page summary")
    #     WIKIPEDIA_ROOT: str = Field(default="https://en.wikipedia.org/wiki", description="Wikipedia root URL")

    def __init__(self):
        self.db = None
    
    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        pass

    async def on_startup(self):
        from morphik import AsyncMorphik
        uri = os.getenv("MORPHIK_URI")
        local = os.getenv("MORPHIK_LOCAL", True)
        timeout = os.getenv("MORPHIK_TIMEOUT", 1000)
        self.db = AsyncMorphik(uri=uri, timeout=timeout, is_local=local)

    async def on_shutdown(self):
        self.db = None

    async def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        print(messages)
        print(user_message)

        results = await self.db.query(query=user_message)
        return results.completion
