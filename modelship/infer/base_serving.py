import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from modelship.logging import get_logger

logger = get_logger("infer.base_serving")


class OpenAIServing(ABC):
    request_id_prefix: str = ""

    @abstractmethod
    async def warmup(self) -> None:
        """Run a minimal inference pass to warm up the model."""

    @staticmethod
    async def run_in_executor(fn: Callable[..., Any], *args: Any) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, *args)
