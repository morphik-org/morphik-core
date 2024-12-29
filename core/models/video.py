from collections import defaultdict
from numbers import Number
from typing import List, Tuple, Optional, Union, Dict
from bisect import bisect_left
import logging

from pydantic import BaseModel

from core.models.documents import Chunk

logger = logging.getLogger(__name__)


class TimeSeriesData:
    def __init__(self, time_to_content: Dict[float, str]):
        """
        Initialize time series data structure for efficient time-based queries

        Args:
            time_to_content: Dictionary mapping timestamps to content
        """
        # Sort timestamps and content for binary search
        sorted_items = sorted(time_to_content.items(), key=lambda x: x[0])
        self.time_to_content = time_to_content
        self.timestamps = [t for t, _ in sorted_items]
        self.contents = [c for _, c in sorted_items]

        # Create reverse mapping
        self.content_to_times = defaultdict(list)
        for t, c in time_to_content.items():
            self.content_to_times[c].append(t)

        logger.debug(f"Initialized TimeSeriesData with {len(sorted_items)} entries")

    def _find_nearest_index(self, time: float) -> int:
        """Find index of nearest timestamp using binary search"""
        if not self.timestamps:  # Handle empty timestamps list
            return -1

        idx = bisect_left(self.timestamps, time)
        if idx == 0:
            return 0
        if idx == len(self.timestamps):
            return len(self.timestamps) - 1
        before = self.timestamps[idx - 1]
        after = self.timestamps[idx]
        return idx if (time - before) > (after - time) else idx - 1

    def at_time(
        self, time: float, padding: Optional[float] = None
    ) -> Union[str, List[Tuple[float, str]]]:
        """
        Get content at or around specified time

        Args:
            time: Target timestamp
            padding: Optional time padding in seconds to get content before and after

        Returns:
            Either single content string or list of (timestamp, content) pairs if padding specified
        """
        if not self.timestamps:  # Handle empty timestamps list
            return [] if padding is not None else ""

        if padding is None:
            idx = self._find_nearest_index(time)
            return self.contents[idx]

        # Find all content within padding window
        start_time = max(time - padding, self.timestamps[0])  # Clamp to first timestamp
        end_time = min(time + padding, self.timestamps[-1])  # Clamp to last timestamp

        start_idx = self._find_nearest_index(start_time)
        end_idx = self._find_nearest_index(end_time)

        # Ensure valid indices
        start_idx = max(0, start_idx)
        end_idx = min(len(self.timestamps) - 1, end_idx)

        logger.debug(f"Retrieving content between {start_time:.2f}s and {end_time:.2f}s")
        return [(self.timestamps[i], self.contents[i]) for i in range(start_idx, end_idx + 1)]

    def times_for_content(self, content: str) -> List[float]:
        """Get all timestamps where this content appears"""
        return self.content_to_times[content]

    def to_chunks(self) -> List[Chunk]:
        return [
            Chunk(content=content, metadata={"timestamp": timestamp})
            for content, timestamp in zip(self.contents, self.timestamps)
        ]


class ParseVideoResult(BaseModel):
    metadata: Dict[str, Number]
    frame_descriptions: TimeSeriesData
    transcript: TimeSeriesData
