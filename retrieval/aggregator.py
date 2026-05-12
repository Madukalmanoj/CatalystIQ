"""Parallel retrieval coordinator with deduplication and dataframe output."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed

import pandas as pd
from loguru import logger

from retrieval.base import BaseRetriever, CatalystRecord


class CatalystAggregator:
    """Run multiple retrievers in parallel and aggregate normalized output."""

    def __init__(self, retrievers: list[BaseRetriever]) -> None:
        """Initialize aggregator with retriever instances.

        Args:
            retrievers: List of source retriever objects.

        Returns:
            None.

        Raises:
            None.
        """
        self.retrievers = retrievers

    def retrieve(self, reaction: str, timeout: int = 60) -> pd.DataFrame:
        """Execute parallel source retrieval, deduplicate, and sort records.

        Each retriever runs in its own thread. If a retriever exceeds the
        timeout it is skipped gracefully — it never blocks the others.

        Args:
            reaction: User reaction string.
            timeout: Per-batch wait time in seconds (default raised to 60 for
                     slow SOAP sources like BRENDA).

        Returns:
            DataFrame of deduplicated candidate records sorted by activity.

        Raises:
            None.
        """
        all_records: list[CatalystRecord] = []

        with ThreadPoolExecutor(max_workers=max(1, len(self.retrievers))) as executor:
            future_map: dict[Future[list[CatalystRecord]], str] = {
                executor.submit(retriever.search, reaction): retriever.source_name
                for retriever in self.retrievers
            }

            try:
                completed = as_completed(future_map, timeout=timeout)
                for future in completed:
                    source = future_map[future]
                    try:
                        records = future.result()
                        all_records.extend(records)
                        logger.info("Retriever '{}' completed with {} records.", source, len(records))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Retriever '{}' failed and was skipped: {}", source, exc)

            except FuturesTimeoutError:
                # Some futures didn't finish in time — collect whatever completed
                for future, source in future_map.items():
                    if future.done():
                        try:
                            records = future.result()
                            all_records.extend(records)
                            logger.info("Retriever '{}' completed (after timeout sweep) with {} records.", source, len(records))
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Retriever '{}' failed: {}", source, exc)
                    else:
                        logger.warning("Retriever '{}' timed out and was skipped.", source)
                        future.cancel()

        if not all_records:
            return pd.DataFrame(
                columns=[
                    "source",
                    "source_id",
                    "name",
                    "formula",
                    "reaction",
                    "activity_metric",
                    "activity_value",
                    "activity_unit",
                    "conditions",
                    "stability",
                    "raw",
                ]
            )

        deduped: dict[tuple[str, str, str], CatalystRecord] = {}
        for record in all_records:
            key = (
                record.formula.strip().lower(),
                record.reaction.strip().lower(),
                record.source.strip().lower(),
            )
            deduped[key] = record

        frame = pd.DataFrame([record.model_dump() for record in deduped.values()])
        if "activity_value" in frame.columns:
            frame["activity_value"] = pd.to_numeric(frame["activity_value"], errors="coerce")
            frame = frame.sort_values(by="activity_value", ascending=True, na_position="last")
        frame = frame.reset_index(drop=True)
        return frame
