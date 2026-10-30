from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Optional


class TranscriptLogger:
    def __init__(self, path: str, echo_to_terminal: bool = True):
        self.path = path
        self.echo = echo_to_terminal
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")
        self._write_raw(
            f"\n{'=' * 90}\nTRANSCRIPT START  {datetime.now(timezone.utc).isoformat()}\n"
            f"File: {path}\n{'=' * 90}\n"
        )

    def _write_raw(self, text: str):
        self._fh.write(text)
        self._fh.flush()
        if self.echo:
            sys.stdout.write(text)
            sys.stdout.flush()

    def log_message(
        self,
        turn_index: int,
        speaker: str,
        role: str,
        content: str,
        relative_gini: Optional[float] = None,
        raw_gini: Optional[float] = None,
        intervention: bool = False,
    ):
        header = f"\n[Turn {turn_index:03d}] {speaker} ({role})"
        if relative_gini is not None and raw_gini is not None:
            header += f"  |  RelativeGini={relative_gini:+.4f}  RawGini={raw_gini:.4f}"
        if intervention:
            header += "  <<< INTERVENTION TRIGGERED >>>"
        block = f"{header}\n{'-' * min(len(header), 90)}\n{content.strip()}\n"
        self._write_raw(block)

    def log_note(self, text: str):
        self._write_raw(f"\n[NOTE] {text}\n")

    def log_section(self, title: str):
        self._write_raw(f"\n\n{'#' * 90}\n# {title}\n{'#' * 90}\n")

    def close(self):
        self._write_raw(
            f"\n{'=' * 90}\nTRANSCRIPT END  {datetime.now(timezone.utc).isoformat()}\n{'=' * 90}\n"
        )
        self._fh.close()
