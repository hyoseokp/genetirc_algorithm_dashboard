from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VALID_NOT_RUN = 0
VALID_STAGED = 2
VALID_COMMITTED = 1


@dataclass(frozen=True)
class CommitMarker:
    """2-phase commit marker for FDTD runs.

    We use a small text file containing one integer:
      0: not run
      2: written but not committed (staged)
      1: committed (complete)
    """

    path: Path

    def read(self) -> int:
        if not self.path.exists():
            return VALID_NOT_RUN
        s = self.path.read_text(encoding="utf-8").strip()
        try:
            v = int(s)
        except ValueError:
            return VALID_NOT_RUN
        if v not in (VALID_NOT_RUN, VALID_STAGED, VALID_COMMITTED):
            return VALID_NOT_RUN
        return v

    def write_atomic(self, value: int) -> None:
        value = int(value)
        if value not in (VALID_NOT_RUN, VALID_STAGED, VALID_COMMITTED):
            raise ValueError(f"invalid commit marker value: {value}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(str(value), encoding="utf-8")
        tmp.replace(self.path)

    def set_not_run(self) -> None:
        self.write_atomic(VALID_NOT_RUN)

    def stage(self) -> None:
        self.write_atomic(VALID_STAGED)

    def commit(self) -> None:
        self.write_atomic(VALID_COMMITTED)

