"""
UASEF — Logging utility (audit 6.10)

기존 print() 호출과 공존. 새 코드는 이 모듈의 logger를 권장.
환경변수로 제어:

    UASEF_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR     # 기본 INFO
    UASEF_LOG_FILE=/path/to/uasef.log            # 미설정시 stderr만
    UASEF_LOG_JSON=1                             # JSON Lines 출력 (CI/parsing)

사용:
    from utils.logging import get_logger
    log = get_logger(__name__)
    log.info("UQM calibration started: n=%d, alpha=%.3f", n, alpha)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Optional


_configured = False


class _JsonFormatter(logging.Formatter):
    """JSON Lines 형식 — 파싱·집계 용이."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _ensure_configured() -> None:
    global _configured
    if _configured:
        return

    level = os.environ.get("UASEF_LOG_LEVEL", "INFO").upper()
    json_mode = os.environ.get("UASEF_LOG_JSON", "0").lower() in ("1", "true", "yes")
    log_file = os.environ.get("UASEF_LOG_FILE")

    root = logging.getLogger("uasef")
    root.setLevel(getattr(logging, level, logging.INFO))
    root.handlers.clear()

    fmt: logging.Formatter
    if json_mode:
        fmt = _JsonFormatter()
    else:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s — %(message)s",
                                datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    _configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """uasef.<name> 네임스페이스의 logger 반환."""
    _ensure_configured()
    if name is None or name.startswith("uasef"):
        return logging.getLogger(name or "uasef")
    return logging.getLogger(f"uasef.{name}")
