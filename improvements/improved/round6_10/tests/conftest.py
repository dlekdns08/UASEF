"""Shared pytest fixtures and path setup (audit 6.10)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# 단위 테스트는 fallback 데이터 허용 (HF 다운로드 없이 동작)
os.environ.setdefault("UASEF_ALLOW_FALLBACK", "1")
