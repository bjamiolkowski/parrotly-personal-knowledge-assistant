from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

LOG_FILE = LOGS_DIR / "queries.log"


def log_query(data: Dict[str, Any]) -> None:
    """
    Append query log as JSON line.
    """
    data["timestamp"] = time.time()

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")