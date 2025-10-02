# logging_utils.py
import json, os, sys, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Any, Dict

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
            "module": record.module,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

def setup_logging(level: str = "INFO", log_dir: str = "./logs"):
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level.upper())
    ch.setFormatter(JsonFormatter())
    root.addHandler(ch)

    # File (rotating)
    fh = RotatingFileHandler(os.path.join(log_dir, "app.log"), maxBytes=10_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(level.upper())
    fh.setFormatter(JsonFormatter())
    root.addHandler(fh)

    # Silence noisy libs a bit
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
