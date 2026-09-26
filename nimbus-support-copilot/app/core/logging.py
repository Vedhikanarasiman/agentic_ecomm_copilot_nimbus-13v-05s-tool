import contextvars
import json
import logging
import sys
import time
import uuid

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": round(time.time(), 3),
            "level": record.levelname,
            "trace_id": trace_id_var.get(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        # allow callers to attach extra structured fields via logger.info(msg, extra={"event": {...}})
        if hasattr(record, "event"):
            payload["event"] = record.event
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def new_trace_id() -> str:
    return str(uuid.uuid4())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
