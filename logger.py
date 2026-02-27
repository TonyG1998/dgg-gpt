import logging
import os
from dotenv import load_dotenv

load_dotenv()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL_CONSOLE = os.getenv("LOG_LEVEL_CONSOLE", LOG_LEVEL).upper()
LOG_FILE = os.getenv("LOG_FILE", "/data/streams/stream-rag/app.log")


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # capture everything at logger level

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console — less verbose
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, LOG_LEVEL_CONSOLE, logging.INFO))
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File — full debug
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
