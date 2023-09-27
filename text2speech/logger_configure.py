import logging
from pathlib import Path


def setup_root():
    LOGS_DIR = Path("logs")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    info_handler = logging.handlers.RotatingFileHandler(
        filename=Path(LOGS_DIR, "info.log"),
        maxBytes=10485760,  # 1 MB
        backupCount=10,
    )
    info_handler.setLevel(logging.INFO)
    error_handler = logging.handlers.RotatingFileHandler(
        filename=Path(LOGS_DIR, "error.log"),
        maxBytes=10485760,  # 1 MB
        backupCount=10,
    )
    error_handler.setLevel(logging.ERROR)

    # Create formatters
    minimal_formatter = logging.Formatter(fmt="%(message)s")
    detailed_formatter = logging.Formatter(fmt="%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n")

    # Hook it all up
    console_handler.setFormatter(fmt=minimal_formatter)
    info_handler.setFormatter(fmt=detailed_formatter)
    error_handler.setFormatter(fmt=detailed_formatter)
    logger.addHandler(hdlr=console_handler)
    logger.addHandler(hdlr=info_handler)
    logger.addHandler(hdlr=error_handler)


def get_logger(name: str):
    """get the logger given the name, first set up root logger if not been configured."""
    if len(logging.root.handlers) > 0:
        return logging.getLogger(name)
    else:
        setup_root()
        return logging.getLogger(name)
