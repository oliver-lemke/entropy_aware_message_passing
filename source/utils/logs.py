import logging
import os.path

import wandb
from utils.config import Config

config = Config()

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(90, 98)

COLOR_SEQ = "\033[0;{}m"
RESET_SEQ = "\033[0m"

COLORS = {
    "DEBUG": BLUE,
    "INFO": WHITE,
    "WARNING": YELLOW,
    "ERROR": MAGENTA,
    "CRITICAL": RED,
}

LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class _WandbHandler(logging.Handler):
    def emit(self, record):
        if not config["wandb"]["enable"]:
            return
        try:
            log_data = {
                "level": record.levelname,
                "file": record.filename,
                "message": record.getMessage(),
            }
            wandb.log(log_data)
        except (wandb.Error, ValueError):
            self.handleError(record)


class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Format the time, level, and filename as usual
        formatted_time = self.formatTime(record, self.datefmt)
        levelname = record.levelname
        filename = record.filename[:-3]

        # Calculate the current length
        current_length = (
            len(formatted_time) + len(levelname) + len(filename) + 6
        )  # Including spaces and dashes

        # If the combined length is less than 50, calculate the needed padding
        padding = " " * (50 - current_length) if current_length < 50 else ""

        # Construct the log message with padding if necessary
        record.message = record.getMessage()
        formatted_record = (
            f"{formatted_time} - {levelname} - {filename}{padding} - {record.message}"
        )

        return formatted_record


class ColoredFormatter(CustomFormatter):
    def format(self, record):
        # First, use the original formatter to get the formatted message
        formatted_message = super().format(record)

        # Apply color based on the level of the log record
        levelname = record.levelname
        if levelname in COLORS:
            # Wrap the entire message in the color sequence
            colored_message = (
                COLOR_SEQ.format(COLORS[levelname]) + formatted_message + RESET_SEQ
            )
            return colored_message
        else:
            return formatted_message


class Logger(logging.Logger):
    def __init__(self):
        super().__init__(name="ProjectLogger")
        self.logs_folder = config.get_subpath("logs")
        self.base_format = "%(asctime)s - %(levelname)s - %(filename)s - %(message)s"
        self.Formatter = CustomFormatter

        self._setup_file_handler("debug.log", logging.DEBUG)
        self._setup_file_handler("info.log", logging.INFO)
        self._setup_file_handler("warning.log", logging.WARNING)
        self._setup_file_handler("error.log", logging.ERROR)
        self._setup_stream_handler(config["logging"]["console"])
        self._setup_wandb_handler(config["logging"]["wandb"])

    def _setup_file_handler(self, file_name: str, level: int, format_str: str = None):
        path = os.path.join(self.logs_folder, file_name)
        handler = logging.FileHandler(path)
        handler.setLevel(level)
        if format_str is None:
            formatter = self.Formatter(self.base_format)
        else:
            formatter = self.Formatter(format_str)
        handler.setFormatter(formatter)
        self.addHandler(handler)
        return handler

    def _setup_stream_handler(self, level: int, format_str: str = None):
        handler = logging.StreamHandler()
        handler.setLevel(level)
        if format_str is None:
            formatter = ColoredFormatter(self.base_format)
        else:
            formatter = ColoredFormatter(format_str)
        handler.setFormatter(formatter)
        self.addHandler(handler)
        return handler

    def _setup_wandb_handler(self, level: int = logging.INFO):
        handler = _WandbHandler()
        handler.setLevel(level)
        formatter = self.Formatter(self.base_format)
        handler.setFormatter(formatter)
        self.addHandler(handler)
        return handler
