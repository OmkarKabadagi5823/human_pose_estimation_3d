__all__ = ['Logger']

import logging

class Logger(logging.Logger):
    """
    Custom logger for logging
    """
    
    def __init__(self, name, level=logging.INFO):
        super().__init__(name, level)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(CustomFormatter())
        self.addHandler(console_handler)
        
        return

class CustomFormatter(logging.Formatter):
    """
    Custom formatter for logging
    """
    
    white_bold = "\x1b[1;1m"
    grey = "\x1b[37;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: white_bold + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)