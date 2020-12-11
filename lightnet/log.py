#
#   Lightnet logger: Logging functionality used within the lightnet package
#   Copyright EAVISE
#
import os
import sys
import types
import logging
import copy
from enum import Enum

__all__ = ['logger']


# Formatter
class ColorCode(Enum):
    """ Color Codes """
    RESET = '\033[00m'
    BOLD = '\033[01m'

    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    WHITE = '\033[37m'
    GRAY = '\033[1;30m'


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, color=True, **kwargs):
        logging.Formatter.__init__(self, msg, **kwargs)
        self.color = color
        self.color_codes = {
            'CRITICAL': ColorCode.RED,
            'ERROR': ColorCode.RED,
            'TRAIN': ColorCode.BLUE,
            'TEST': ColorCode.BLUE,
            'DEPRECATED': ColorCode.YELLOW,
            'EXPERIMENTAL': ColorCode.YELLOW,
            'WARNING': ColorCode.YELLOW,
            'INFO': ColorCode.WHITE,
            'DEBUG': ColorCode.GRAY,
            'METADATA': ColorCode.GRAY,
        }

    def format(self, record):
        record = copy.copy(record)
        levelname = record.levelname
        name = record.name
        if self.color:
            color = self.color_codes[levelname] if levelname in self.color_codes else ''
            record.levelname = f'{ColorCode.BOLD.value}{color.value}{levelname:12}{ColorCode.RESET.value}'
        else:
            record.levelname = f'{levelname:12}'
        return logging.Formatter.format(self, record)

    def setColor(self, value):
        """ Enable or disable colored output for this handler """
        self.color = value


# Filter
class LevelFilter(logging.Filter):
    def __init__(self, levels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.levels = levels

    def filter(self, record):
        if self.levels is None or record.levelname in self.levels:
            return True
        else:
            return False


# Logging levels
def log_once_function(lvl, msg_set):
    def log_once(self, message, *args, **kwargs):
        if not hasattr(self, msg_set):
            setattr(self, msg_set, set())

        msgs = getattr(self, msg_set)
        if self.isEnabledFor(lvl) and message not in msgs:
            msgs.add(message)
            self._log(lvl, message, args, **kwargs)

    return log_once


def log_function(lvl):
    def log(self, message, *args, **kwargs):
        if self.isEnabledFor(lvl):
            self._log(lvl, message, args, **kwargs)

    return log


def test(self, message, *args, **kwargs):
    if self.isEnabledFor(38):
        self._log(38, message, args, **kwargs)


def train(self, message, *args, **kwargs):
    if self.isEnabledFor(39):
        self._log(39, message, args, **kwargs)


# Metadata should usually not be printed to the console and is mainly used for logfiles
logging.addLevelName(1, 'METADATA')
logging.Logger.metadata = log_function(1)

# Experimental is a special warning mode, and thus it should be filtered the same (lower) as warning
logging.addLevelName(28, 'EXPERIMENTAL')
logging.Logger.experimental = log_once_function(28, 'experimental_msgs')

# Deprecated is a special warning mode, and thus it should be filtered the same (lower) as warning
logging.addLevelName(29, 'DEPRECATED')
logging.Logger.deprecated = log_once_function(29, 'deprecated_msgs')

# error_once function logs an error, but only once (useful in loops, but slows down code!)
logging.Logger.error_once = log_once_function(40, 'error_msgs')

# Test is a special info log, but with a much higher level. We only filter it with the highest log setting
logging.addLevelName(48, 'TEST')
logging.Logger.test = log_function(48)

# Train is a special info log, but with a much higher level. We only filter it with the highest log setting
logging.addLevelName(49, 'TRAIN')
logging.Logger.train = log_function(49)


# Console Handler
ch = logging.StreamHandler()
ch.setFormatter(ColoredFormatter('{levelname} {message}', style='{'))
if 'LN_LOGLVL' in os.environ:
    lvl = os.environ['LN_LOGLVL'].upper()
    try:
        ch.setLevel(int(lvl))
    except ValueError:
        ch.setLevel(lvl)

    if ch.level <= 10:
        ch.setFormatter(ColoredFormatter('{levelname} [{name}] {message}', style='{'))
else:
    ch.setLevel(logging.INFO)


# File Handler
def createFileHandler(self, filename, levels=None, filemode='a'):
    """ Create a file to write log messages of certaing levels """
    fh = logging.FileHandler(filename=filename, mode=filemode)
    fh.setLevel(logging.NOTSET)
    fh.addFilter(LevelFilter(levels))
    fh.setFormatter(logging.Formatter('{levelname} [{name}] {message}', style='{'))
    logger.addHandler(fh)
    return fh


# Logger
logger = logging.getLogger('lightnet')
logger.setLevel(1)
logger.addHandler(ch)
logger.setConsoleLevel = ch.setLevel
logger.setConsoleColor = ch.formatter.setColor
logger.setLogFile = types.MethodType(createFileHandler, logger)

# Disable color if ANSI not supported -> Code taken from django.core.management.color.supports_color
# Note that if you use the colorama plugin, you can reenable the colors
supported_platform = sys.platform != 'Pocket PC' and (sys.platform != 'win32' or 'ANSICON' in os.environ)
is_a_tty = hasattr(ch.stream, 'isatty') and ch.stream.isatty()
if not supported_platform or not is_a_tty:
    logger.setConsoleColor(False)
