#
#   Lightnet logger: Logging functionality used within the lightnet package
#   Copyright EAVISE
#

from enum import IntEnum, Enum

__all__ = ['log', 'Loglvl']


class Loglvl(IntEnum):
    """ Different levels of logging.

    Attributes:
        ALL: Show all log messages.
        DEBUG: Show all debug messages and above.
        VERBOSE: Show all verbose messages and above.
        WARN: Show all warn messages and above. This is the default.
        ERROR: Show all error messages and above.
        NONE: Show no log messages.
    """
    ALL         = -1
    NONE        = 999

    DEBUG       = 0
    VERBOSE     = 1
    WARN        = 2
    ERROR       = 3


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


def colorize(msg, color):
    return ColorCode.BOLD.value + color.value + msg + ColorCode.RESET.value


class Logger:
    """ Logging functionality. An instance log is created for this entire package upon first import.
    Use it by accessing ``lightnet.log()``. 

    Args:
       lvl (lightnet.logger.Loglvl): The log level for this message
       msg (str): The message to print out
       error (Error, optional): Optional error to raise with the message
    """
    def __init__(self):
        self.level = Loglvl.WARN
        self.color = True
        self.lvl_msg = ['[DEBUG]   ', '[VERBOSE] ', '[WARN]    ', '[ERROR]   ']
        self.lvl_col = [ColorCode.GRAY, ColorCode.WHITE, ColorCode.YELLOW, ColorCode.RED]

    def __call__(self, lvl, msg, error=None):
        """ Print out log message if lvl is higher than the set Loglvl """
        if lvl >= self.level:
            if lvl < len(self.lvl_msg):
                pre_msg = self.lvl_msg[lvl]
                if self.color:
                    pre_msg = colorize(pre_msg, self.lvl_col[lvl])
            else:
                pre_msg = '       '
            if error is None:
                print(f'{pre_msg} {msg}')

        if error is not None:
            if lvl >= self.level:
                raise error(f'\n{pre_msg} {msg}')
            else:
                raise error


# Create single logger object
try:
    log
except NameError:
    log = Logger()
