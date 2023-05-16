import getopt
import logging
from logging.handlers import RotatingFileHandler
import sys

def configurate_logger(name: str = __name__):
    """Configure logging
        command-line arguments
        --log=DEBUG or --log=debug
        --logfile=hh_parser.log
    """
    log_level = logging.DEBUG
    logfile = 'hh_parser.log'
    for argument, value in getopt.getopt(sys.argv[1:], [], ['log=', 'logfile='])[0]:
        if argument in {'--log'}:
            log_level = getattr(logging, value.upper(), None)
            if not isinstance(log_level, int):
                raise ValueError(f'Invalid log level: {value}')
        elif argument in {'--logfile'}:
            logfile = value

    format_str = '%(asctime)s  %(levelname)s:  %(message)s'

    log = logging.getLogger(name)
    log.setLevel(log_level)

    formatter = logging.Formatter(format_str)
    handler = RotatingFileHandler(logfile, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    handler.setFormatter(formatter)
    handler.setLevel(log_level)
    log.addHandler(handler)

    formatter = logging.Formatter(format_str)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    log.addHandler(handler)

    return log