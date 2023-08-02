import logging
import sys

try:
    import tqdm

    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False


class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


# Idea borrowed from
# https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/utils/logging.py
def get_logger(name=None, level=logging.INFO) -> logging.Logger:
    """Returns a logger that is configured as:
    - by default INFO level or higher messages are logged out in STDOUT.
    - format includes file name, line number, etc.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.hasHandlers():
        # Remove existing handlers so that capsys can capture
        # the output from patched sys.stdout
        for handler in logger.handlers:
            logger.removeHandler(handler)

    log_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
    )

    # Send everything to stdout
    if HAVE_TQDM:
        handler_out: logging.StreamHandler = TqdmLoggingHandler(sys.stdout)
    else:
        handler_out = logging.StreamHandler(sys.stdout)
    handler_out.setFormatter(log_formatter)
    logger.addHandler(handler_out)

    return logger


# Delay evaluation of "logger" attribute so that capsys can capture the
# output of this logger.
def __getattr__(name):
    if name == "logger":
        return get_logger(name="feynman", level=logging.INFO)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
