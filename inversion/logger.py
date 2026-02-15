import logging
from typing import Any, Dict, Callable
from time import strftime, gmtime
import shutil
from wcwidth import wcswidth

def format_time_minutes(seconds: float) -> str:
    return strftime("%M:%S", gmtime(seconds))

def format_time_hours(seconds: float) -> str:
    return strftime("%H:%M:%S", gmtime(seconds))

def strip(msg: str) -> str:
    return msg.replace('\n', '').replace('\r', '')


class Logging:
    def __init__(self, log_path, log_name, defined: Dict[str, str] = dict()):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(f'{log_path}/{log_name}-{strftime("%Y_%m_%d__%H_%M_%S")}.log')],
            force=True
        )
        self.logger = logging.getLogger(log_name)

        self._defined = defined
    
    def require_defined(self, name: str) -> str:
        if name not in self._defined:
            raise AttributeError(f"'{name}' is not defined.")
        return self._defined[name]
        
    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def critical(self, msg: str) -> None:
        self.logger.critical(msg)

    def write(self, msg: str, end: str) -> None:
        stripped = strip(msg)

        print(msg, end=end, flush=True)
        if stripped != '': self.info(stripped)

    def new_line(self):
        self.write('', end='\n')

    def __getattr__(self, name: str) -> Callable:
        sentinel = object()          # private marker that can’t clash with user input

        def dynamic_method(*args: Any, end: str = sentinel) -> None:
            msg = self.require_defined(name)
            msg = msg.format(*args)

            if end is sentinel:
                # current terminal width; fallback to 80 if stdout isn't a TTY
                width = shutil.get_terminal_size(fallback=(80, 24)).columns

                # we only care about the *visible* length of the last line
                last_line_len = wcswidth(msg.splitlines()[-1])
                
                padding = max(width - last_line_len, 0)
                end = " " * padding

            self.write(msg, end)

        return dynamic_method