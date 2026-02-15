import logging
import os
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler

import dotenv
import git

def load_envs(env_file: Optional[str] = None) -> None:
    """Load environment variables from a file.

    This is equivalent to sourcing the file in a shell.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: The file that defines the environment variables to use. If None,
                     it searches for a `.env` file in the project.
    """
    if env_file is None:
        env_file = dotenv.find_dotenv(usecwd=True)
    dotenv.load_dotenv(dotenv_path=env_file, override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)

load_envs()

PROJECT_ROOT = Path(
    git.Repo(Path.cwd(), search_parent_directories=True).working_dir
)

logger.debug(f"Inferred project root: {PROJECT_ROOT}")
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

__all__ = ["PROJECT_ROOT"]