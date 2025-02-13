from enum import IntEnum
from typing import NoReturn, override

DATASET_DIR_PATH = "datasets"
FINGERPRINTS_IMAGE_FILE_EXTENSION = ".tif"

DATABASE_DIR_PATH = "database"
FINGERPRINTS_DATABASE_FILE_EXTENSION = ".npy"

class HelpCommand(IntEnum):
    Full = 0
    Short = 1
    Long = 2

    @override
    def __str__(self) -> str:
        match self:
            case HelpCommand.Full: return "help"
            case HelpCommand.Short: return "--h"
            case HelpCommand.Long: return "--help"

class ExitCode(IntEnum):
    Success = 0
    Failure = 1

def on_walk_error_raise(error: OSError) -> NoReturn:
    raise error
