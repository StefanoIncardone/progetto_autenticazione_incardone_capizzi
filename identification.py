import os
import sys
import numpy
from numpy.typing import NDArray
from typing import Any
from config import DATABASE_DIR_PATH, DATASET_DIR_PATH, FINGERPRINTS_IMAGE_FILE_EXTENSION
from fingerprint import FM_CONFIG, Fingerprint
from utils import FINGERPRINTS_DATABASE_FILE_EXTENSION, ExitCode, on_walk_error_raise, HelpCommand

dataset_dir_path = os.path.normpath(DATASET_DIR_PATH)
database_dir_path = os.path.normpath(DATABASE_DIR_PATH)

USAGE = "Usage"
COMMAND = "Command"
UNKNOWN_IDENTITY_FULL_TAG = "unknown_identity_full_tag"

def help_message(executable_name: str) -> str:
    return f"""{USAGE}: {executable_name} [{COMMAND}]

{COMMAND}:
    {HelpCommand.Full}, {HelpCommand.Short}, {HelpCommand.Long}             Display this message (default: displayed when no arguments are provided)
    <{UNKNOWN_IDENTITY_FULL_TAG}>   Identify the subject with tag `{UNKNOWN_IDENTITY_FULL_TAG}`"""

def main() -> ExitCode:
    executable_name, *cli_arguments = sys.argv
    if len(cli_arguments) == 0:
        print(help_message(executable_name))
        return ExitCode.Success

    for cli_argument in cli_arguments:
        match cli_argument:
            case "help" | "-h" | "--help":
                print(help_message(executable_name))
                return ExitCode.Success
            case _: pass

    unknown_identity_full_tag, *unexpected_cli_arguments = cli_arguments
    if len(unexpected_cli_arguments) > 0:
        # `+ 1` in the middle for the spaces in between arguments
        pointers_and_cause_offset = len(executable_name) + 1 + len(unknown_identity_full_tag)

        unexpected_cli_arguments_text = " ".join(unexpected_cli_arguments)
        unexpected_cli_arguments_pointers = ["^" * len(unexpected_cli_argument) for unexpected_cli_argument in unexpected_cli_arguments]
        pointers = " ".join(unexpected_cli_arguments_pointers)

        print(
f"""Error: unexpected arguments
|
| {executable_name} {unknown_identity_full_tag} {unexpected_cli_arguments_text}
| {"":>{pointers_and_cause_offset}} {pointers}
""", file = sys.stderr
        )
        return ExitCode.Failure

    unknown_identity_image_file_path = os.path.join(dataset_dir_path, unknown_identity_full_tag) + FINGERPRINTS_IMAGE_FILE_EXTENSION
    unknown_identity_image_file_path = os.path.normpath(unknown_identity_image_file_path)
    if not os.path.exists(unknown_identity_image_file_path):
        pointers_and_cause_offset = len(executable_name)
        pointers = "^" * len(unknown_identity_full_tag)

        print(
f"""Error: fingerprint file with tag `{unknown_identity_full_tag}` does not exit (`{unknown_identity_image_file_path}`)
|
| {executable_name} {unknown_identity_full_tag}
| {"":>{pointers_and_cause_offset}} {pointers} does not exist
""", file = sys.stderr
        )
        return ExitCode.Failure

    normalized_unknown_identity_full_tag = os.path.normpath(unknown_identity_full_tag)
    unknown_identity_fingerprint_tag = os.path.basename(normalized_unknown_identity_full_tag)
    _unknown_identity_finger_tag, unknown_identity_acquisition_tag = unknown_identity_fingerprint_tag.split(sep = "_")
    unknown_fingerprint = Fingerprint.from_config(
        unknown_identity_image_file_path,
        unknown_identity_acquisition_tag,
        mcc_reference_cell_coordinates = None,
    )

    max_average_matching_score = 0
    matching_identity: str | None = None
    most_similar_identity: str | None = None
    for dir_path, _dir_names, file_paths in os.walk(database_dir_path, onerror = on_walk_error_raise):
        for file_path in file_paths:
            identity_database_file_path = os.path.join(dir_path, file_path)
            identity_database_full_finger_tag, identity_database_extension = os.path.splitext(identity_database_file_path)
            if identity_database_extension != FINGERPRINTS_DATABASE_FILE_EXTENSION:
                pointers_and_cause_offset = len(identity_database_file_path) - len(identity_database_extension)
                pointers = "^" * len(identity_database_extension)

                print(
f"""Warning: ignoring file `{identity_database_file_path}`, wrong file extension
|
| {identity_database_file_path}
| {"":>{pointers_and_cause_offset}}{pointers} expected `{FINGERPRINTS_DATABASE_FILE_EXTENSION}`, got `{identity_database_extension}`
""", file = sys.stderr
                )
                continue

            _database_file_path, *identity_full_finger_tag_components = identity_database_full_finger_tag.split(os.path.sep)
            normalized_identity_full_finger_tag = os.path.join(*identity_full_finger_tag_components)
            identity_full_finger_tag = normalized_identity_full_finger_tag.replace("\\", "/")

            templates: NDArray[Any] = numpy.load(identity_database_file_path, allow_pickle = True)
            total_fingerprints = 0
            total_matching_score = 0
            for template in templates:
                template: Fingerprint

                normalized_identity_full_tag = f"{normalized_identity_full_finger_tag}_{template.acquisition_tag}"
                if normalized_unknown_identity_full_tag == normalized_identity_full_tag:
                    continue
                total_fingerprints += 1

                matching_score = unknown_fingerprint.matching_score(template, FM_CONFIG.matching_algorithm)
                total_matching_score += matching_score

            average_matching_score = total_matching_score / total_fingerprints
            result: str
            if average_matching_score >= FM_CONFIG.matching_score_genuine_threshold.value:
                if average_matching_score > max_average_matching_score:
                    max_average_matching_score = average_matching_score
                    matching_identity = identity_full_finger_tag
                result = "matches"
            else:
                if average_matching_score > max_average_matching_score:
                    max_average_matching_score = average_matching_score
                    most_similar_identity = identity_full_finger_tag
                result = "does not match"
            average_matching_score = round(average_matching_score, ndigits = 2)
            print(f"Info: verifying `{unknown_identity_full_tag}` against `{identity_full_finger_tag}` = {average_matching_score:.2f}/{FM_CONFIG.matching_score_genuine_threshold.value:.2f} -> {result}")

    if matching_identity is not None:
        print(f"Result: fingerprint `{unknown_identity_full_tag}` matches the identity `{matching_identity}`")
    else:
        print(f"Result: fingerprint `{unknown_identity_full_tag}` does not match any known identity")
        print(f"Info: the most similar identity is `{most_similar_identity}`")

    return ExitCode.Success

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
