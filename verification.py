import os
import sys
from typing import Any
import numpy
from numpy.typing import NDArray
from config import DATABASE_DIR_PATH, DATASET_DIR_PATH, FINGERPRINTS_IMAGE_FILE_EXTENSION
from fingerprint import FM_CONFIG, Fingerprint
from utils import FINGERPRINTS_DATABASE_FILE_EXTENSION, ExitCode, HelpCommand

dataset_dir_path = os.path.normpath(DATASET_DIR_PATH)
database_dir_path = os.path.normpath(DATABASE_DIR_PATH)

USAGE = "Usage"
COMMAND = "Command"
IDENTITY_FULL_TAG = "identity_full_tag"
TEMPLATE_FULL_FINGER_TAG = "template_full_finger_tag"

def help_message(executable_name: str) -> str:
    return f"""{USAGE}: {executable_name} [{COMMAND}]

{COMMAND}:
    {HelpCommand.Full}, {HelpCommand.Short}, {HelpCommand.Long}                                              Display this message (default: displayed when no arguments are provided)
    <{IDENTITY_FULL_TAG}> <{TEMPLATE_FULL_FINGER_TAG}>   Verify if the subject with tag `{IDENTITY_FULL_TAG}` matches the one with `{TEMPLATE_FULL_FINGER_TAG}`"""

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

    identity_full_tag, *other_arguments = cli_arguments
    if len(other_arguments) == 0:
        # `+ 1` in the middle for the spaces in between arguments
        pointers_and_cause_offset = len(executable_name) + 1 + len(identity_full_tag)

        print(
f"""Error: missing {TEMPLATE_FULL_FINGER_TAG}
|
| {executable_name} {identity_full_tag}
| {"":>{pointers_and_cause_offset}} ^
""", file = sys.stderr
        )
        return ExitCode.Failure

    template_full_finger_tag, *unexpected_cli_arguments = other_arguments
    if len(unexpected_cli_arguments) > 0:
        # `+ 1` in the middle for the spaces in between arguments
        pointers_and_cause_offset = len(executable_name) + 1 + len(identity_full_tag) + 1 + len(template_full_finger_tag)

        unexpected_cli_arguments_text = " ".join(unexpected_cli_arguments)
        unexpected_cli_arguments_pointers = ["^" * len(unexpected_cli_argument) for unexpected_cli_argument in unexpected_cli_arguments]
        pointers = " ".join(unexpected_cli_arguments_pointers)

        print(
f"""Error: unexpected arguments
|
| {executable_name} {identity_full_tag} {template_full_finger_tag} {unexpected_cli_arguments_text}
| {"":>{pointers_and_cause_offset}} {pointers}
""", file = sys.stderr
        )
        return ExitCode.Failure

    identity_image_file_path = os.path.join(dataset_dir_path, identity_full_tag) + FINGERPRINTS_IMAGE_FILE_EXTENSION
    identity_image_file_path = os.path.normpath(identity_image_file_path)
    if not os.path.exists(identity_image_file_path):
        pointers_and_cause_offset = len(executable_name)
        pointers = "^" * len(identity_full_tag)

        print(
f"""Error: fingerprint file with tag `{identity_full_tag}` does not exit (`{identity_image_file_path}`)
|
| {executable_name} {identity_full_tag} {template_full_finger_tag}
| {"":>{pointers_and_cause_offset}} {pointers} does not exist
""", file = sys.stderr
        )
        return ExitCode.Failure

    template_database_path = os.path.join(database_dir_path, template_full_finger_tag) + FINGERPRINTS_DATABASE_FILE_EXTENSION
    template_database_path = os.path.normpath(template_database_path)
    if not os.path.exists(template_database_path):
        # `+ 1` in the middle for the spaces in between arguments
        pointers_and_cause_offset = len(executable_name) + 1 + len(identity_full_tag)
        pointers = "^" * len(template_full_finger_tag)

        print(
f"""Error: database file with tag `{template_full_finger_tag}` does not exit (`{template_database_path}`)
|
| {executable_name} {identity_full_tag} {template_full_finger_tag}
| {"":>{pointers_and_cause_offset}} {pointers} does not exist
""", file = sys.stderr
        )
        return ExitCode.Failure

    template_database_full_finger_tag, template_database_extension = os.path.splitext(template_database_path)
    if template_database_extension != FINGERPRINTS_DATABASE_FILE_EXTENSION:
        pointers_and_cause_offset = len(template_database_path) - len(template_database_extension)
        pointers = "^" * len(template_database_extension)

        print(
f"""Error: ignoring file `{template_database_path}`, wrong file extension
|
| {template_database_path}
| {"":>{pointers_and_cause_offset}}{pointers} expected `{FINGERPRINTS_DATABASE_FILE_EXTENSION}`, got `{template_database_extension}`
""", file = sys.stderr
        )
        return ExitCode.Failure

    normalized_identity_full_tag = os.path.normpath(identity_full_tag)
    identity_fingerprint_tag = os.path.basename(normalized_identity_full_tag)
    _identity_finger_tag, identity_acquisition_tag = identity_fingerprint_tag.split(sep = "_")
    identity = Fingerprint.from_config(
        identity_image_file_path,
        identity_acquisition_tag,
        mcc_reference_cell_coordinates = None,
    )

    _database_file_path, *template_full_finger_tag_components = template_database_full_finger_tag.split(os.path.sep)
    normalized_template_full_finger_tag = os.path.join(*template_full_finger_tag_components)
    template_full_finger_tag = normalized_template_full_finger_tag.replace("\\", "/")

    templates: NDArray[Any] = numpy.load(template_database_path, allow_pickle = True)
    total_fingerprints = 0
    total_matching_score = 0
    for template in templates:
        template: Fingerprint

        normalized_template_full_tag = f"{normalized_template_full_finger_tag}_{template.acquisition_tag}"
        if normalized_identity_full_tag == normalized_template_full_tag:
            continue
        total_fingerprints += 1

        matching_score = identity.matching_score(template, FM_CONFIG.matching_algorithm)
        total_matching_score += matching_score

    average_matching_score = total_matching_score / total_fingerprints
    result: str
    if average_matching_score >= FM_CONFIG.matching_score_genuine_threshold.value:
        result = "matches"
    else:
        result = "does not match"
    average_matching_score = round(average_matching_score, ndigits = 2)
    print(f"Info: verifying `{identity_full_tag}` against `{template_full_finger_tag}` = {average_matching_score:.2f}/{FM_CONFIG.matching_score_genuine_threshold.value:.2f} -> {result}")

    return ExitCode.Success

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
