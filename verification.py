import os
import sys
import numpy
from numpy.typing import NDArray
from typing import Any
from config import MatchingAlgorithm
from fingerprint import FM_CONFIG, Fingerprint, FingerprintAcquisition
from utils import DATABASE_DIR_PATH, DATASET_DIR_PATH, FINGERPRINTS_DATABASE_FILE_EXTENSION, FINGERPRINTS_IMAGE_FILE_EXTENSION, ExitCode, HelpCommand

dataset_dir_path = os.path.normpath(DATASET_DIR_PATH)
database_dir_path = os.path.normpath(DATABASE_DIR_PATH)

USAGE = "Usage"
COMMAND = "Command"
TRUE_IDENTITY_FULL_TAG = "true_identity_full_tag"
EXPECTED_IDENTITY_FULL_FINGER_TAG = "expected_identity_full_finger_tag"

def help_message(executable_name: str) -> str:
    return f"""{USAGE}: {executable_name} [{COMMAND}]

{COMMAND}:
    {HelpCommand.Full}, {HelpCommand.Short}, {HelpCommand.Long}                                              Display this message (default: displayed when no arguments are provided)
    <{TRUE_IDENTITY_FULL_TAG}> <{EXPECTED_IDENTITY_FULL_FINGER_TAG}>   Verify if the subject with tag `{TRUE_IDENTITY_FULL_TAG}` matches the one with `{EXPECTED_IDENTITY_FULL_FINGER_TAG}`"""

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

    true_identity_full_tag, *other_arguments = cli_arguments
    if len(other_arguments) == 0:
        # `+ 1` in the middle for the spaces in between arguments
        pointers_and_cause_offset = len(executable_name) + 1 + len(true_identity_full_tag)

        print(
f"""Error: missing {EXPECTED_IDENTITY_FULL_FINGER_TAG}
|
| {executable_name} {true_identity_full_tag}
| {"":>{pointers_and_cause_offset}} ^
""", file = sys.stderr
        )
        return ExitCode.Failure

    expected_identity_full_finger_tag, *unexpected_cli_arguments = other_arguments
    if len(unexpected_cli_arguments) > 0:
        # `+ 1` in the middle for the spaces in between arguments
        pointers_and_cause_offset = len(executable_name) + 1 + len(true_identity_full_tag) + 1 + len(expected_identity_full_finger_tag)

        unexpected_cli_arguments_text = " ".join(unexpected_cli_arguments)
        unexpected_cli_arguments_pointers = ["^" * len(unexpected_cli_argument) for unexpected_cli_argument in unexpected_cli_arguments]
        pointers = " ".join(unexpected_cli_arguments_pointers)

        print(
f"""Error: unexpected arguments
|
| {executable_name} {true_identity_full_tag} {expected_identity_full_finger_tag} {unexpected_cli_arguments_text}
| {"":>{pointers_and_cause_offset}} {pointers}
""", file = sys.stderr
        )
        return ExitCode.Failure

    true_identity_image_file_path = os.path.join(dataset_dir_path, true_identity_full_tag) + FINGERPRINTS_IMAGE_FILE_EXTENSION
    true_identity_image_file_path = os.path.normpath(true_identity_image_file_path)
    if not os.path.exists(true_identity_image_file_path):
        pointers_and_cause_offset = len(executable_name)
        pointers = "^" * len(true_identity_full_tag)

        print(
f"""Error: fingerprint file with tag `{true_identity_full_tag}` does not exit (`{true_identity_image_file_path}`)
|
| {executable_name} {true_identity_full_tag} {expected_identity_full_finger_tag}
| {"":>{pointers_and_cause_offset}} {pointers} does not exist
""", file = sys.stderr
        )
        return ExitCode.Failure

    expected_identity_database_path = os.path.join(database_dir_path, expected_identity_full_finger_tag) + FINGERPRINTS_DATABASE_FILE_EXTENSION
    expected_identity_database_path = os.path.normpath(expected_identity_database_path)
    if not os.path.exists(expected_identity_database_path):
        # `+ 1` in the middle for the spaces in between arguments
        pointers_and_cause_offset = len(executable_name) + 1 + len(true_identity_full_tag)
        pointers = "^" * len(expected_identity_full_finger_tag)

        print(
f"""Error: database file with tag `{expected_identity_full_finger_tag}` does not exit (`{expected_identity_database_path}`)
|
| {executable_name} {true_identity_full_tag} {expected_identity_full_finger_tag}
| {"":>{pointers_and_cause_offset}} {pointers} does not exist
""", file = sys.stderr
        )
        return ExitCode.Failure

    normalized_true_identity_full_tag = os.path.normpath(true_identity_full_tag)
    true_identity_fingerprint_tag = os.path.basename(normalized_true_identity_full_tag)
    _true_identity_finger_tag, true_identity_acquisition_tag = true_identity_fingerprint_tag.split(sep = "_")
    fingerprint_to_verify = Fingerprint(
        true_identity_image_file_path,
        true_identity_acquisition_tag,
        mcc_reference_cell_coordinates = None,
    )

    expected_identity_database_full_finger_tag, expected_identity_database_extension = os.path.splitext(expected_identity_database_path)
    if expected_identity_database_extension != FINGERPRINTS_DATABASE_FILE_EXTENSION:
        pointers_and_cause_offset = len(expected_identity_database_path) - len(expected_identity_database_extension)
        pointers = "^" * len(expected_identity_database_extension)

        print(
f"""Warning: ignoring file `{expected_identity_database_path}`, wrong file extension
|
| {expected_identity_database_path}
| {"":>{pointers_and_cause_offset}}{pointers} expected `{FINGERPRINTS_DATABASE_FILE_EXTENSION}`, got `{expected_identity_database_extension}`
""", file = sys.stderr
        )
        return ExitCode.Failure

    _database_file_path, *expected_identity_full_finger_tag_components = expected_identity_database_full_finger_tag.split(os.path.sep)
    normalized_expected_identity_full_finger_tag = os.path.join(*expected_identity_full_finger_tag_components)
    expected_identity_full_finger_tag = normalized_expected_identity_full_finger_tag.replace("\\", "/")

    total_fingerprints = 0
    total_matching_score = 0
    fingerprints_database: NDArray[Any] = numpy.load(expected_identity_database_path, allow_pickle = True)

    print(f"Info: genuine matching score threshold = {FM_CONFIG.matching_score_genuine_threshold.value}")
    for acquisition in fingerprints_database:
        acquisition: FingerprintAcquisition

        normalized_expected_identity_full_tag = f"{normalized_expected_identity_full_finger_tag}_{acquisition.tag}"
        if normalized_true_identity_full_tag == normalized_expected_identity_full_tag:
            continue
        total_fingerprints += 1

        matching_score: float
        match FM_CONFIG.matching_algorithm:
            case MatchingAlgorithm.LocalStructures:
                matching_score = fingerprint_to_verify.acquisition.features.matching_score_local_structures(
                    acquisition.features,
                )
            case MatchingAlgorithm.Hough:
                matching_score = fingerprint_to_verify.acquisition.features.matching_score_hough(
                    acquisition.features,
                )

        print(f"Info: verifying `{true_identity_full_tag}` against `{expected_identity_full_finger_tag}_{acquisition.tag}` = {round(matching_score, ndigits = 2)}")

        total_matching_score += matching_score

    average_matching_score = round(total_matching_score / total_fingerprints, ndigits = 2)
    print(f"Result: average matching score = {average_matching_score}")
    if average_matching_score >= FM_CONFIG.matching_score_genuine_threshold.value:
        print(f"Verification Result: fingerprint `{true_identity_full_tag}` matches the expected identity of `{expected_identity_full_finger_tag}`")
    else:
        print(f"Verification Result: fingerprint `{true_identity_full_tag}` does not match the expected identity of `{expected_identity_full_finger_tag}`")
    print()

    return ExitCode.Success

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
