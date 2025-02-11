from enum import IntEnum
import os
import sys
from typing import Any, NoReturn

import numpy
from numpy.typing import NDArray

from config import DATABASE_DIR_PATH
from fingerprint import FM_CONFIG, Fingerprint
from utils import FINGERPRINTS_DATABASE_FILE_EXTENSION
database_dir_path = os.path.normpath(DATABASE_DIR_PATH)

class ExitCode(IntEnum):
    Success = 0
    Failure = 1

def main() -> ExitCode:
    def on_walk_error_raise(error: OSError) -> NoReturn:
        raise error

    databases: list[tuple[str, list[Fingerprint]]] = []
    for dir_path, _dir_names, file_paths in os.walk(database_dir_path, onerror = on_walk_error_raise):
        for file_path in file_paths:
            database_file_path = os.path.join(dir_path, file_path)
            fingerprint_tag, fingerprint_database_file_extension = os.path.splitext(database_file_path)
            if fingerprint_database_file_extension != FINGERPRINTS_DATABASE_FILE_EXTENSION:
                pointers_and_cause_offset = len(database_file_path) - len(fingerprint_database_file_extension)
                pointers = "^" * len(fingerprint_database_file_extension)

                print(
f"""Warning: ignoring file `{database_file_path}`, wrong file extension
|
| {database_file_path}
| {"":>{pointers_and_cause_offset}}{pointers} expected `{FINGERPRINTS_DATABASE_FILE_EXTENSION}`, got `{fingerprint_database_file_extension}`
""", file = sys.stderr
                )
                continue

            database: NDArray[Any] = numpy.load(database_file_path, allow_pickle = True)
            acquisitions: list[Fingerprint] = database.tolist() # type: ignore
            databases.append((fingerprint_tag, acquisitions))

    print(f"Info: genuine matching score threshold = {FM_CONFIG.matching_score_genuine_threshold.value}\n")

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    fingerprints_database_index = 0
    while fingerprints_database_index < len(databases):
        fingerprints_tag, fingerprints_database = databases[fingerprints_database_index]
        fingerprints_database_index += 1

        for fingerprint_index, fingerprint in enumerate(fingerprints_database):
            print(f"Info: evaluating `{fingerprints_tag}_{fingerprint.acquisition_tag}`")

            # testing non match rates against the same identity
            next_same_fingerprint_index = fingerprint_index + 1
            while next_same_fingerprint_index < len(fingerprints_database):
                next_same_fingerprint = fingerprints_database[next_same_fingerprint_index]
                next_same_fingerprint_index += 1

                matching_score = fingerprint.matching_score(next_same_fingerprint, FM_CONFIG.matching_algorithm)
                result: str
                if matching_score >= FM_CONFIG.matching_score_genuine_threshold.value:
                    true_positives += 1
                    result = "matches"
                else:
                    false_negatives += 1
                    result = "does not match"
                matching_score = round(matching_score, ndigits = 2)

                print(f"    Info: against `{fingerprints_tag}_{next_same_fingerprint.acquisition_tag}` = {matching_score:.2f}/{FM_CONFIG.matching_score_genuine_threshold.value:.2f} -> {result}")

            # testing non match rates against other identities
            other_fingerprints_database_index = fingerprints_database_index
            while other_fingerprints_database_index < len(databases):
                other_fingerprints_tag, other_fingerprints = databases[other_fingerprints_database_index]
                other_fingerprints_database_index += 1

                other_total_matching_score = 0
                for _next_other_fingerprint_index, next_other_fingerprint in enumerate(other_fingerprints):
                    matching_score = fingerprint.matching_score(next_other_fingerprint, FM_CONFIG.matching_algorithm)
                    if matching_score >= FM_CONFIG.matching_score_genuine_threshold.value:
                        false_positives += 1
                    else:
                        true_negatives += 1
                    other_total_matching_score += matching_score

                other_average_matching_score = other_total_matching_score / len(other_fingerprints)
                result: str
                if other_average_matching_score >= FM_CONFIG.matching_score_genuine_threshold.value:
                    result = "matches"
                else:
                    result = "does not match"
                other_average_matching_score = round(other_average_matching_score, ndigits = 2)
                print(f"    Info: against `{other_fingerprints_tag}`   = {other_average_matching_score:.2f}/{FM_CONFIG.matching_score_genuine_threshold.value:.2f} -> {result}")
        #     break # for visualization
        # break # for visualization

    real_rejections_count = false_positives + true_negatives
    far = round(false_positives * 100 / real_rejections_count, ndigits = 2)

    real_acceptions_count = false_negatives + true_positives
    frr = round(false_negatives * 100 / real_acceptions_count, ndigits = 2)

    print(f"Result: FAR = {far}%, FRR = {frr}%")

    return ExitCode.Success

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
