from enum import IntEnum
import os
import sys
from typing import Any, NoReturn

import numpy
from numpy.typing import NDArray

from config import MatchingAlgorithm
from fingerprint import FM_CONFIG, FingerprintAcquisition
from fingerprint import MinutiaKind # type: ignore (needed for numpy.load)
from utils import DATABASE_DIR_PATH, FINGERPRINTS_DATABASE_FILE_EXTENSION
database_dir_path = os.path.normpath(DATABASE_DIR_PATH)

class ExitCode(IntEnum):
    Success = 0
    Failure = 1

def main() -> ExitCode:
    def on_walk_error_raise(error: OSError) -> NoReturn:
        raise error

    databases: list[tuple[str, list[FingerprintAcquisition]]] = []
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
            acquisitions: list[FingerprintAcquisition] = database.tolist() # type: ignore
            databases.append((fingerprint_tag, acquisitions))

    print(f"Info: genuine matching score threshold = {FM_CONFIG.matching_score_genuine_threshold.value}\n")

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    fingerprints_database_index = 0
    while fingerprints_database_index < len(databases) - 1:
        fingerprints_tag, fingerprints_database = databases[fingerprints_database_index]
        fingerprints_database_index += 1

        fingerprint_index = 0
        while fingerprint_index < len(fingerprints_database) - 1:
            fingerprint = fingerprints_database[fingerprint_index]
            print(f"Info: evaluating `{fingerprints_tag}_{fingerprint.tag}`")

            fingerprint_index += 1

            # testing non match rates against other identities
            other_fingerprints_database_index = fingerprints_database_index
            while other_fingerprints_database_index < len(fingerprints_database):
                _other_fingerprints_tag, other_fingerprints = databases[other_fingerprints_database_index]
                other_fingerprints_database_index += 1

                next_other_fingerprint_index = 0
                while next_other_fingerprint_index < len(other_fingerprints):
                    next_other_fingerprint = other_fingerprints[next_other_fingerprint_index]
                    next_other_fingerprint_index += 1

                    matching_score: float
                    match FM_CONFIG.matching_algorithm:
                        case MatchingAlgorithm.LocalStructures:
                            matching_score = fingerprint.features.matching_score_local_structures(
                                next_other_fingerprint.features,
                            )
                        case MatchingAlgorithm.Hough:
                            matching_score = fingerprint.features.matching_score_hough(
                                next_other_fingerprint.features,
                            )
                    if matching_score >= FM_CONFIG.matching_score_genuine_threshold.value:
                        false_positives += 1
                    else:
                        true_negatives += 1

            # testing non match rates against the same identity
            next_same_fingerprint_index = fingerprint_index
            while next_same_fingerprint_index < len(fingerprints_database):
                next_same_fingerprint = fingerprints_database[next_same_fingerprint_index]
                next_same_fingerprint_index += 1

                matching_score: float
                match FM_CONFIG.matching_algorithm:
                    case MatchingAlgorithm.LocalStructures:
                        matching_score = fingerprint.features.matching_score_local_structures(
                            next_same_fingerprint.features,
                        )
                    case MatchingAlgorithm.Hough:
                        matching_score = fingerprint.features.matching_score_hough(
                            next_same_fingerprint.features,
                        )
                matching_score = fingerprint.features.matching_score_local_structures(next_same_fingerprint.features)
                if matching_score >= FM_CONFIG.matching_score_genuine_threshold.value:
                    true_positives += 1
                else:
                    false_negatives += 1

    far: float
    if (real_rejections_count := false_positives + true_negatives) == 0:
        far = float("nan")
    else:
        far = false_positives / real_rejections_count

    frr: float
    if (real_acceptions_count := false_negatives + true_positives) == 0:
        frr = float("nan")
    else:
        frr = false_negatives / real_acceptions_count

    print(f"Result: FAR = {far}, FRR = {frr}")

    return ExitCode.Success

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
