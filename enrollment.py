from __future__ import annotations
import sys
import os
import numpy
from fingerprint import FE_CONFIG, Fingerprint, FingerprintAcquisition, MccReferenceCellCoordinates
from utils import DATABASE_DIR_PATH, DATASET_DIR_PATH, FINGERPRINTS_IMAGE_FILE_EXTENSION, ExitCode, on_walk_error_raise

dataset_dir_path = os.path.normpath(DATASET_DIR_PATH)
database_dir_path = os.path.normpath(DATABASE_DIR_PATH)

type DatabaseEnrollment = dict[str, list[FingerprintAcquisition]]


def main() -> ExitCode:
    fingerprint_file_paths: list[str] = []
    database_finger_tags: list[str] = []
    acquisition_tags: list[str] = []

    for dir_path, _dir_names, file_paths in os.walk(dataset_dir_path, onerror = on_walk_error_raise):
        for file_path in file_paths:
            file_name, file_extension = os.path.splitext(file_path)
            fingerprint_file_path = os.path.join(dir_path, file_path)
            if file_extension != FINGERPRINTS_IMAGE_FILE_EXTENSION:
                pointers_and_cause_offset = len(fingerprint_file_path) - len(file_extension)
                pointers = "^" * len(file_extension)

                print(
f"""Warning: ignoring file `{fingerprint_file_path}`, wrong file extension
|
| {fingerprint_file_path}
| {"":>{pointers_and_cause_offset}}{pointers} expected `{FINGERPRINTS_IMAGE_FILE_EXTENSION}`, got `{file_extension}`
""", file = sys.stderr
                )
                continue

            file_tags = file_name.split(sep = "_")
            if len(file_tags) != 2:
                pointers_and_cause_offset = len(dir_path)
                pointers = "^" * len(file_name)
                print(
f"""Error: wrong file format
|
| {fingerprint_file_path}
| {"":>{pointers_and_cause_offset}} {pointers} should be in the format `[finger-tag]_[acquisition-tag]`, i.e.: 101_1
""", file = sys.stderr
                )
                return ExitCode.Failure

            database_finger_tag, acquisition_tag = file_tags
            database_finger_tag = os.path.join(dir_path, database_finger_tag)

            fingerprint_file_paths.append(fingerprint_file_path)
            database_finger_tags.append(database_finger_tag)
            acquisition_tags.append(acquisition_tag)

    mcc_reference_cell_coordinates = MccReferenceCellCoordinates(
        FE_CONFIG.mcc_total_radius,
        FE_CONFIG.mcc_circles_radius,
    )
    database: DatabaseEnrollment = {}
    for fingerprint_file_path, acquisition_tag, database_finger_tag in zip(
        fingerprint_file_paths,
        acquisition_tags,
        database_finger_tags,
        strict = True
    ):
        print(f"Info: enrolling `{fingerprint_file_path}`")
        fingerprint = Fingerprint(
            fingerprint_file_path,
            acquisition_tag,
            mcc_reference_cell_coordinates = mcc_reference_cell_coordinates,
        )
        acquisitions = database.setdefault(database_finger_tag, [])
        acquisitions.append(fingerprint.acquisition)

    for (database_finger_tag, acquisitions) in database.items():
        _root, *components, tag = database_finger_tag.split(os.path.sep)
        sub_dirs = os.path.join(database_dir_path, *components)
        os.makedirs(sub_dirs, exist_ok = True)
        database_file_path = os.path.join(sub_dirs, tag)

        acquisitions_text = "acquisition" if len(acquisitions) == 1 else "acquisitions"
        print(f"Info: saving `{database_file_path}` with {len(acquisitions)} {acquisitions_text} per fingerprint")

        numpy.save(
            database_file_path,
            arr = acquisitions, # type: ignore
            allow_pickle = True,
        )

    return ExitCode.Success

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
