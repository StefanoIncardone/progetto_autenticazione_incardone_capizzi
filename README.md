# Biometric Authentication for IT Systems Security project

The aim of this project is to develop a multi-instance biometric fingerprint identification and
verification system, with the goal of giving a quantitative performance and accuracy evaluation.

## System Requirements

- Python version 3.13

## Project's structure

### Features extraction

- [`config.py`](config.py): configuration parameters used during features extraction
- [`fingerprint.py`](fingerprint.py): algorithms and structures used to extract features
- [`references/`](references/): papers and documents used as references for the features
    extraction process

### Enrollment and database creation

- [`enrollment.py`](enrollment.py): features extraction of all the fingerprints in the folder
    specified with the `DATASET_DIR_PATH` configuration key, to create the database folder
    specified with the `DATABASE_DIR_PATH` configuration key
- used datasets FVC2006, the filenames are structured in the following way:
    - `database full tag`: `DATASET_DIR_PATH/FVC2006/db1_b/101_1`
    - `full tag`: `FVC2006/db1_b/101_1`
    - `database tag`: `FVC2006/db1_b/`
    - `finger tag`: `101`
    - `acquisition tag`: `1`
    - `full finger tag`: `FVC2006/db1_b/101`
    - `fingerprint tag`: `101_1`

### Identification and Verification

- [`identification.py`](identification.py), identification module:
    get the declared identity as input in order to identify it and return the identity it represents
- [`verification.py`](verification.py), verification module:
    gets the declared identity as input and an expected identity to verify if the declared identity
    is genuine or an impostor

### Performance evaluation

- [`performance_evaluation.py`](performance_evaluation.py): evaluates the performance of the
    matching algorithm in order to measure it's **FAR** and **FRR** characteristics

### Interactive visualization of the features extraction and matching process

- [`interactive_visualization.ipynb`](interactive_visualization.ipynb): interactive notebook to
    visualize in real time the features extraction and matching process

## Usage

1. **enrollment.py**, creation of the database:

    ```shell
    python3.13 enrollment.py
    ```

2. **verification.py**, verification of the declared identity:
    - usage help:

        ```shell
        python3.13 verification.py --help
        ```

    - fingerprint verification:

        ```shell
        python3.13 verification.py FVC2006/db1_1/101_1 FVC2006/db1_1/101
        ```

3. **identification.py**, identification of the declared identity:
    - usage help:

        ```shell
        python3.13 identification.py --help
        ```

    - fingeprint identification:

        ```shell
        python3.13 identification.py FVC2006/db1_1/101_1
        ```

4. **performance_evaluation.py**, **FAR** and **FRR** metrics performance evaluation

    ```shell
    python3.13 performance_evaluation.py
    ```
