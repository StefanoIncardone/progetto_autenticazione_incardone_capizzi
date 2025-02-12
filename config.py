from __future__ import annotations
from enum import IntEnum

# ENROLLMENT
DATASET_DIR_PATH = "datasets"
FINGERPRINTS_IMAGE_FILE_EXTENSION = ".tif"
DATABASE_DIR_PATH = "database"

# FEATURE EXTRACTION
GRADIENT_SOBEL_FILTER_LENGTH = 1
"""Must be odd and in the range [1, 31]"""

GRADIENT_MODULE_BLOCK_LENGTH = 31
"""Must be in the range [1, 31]"""

SEGMENTATION_MASK_THRESHOLD_SCALE = 0.3
"""Must be in the range [0.0, 1.0]"""

DIRECTIONAL_MAP_BLOCK_LENGTH = 21
"""Must be in the range [0.0, 1.0]"""

DIRECTIONAL_MAP_BLUR_FILTER_LENGTH = -1
"""Must be odd in the range [1, 31]

When set to -1 no blur filter will be applied
"""

LOCAL_RIDGE_BLOCK_ROWS = 96
"""Must be in the range [0, 256]"""

LOCAL_RIDGE_BLOCK_COLUMNS = 96
"""Must be in the range [0, 256]"""

GABOR_FILTERS_COUNT = 8
"""Must be in the range [1, 16]"""

GABOR_FILTERS_SIGMA = 0.4
GABOR_FILTERS_GAMMA = 1.0

BINARIZATION_BLOCK_SIZE = 31
"""Must be odd and in the range [3, 31]"""

SINGULARITIES_MIN_DISTANCE_FROM_BORDER = 10
"""Must be in the range [0, 255]"""

MINUTIAE_MIN_DISTANCE_FROM_BORDER = 15
"""Must be in the range [0, 255]"""

MINUTIAE_FOLLOWED_LENGTH_MIN = 15
"""Must be in the range [0, 64] and smaller than `MINUTIAE_FOLLOWED_LENGTH_MAX`"""

MINUTIAE_FOLLOWED_LENGTH_MAX = 20
"""Must be in the range [0, 64] and greater than `MINUTIAE_FOLLOWED_LENGTH_MIN`"""

MCC_RADIUS = 50
MCC_DENSITY = 16
MCC_GAUSSIAN_STD = 8.0
MCC_SIGMOID_TAU = 400.0
MCC_SIGMOID_MU = 1e-2

# MATCHING
MATCHING_SCORE_GENUINE_THRESHOLD = 0.5
"""Must be in the range [0.0, 1.0]"""

class MatchingAlgorithmKind(IntEnum):
    LocalStructures = 0
    HoughRatha = 1
    HoughChouta = 2

MATCHING_ALGORITHM = MatchingAlgorithmKind.LocalStructures

LOCAL_STRUCTURES_MATCHING_PAIR_COUNT = 1

HOUGH_MATCHING_PIXELS_DISTANCE_THRESHOLD = 25
"""Must be in the range [0, 255 * sqrt(2)]"""

HOUGH_MATCHING_ANGLE_DISTANCE_THRESHOLD = 20
"""Must be in the range [0, 180]"""

HOUGH_RATHA_MATCHING_ALIGNMENT_ANGLE_FREEDOM = 6
"""Must be in the range [0, 180]"""

HOUGH_RATHA_MATCHING_ALIGNMENT_SCALE_FREEDOM = 4
"""Must be in the range [0, 100]"""

HOUGH_CHOUTA_MATCHING_ERR_FREEDOM = 2
"""Must be in the range [0, 100]"""
