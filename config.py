from __future__ import annotations
from enum import IntEnum

# FEATURE EXTRACTION
GRADIENT_SOBEL_FILTER_LENGTH = 1
"""Must be odd and in the range [1, 31]"""

GRADIENT_MODULE_BLOCK_LENGTH = 31
"""Must be in the range [1, 31]"""

SEGMENTATION_MASK_THRESHOLD_SCALE = 0.2
"""Must be in the range [0.0, 1.0]"""

DIRECTIONAL_MAP_BLOCK_LENGTH = 21
"""Must be in the range [0.0, 1.0]"""

DIRECTIONAL_MAP_BLUR_FILTER_LENGTH = -1
"""Must be odd in the range [1, 31]

When set to -1 no blur filter will be applied
"""

LOCAL_RIDGE_BLOCK_ROWS = 96
"""Must be a multiple of 8 and in the range [8, 256]"""

LOCAL_RIDGE_BLOCK_COLUMNS = 96
"""Must be a multiple of 8 and in the range [8, 256]"""

GABOR_FILTERS_COUNT = 8
"""Must be in the range [1, 16]"""

GABOR_FILTERS_SIGMA = 0.4
GABOR_FILTERS_GAMMA = 1.0

BINARIZATION_THRESHOLD = 64
"""Must be in the range [0, 255]"""

SINGULARITIES_MIN_DISTANCE_FROM_BORDER = 10
"""Must be in the range [0, 255]"""

MINUTIAE_MIN_DISTANCE_FROM_BORDER = 10
"""Must be in the range [0, 255]"""

MINUTIAE_FOLLOWED_LENGTH_MIN = 10
"""Must be in the range [0, 64] and smaller than `MINUTIAE_FOLLOWED_LENGTH_MAX`"""

MINUTIAE_FOLLOWED_LENGTH_MAX = 20
"""Must be in the range [0, 64] and greater than `MINUTIAE_FOLLOWED_LENGTH_MIN`"""

MCC_TOTAL_RADIUS = 100
MCC_CIRCLES_RADIUS = 16
MCC_GAUSSIAN_STD = 7.0
MCC_SIGMOID_TAU = 400.0
MCC_SIGMOID_MU = 1e-2

# MATCHING
MATCHING_SCORE_GENUINE_THRESHOLD = 0.55
"""Must be in the range [0.0, 1.0]"""

class MatchingAlgorithm(IntEnum):
    LocalStructures = 0
    Hough = 1

MATCHING_ALGORITHM = MatchingAlgorithm.LocalStructures
LOCAL_STRUCTURES_MATCHING_MINUTIAE_PAIR_COUNT = 20
HOUGH_MATCHING_MINUTIAE_PIXELS_DISTANCE_THRESHOLD = 15
HOUGH_MATCHING_MINUTIAE_ANGLE_DEGREES_DISTANCE_THRESHOLD = 15
"""Must be in the range [0, 180]"""
