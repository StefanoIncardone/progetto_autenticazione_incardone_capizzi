from __future__ import annotations
from abc import ABC
from enum import IntEnum
from collections.abc import Generator
from typing import NamedTuple, Self, Any, TypeGuard, cast
import typing
import cv2
import math
import numpy
from numpy import (
    uint8 as u8,

    uint16 as u16,

    uint32 as u32,
    int32 as i32,

    int64 as i64,

    float32 as f32,
    float64 as f64,
    floating,
)
from numpy.typing import NDArray
from config import (
    BINARIZATION_THRESHOLD,
    DIRECTIONAL_MAP_BLOCK_LENGTH,
    DIRECTIONAL_MAP_BLUR_FILTER_LENGTH,
    GABOR_FILTERS_COUNT,
    GABOR_FILTERS_GAMMA,
    GABOR_FILTERS_SIGMA,
    GRADIENT_MODULE_BLOCK_LENGTH,
    GRADIENT_SOBEL_FILTER_LENGTH,
    LOCAL_RIDGE_BLOCK_COLUMNS,
    LOCAL_RIDGE_BLOCK_ROWS,
    MATCHING_ALGORITHM,
    MATCHING_SCORE_GENUINE_THRESHOLD,
    MCC_CIRCLES_RADIUS,
    MCC_GAUSSIAN_STD,
    MCC_SIGMOID_MU,
    MCC_SIGMOID_TAU,
    MCC_TOTAL_RADIUS,
    LOCAL_STRUCTURES_MATCHING_MINUTIAE_PAIR_COUNT,
    HOUGH_MATCHING_MINUTIAE_PIXELS_DISTANCE_THRESHOLD,
    HOUGH_MATCHING_MINUTIAE_ANGLE_DEGREES_DISTANCE_THRESHOLD,
    MINUTIAE_MIN_DISTANCE_FROM_BORDER,
    MINUTIAE_FOLLOWED_LENGTH_MAX,
    MINUTIAE_FOLLOWED_LENGTH_MIN,
    SEGMENTATION_MASK_THRESHOLD_SCALE,
    SINGULARITIES_MIN_DISTANCE_FROM_BORDER,
    MatchingAlgorithm,
)

def assert_type[T](expected_type: type[T], value: Any) -> TypeGuard[T]:
    expected_type_origin: type | None = typing.get_origin(value)
    if expected_type_origin is not None:
        expected_type = expected_type_origin

    value_type: type = type(value)
    if value_type is not expected_type:
        raise TypeError(f"actual type of '{value_type.__name__}' doesn't match expected type of '{expected_type.__name__}'")
    return True

class Range[T: int | float](NamedTuple):
    min: T
    max: T
    step: T
    value: T

    @staticmethod
    def new(min: T, max: T, step: T, value: T) -> Range[T]:
        if min > max:
            raise ValueError(f"{min = } cannot be greater than {max = }")
        if value < min or value > max:
            raise ValueError(f"{value = } must be in the range {min}..{max}")
        return Range(min, max, step, value)

class Bounds[T: int | float](NamedTuple):
    min: T
    max: T
    low: T
    high: T
    step: T

    @staticmethod
    def new(min: T, max: T, low: T, high: T, step: T) -> Bounds[T]:
        if min > max:
            raise ValueError(f"{min = } cannot be greater than {max = }")
        if low < min:
            raise ValueError(f"{low = } cannot be smaller than {min = }")
        if high > max:
            raise ValueError(f"{high = } cannot be greatex than {max = }")
        return Bounds(min, max, low, high, step)

class FeaturesExtractionConfig:
    __slots__ = (
        "gradient_sobel_filter_length",
        "gradient_module_block_length",
        "segmentation_mask_threshold_scale",

        "directional_map_block_length",
        "directional_map_blur_filter_length",

        "local_ridge_block_rows",
        "local_ridge_block_columns",
        "gabor_filters_count",
        "gabor_filters_sigma",
        "gabor_filters_gamma",
        "binarization_threshold",

        "singularities_min_distance_from_border",

        "minutiae_min_distance_from_border",
        "minutiae_followed_length",

        "mcc_total_radius",
        "mcc_circles_radius",
        "mcc_gaussian_std",
        "mcc_sigmoid_tau",
        "mcc_sigmoid_mu",
    )

    gradient_sobel_filter_length: Range[int]
    gradient_module_block_length: Range[int]
    segmentation_mask_threshold_scale: Range[float]

    directional_map_block_length: Range[int]
    directional_map_blur_filter_length: Range[int]

    local_ridge_block_rows: Range[int]
    local_ridge_block_columns: Range[int]
    gabor_filters_count: Range[int]
    gabor_filters_sigma: float
    gabor_filters_gamma: float
    binarization_threshold: Range[int]

    singularities_min_distance_from_border: Range[int]

    minutiae_min_distance_from_border: Range[int]
    minutiae_followed_length: Bounds[int]

    mcc_total_radius: int
    mcc_circles_radius: int
    mcc_gaussian_std: float
    mcc_sigmoid_tau: float
    mcc_sigmoid_mu: float

    def __init__( # type: ignore
        self,
        *,
        gradient_sobel_filter_length: Any = GRADIENT_SOBEL_FILTER_LENGTH,
        gradient_module_block_length: Any = GRADIENT_MODULE_BLOCK_LENGTH,
        segmentation_mask_threshold_scale: Any = SEGMENTATION_MASK_THRESHOLD_SCALE,
        directional_map_block_length: Any = DIRECTIONAL_MAP_BLOCK_LENGTH,
        directional_map_blur_filter_length: Any = DIRECTIONAL_MAP_BLUR_FILTER_LENGTH,
        local_ridge_block_rows: Any = LOCAL_RIDGE_BLOCK_ROWS,
        local_ridge_block_columns: Any = LOCAL_RIDGE_BLOCK_COLUMNS,
        gabor_filters_count: Any = GABOR_FILTERS_COUNT,
        gabor_filters_sigma: Any = GABOR_FILTERS_SIGMA,
        gabor_filters_gamma: Any = GABOR_FILTERS_GAMMA,
        binarization_threshold: Any = BINARIZATION_THRESHOLD,
        singularities_min_distance_from_border: Any = SINGULARITIES_MIN_DISTANCE_FROM_BORDER,
        minutiae_min_distance_from_border: Any = MINUTIAE_MIN_DISTANCE_FROM_BORDER,
        minutiae_followed_length_min: Any = MINUTIAE_FOLLOWED_LENGTH_MIN,
        minutiae_followed_length_max: Any = MINUTIAE_FOLLOWED_LENGTH_MAX,
        mcc_total_radius: Any = MCC_TOTAL_RADIUS,
        mcc_circles_radius: Any = MCC_CIRCLES_RADIUS,
        mcc_gaussian_std: Any = MCC_GAUSSIAN_STD,
        mcc_sigmoid_tau: Any = MCC_SIGMOID_TAU,
        mcc_sigmoid_mu: Any = MCC_SIGMOID_MU,
    ) -> None:
        _ = assert_type(int, gradient_sobel_filter_length)
        _ = assert_type(int, gradient_module_block_length)
        _ = assert_type(float, segmentation_mask_threshold_scale)
        _ = assert_type(int, directional_map_block_length)
        _ = assert_type(int, directional_map_blur_filter_length)
        _ = assert_type(int, local_ridge_block_rows)
        _ = assert_type(int, local_ridge_block_columns)
        _ = assert_type(int, gabor_filters_count)
        _ = assert_type(float, gabor_filters_sigma)
        _ = assert_type(float, gabor_filters_gamma)
        _ = assert_type(int, binarization_threshold)
        _ = assert_type(int, singularities_min_distance_from_border)
        _ = assert_type(int, minutiae_min_distance_from_border)
        _ = assert_type(int, minutiae_followed_length_min)
        _ = assert_type(int, minutiae_followed_length_max)
        _ = assert_type(int, mcc_total_radius)
        _ = assert_type(int, mcc_circles_radius)
        _ = assert_type(float, mcc_gaussian_std)
        _ = assert_type(float, mcc_sigmoid_tau)
        _ = assert_type(float, mcc_sigmoid_mu)

        if gradient_sobel_filter_length & 1 == 0:
            raise ValueError("gradient_sobel_filter_length must be odd")
        if directional_map_blur_filter_length & 1 == 0:
            raise ValueError("directional_map_blur_filter_length must be odd")
        if local_ridge_block_rows % 8 != 0:
            raise ValueError("local_ridge_block_rows must be a multiple of 8")
        if local_ridge_block_columns % 8 != 0:
            raise ValueError("local_ridge_block_columns must be a multiple of 8")

        self.gradient_sobel_filter_length = Range[int].new(
            min = 1,
            max = 31,
            step = 2,
            value = gradient_sobel_filter_length,
        )
        self.gradient_module_block_length = Range[int].new(
            min = 1,
            max = 31,
            step = 1,
            value = gradient_module_block_length,
        )
        self.segmentation_mask_threshold_scale = Range[float].new(
            min = 0.0,
            max = 1.0,
            step = 0.01,
            value = segmentation_mask_threshold_scale
        )
        self.directional_map_block_length = Range[int].new(
            min = 1,
            max = 31,
            step = 1,
            value = directional_map_block_length,
        )
        self.directional_map_blur_filter_length = Range[int].new(
            min = -1,
            max = 31,
            step = 2,
            value = directional_map_blur_filter_length,
        )

        self.local_ridge_block_rows = Range[int].new(
            min = 8,
            max = 256,
            step = 8,
            value = local_ridge_block_rows,
        )
        self.local_ridge_block_columns = Range[int].new(
            min = 8,
            max = 256,
            step = 8,
            value = local_ridge_block_columns,
        )
        self.gabor_filters_count = Range[int].new(
            min = 1,
            max = 16,
            step = 1,
            value = gabor_filters_count,
        )
        self.gabor_filters_sigma = gabor_filters_sigma
        self.gabor_filters_gamma = gabor_filters_gamma
        self.binarization_threshold = Range[int].new(
            min = 0,
            max = 255,
            step = 1,
            value = binarization_threshold,
        )

        self.singularities_min_distance_from_border = Range[int].new(
            min = 0,
            max = 255,
            step = 1,
            value = singularities_min_distance_from_border,
        )

        self.minutiae_min_distance_from_border = Range[int].new(
            min = 0,
            max = 255,
            step = 1,
            value = minutiae_min_distance_from_border,
        )
        self.minutiae_followed_length = Bounds[int].new(
            min = 0,
            max = 64,
            low = minutiae_followed_length_min,
            high = minutiae_followed_length_max,
            step = 1,
        )

        self.mcc_total_radius = mcc_total_radius
        self.mcc_circles_radius = mcc_circles_radius
        self.mcc_gaussian_std = mcc_gaussian_std
        self.mcc_sigmoid_tau = mcc_sigmoid_tau
        self.mcc_sigmoid_mu = mcc_sigmoid_mu

class FeaturesMatchingConfig:
    __slots__ = (
        "matching_score_genuine_threshold",
        "matching_algorithm",
        "local_structures_matching_minutiae_pair_count",
        "hough_matching_minutiae_pixels_distance_threshold",
        "hough_matching_minutiae_angle_degrees_distance_threshold",
    )

    matching_score_genuine_threshold: Range[float]
    matching_algorithm: MatchingAlgorithm
    local_structures_matching_minutiae_pair_count: int
    hough_matching_minutiae_pixels_distance_threshold: int
    hough_matching_minutiae_angle_degrees_distance_threshold: Range[int]

    def __init__( # type: ignore
        self,
        *,
        matching_score_genuine_threshold: Any = MATCHING_SCORE_GENUINE_THRESHOLD,
        matching_algorithm: Any = MATCHING_ALGORITHM,
        local_structures_matching_minutiae_pair_count: Any = LOCAL_STRUCTURES_MATCHING_MINUTIAE_PAIR_COUNT,
        hough_matching_minutiae_pixels_distance_threshold: Any = HOUGH_MATCHING_MINUTIAE_PIXELS_DISTANCE_THRESHOLD,
        hough_matching_minutiae_angle_degrees_distance_threshold: Any = HOUGH_MATCHING_MINUTIAE_ANGLE_DEGREES_DISTANCE_THRESHOLD,
    ) -> None:
        _ = assert_type(float, matching_score_genuine_threshold)
        _ = assert_type(MatchingAlgorithm, matching_algorithm)
        _ = assert_type(int, local_structures_matching_minutiae_pair_count)
        _ = assert_type(int, hough_matching_minutiae_pixels_distance_threshold)
        _ = assert_type(int, hough_matching_minutiae_angle_degrees_distance_threshold)

        if local_structures_matching_minutiae_pair_count < 0:
            raise ValueError("local_structures_matching_minutiae_pair_count must be positive")

        if hough_matching_minutiae_pixels_distance_threshold < 0:
            raise ValueError("hough_matching_minutiae_pixels_distance_threshold must be positive")

        self.matching_score_genuine_threshold = Range[float].new(
            min = 0.0,
            max = 1.0,
            step = 0.01,
            value = matching_score_genuine_threshold,
        )
        self.matching_algorithm = matching_algorithm
        self.local_structures_matching_minutiae_pair_count = local_structures_matching_minutiae_pair_count
        self.hough_matching_minutiae_pixels_distance_threshold = hough_matching_minutiae_pixels_distance_threshold
        self.hough_matching_minutiae_angle_degrees_distance_threshold = Range[int].new(
            min = 0,
            max = 31,
            step = 180,
            value = hough_matching_minutiae_angle_degrees_distance_threshold,
        )

FE_CONFIG = FeaturesExtractionConfig()
FM_CONFIG = FeaturesMatchingConfig()

# UTILS
PI = numpy.pi
PI_DIV_2 = PI / 2

U8_BITS = numpy.iinfo(u8).bits
U8_MIN = numpy.iinfo(u8).min
U8_MAX = numpy.iinfo(u8).max

class Point(NamedTuple):
    column: int
    row: int

    def with_angle(self, angle: float) -> PointWithAngle:
        return PointWithAngle(column = self.column, row = self.row, angle = angle)

class PointWithAngle(NamedTuple):
    column: int
    row: int
    angle: float

def is_close(row_0: int, column_0: int, row_1: int, column_1: int, radius: int) -> bool:
    return (column_0 - column_1) ** 2 + (row_0 - row_1) ** 2 <= (radius ** 2)

def angles_abs_difference(angle_0: float, angle_1: float) -> float:
    angles_abs_difference = abs(angle_0 - angle_1)
    return min(angles_abs_difference, 2 * PI - angles_abs_difference)

class SingualirtyPoincareIndex(IntEnum):
    No = 0
    Core = 1
    Delta = -1
    Whorl = 2

class Singularity(ABC, Point): pass
class Core(Singularity): pass
class Delta(Singularity): pass
class Whorl(Singularity): pass

class MinutiaCrossingNumber(IntEnum):
    No = 0
    Termination = 1
    Continuation = 2
    Bifurcation = 3
    Unknown = 4

class Minutia(ABC, Point): pass
class Termination(Minutia): pass
class Bifurcation(Minutia): pass

class MinutiaWithAngle(ABC, PointWithAngle): pass
class TerminationWithAngle(MinutiaWithAngle): pass
class BifurcationWithAngle(MinutiaWithAngle): pass

class NeighborXOffset(IntEnum):
    Left = -1
    Center = 0
    Right = 1

class NeighborYOffset(IntEnum):
    Bottom = -1
    Center = 0
    Top = 1

class Neighbor(NamedTuple):
    row_offset: NeighborYOffset
    column_offset: NeighborXOffset
    distance_from_center: float

    @staticmethod
    def new(vertical_spot: NeighborYOffset, horizontal_spot: NeighborXOffset) -> Neighbor:
        distance = (horizontal_spot ** 2 + vertical_spot ** 2) ** 0.5
        return Neighbor(vertical_spot, horizontal_spot, distance)

def get_neighbor_direction(
    directional_map: NDArray[f32],
    block_length: int,
    block_row: int,
    block_column: int,
    neighbor: Neighbor,
) -> f32:
    neighbor_row = block_row + neighbor.row_offset * block_length
    neighbor_column = block_column + neighbor.column_offset * block_length
    neighbor_direction: f32 = directional_map[neighbor_row, neighbor_column]
    return neighbor_direction

def compute_direction_difference(next_neighbor_direction: f32, neighbor_direction: f32) -> f32:
    direction_difference = next_neighbor_direction - neighbor_direction
    if direction_difference > (PI_DIV_2):
        direction_difference = f32(PI) - direction_difference
    elif direction_difference < (-PI_DIV_2):
        direction_difference = f32(PI) + direction_difference
    else:
        direction_difference = -direction_difference
    return direction_difference

def compute_crossing_number(neighbors: NDArray[u8]) -> int:
    return numpy.count_nonzero(neighbors < numpy.roll(neighbors, shift = -1))

def bits(number: int, bits_to_extract: int) -> Generator[int]:
    assert bits_to_extract > 0, "At least one bit needs to be extracted"
    while bits_to_extract > 0:
        bit = number & 1
        yield bit
        number >>= 1
        bits_to_extract -= 1

NEIGHBORS = [
    Neighbor.new(NeighborYOffset.Bottom, NeighborXOffset.Left),
    Neighbor.new(NeighborYOffset.Bottom, NeighborXOffset.Center),
    Neighbor.new(NeighborYOffset.Bottom, NeighborXOffset.Right),
    Neighbor.new(NeighborYOffset.Center, NeighborXOffset.Right),
    Neighbor.new(NeighborYOffset.Top, NeighborXOffset.Right),
    Neighbor.new(NeighborYOffset.Top, NeighborXOffset.Center),
    Neighbor.new(NeighborYOffset.Top, NeighborXOffset.Left),
    Neighbor.new(NeighborYOffset.Center, NeighborXOffset.Left),
]
CENTER_NEIGHBORS_BIT_FIELD = [numpy.fromiter(bits(number, U8_BITS), dtype = u8) for number in range(U8_MAX + 1)]

CROSSING_NUMBER_LUT = numpy.fromiter((compute_crossing_number(neighbor) for neighbor in CENTER_NEIGHBORS_BIT_FIELD), dtype = u8)
CROSSING_NUMBER_FILTER = numpy.array([
    [1,   2,  4],
    [128, 0,  8],
    [64,  32, 16],
], dtype = u8)

def compute_next_ridge_following_direction(previous_direction: int, neighbors: NDArray[u8]) -> list[int]:
    next_positions: list[int] = numpy.argwhere(neighbors != 0).ravel().tolist()
    if len(next_positions) > 0 and previous_direction != 8:
        # There is a previous direction:
        # return all the next directions, sorted according to the distance from it,
        # except the direction, if any, that corresponds to the previous position
        next_positions.sort(key = lambda direction: 4 - abs(abs(direction - previous_direction) - 4))

        # the direction of the previous position is the opposite one
        if next_positions[-1] == (previous_direction + 4) % 8:
            _ = next_positions.pop()
    return next_positions

DIRECTIONS_DISTANCES_LUT = [[compute_next_ridge_following_direction(direction, neighbors) for direction in range(len(NEIGHBORS) + 1)] for neighbors in CENTER_NEIGHBORS_BIT_FIELD]

def follow_ridge_and_compute_angle(
    minutia_row: int,
    minutia_column: int,
    direction: int,
    neighbors_counts: NDArray[u8],
    followed_length_min: int,
    followed_length_max: int,
) -> float | None:
    start_row, start_column = minutia_row, minutia_column
    followed_length = 0.0
    while followed_length < followed_length_max:
        neighbor_value: u8 = neighbors_counts[start_row, start_column]
        next_directions = DIRECTIONS_DISTANCES_LUT[neighbor_value][direction]
        if len(next_directions) == 0:
            break

        found_another_minutia = False
        for next_direction in next_directions:
            neighbor_offset = NEIGHBORS[next_direction]
            next_direction_neighbors_count: u8 = neighbors_counts[
                start_row + neighbor_offset.row_offset,
                start_column + neighbor_offset.column_offset
            ]
            next_direction_crossing_number: u8 = CROSSING_NUMBER_LUT[next_direction_neighbors_count]
            if (next_direction_crossing_number == MinutiaCrossingNumber.Termination
                or next_direction_crossing_number == MinutiaCrossingNumber.Bifurcation):
                found_another_minutia = True
        if found_another_minutia:
            break

        # Only the first direction has to be followed
        direction = next_directions[0]
        neighbor_offset = NEIGHBORS[direction]
        start_row += neighbor_offset.row_offset
        start_column += neighbor_offset.column_offset
        followed_length += neighbor_offset.distance_from_center

    if followed_length >= followed_length_min:
        return math.atan2(-start_row + minutia_row, start_column - minutia_column)
    else:
        return None

def Mcc_Gaussian[XS: floating[Any]](xs: NDArray[XS], std: float) -> NDArray[XS]:
    """Gaussian function with mean = 0"""
    return numpy.exp(-0.5 * (xs / std) ** 2) / (std * math.tau ** 0.5)

def Mcc_Sigmoid[XS: floating[Any]](xs: NDArray[XS], tau: float, mu: float) -> NDArray[XS]:
    """Sigmoid function with result in the range [0, 1]"""
    return cast(NDArray[XS], 1.0 / (1.0 + numpy.exp(-tau * (xs - mu))))

class Singularities:
    __slots__ = (
        "all",
        "filtered",
    )

    all: list[Singularity]
    filtered: list[Singularity]

    def __init__( # type: ignore
        self,
        directional_map: NDArray[f32],
        directional_map_block_length: int,
        segmentation_mask_distance_map: NDArray[f32],
        min_distance_from_border: int,
    ) -> None:
        self.all = []

        fingerprint_rows, fingerprint_columns = directional_map.shape
        half_directional_map_block_length = directional_map_block_length // 2
        block_start = directional_map_block_length + half_directional_map_block_length - 1
        block_row_end = fingerprint_rows - directional_map_block_length
        block_column_end = fingerprint_columns - directional_map_block_length
        for block_row in range(block_start, block_row_end, directional_map_block_length):
            for block_column in range(block_start, block_column_end, directional_map_block_length):
                total_direction_difference = 0.0
                neighbors = iter(NEIGHBORS)
                neighbor_direction = get_neighbor_direction(
                    directional_map,
                    directional_map_block_length,
                    block_row,
                    block_column,
                    next(neighbors)
                )
                last_neighbor_direction = neighbor_direction
                for next_neighbor in neighbors:
                    next_neighbor_direction = get_neighbor_direction(
                        directional_map,
                        directional_map_block_length,
                        block_row,
                        block_column,
                        next_neighbor
                    )
                    total_direction_difference += compute_direction_difference(next_neighbor_direction, neighbor_direction)
                    neighbor_direction = next_neighbor_direction
                total_direction_difference += compute_direction_difference(last_neighbor_direction, neighbor_direction)

                poincare_index = int(round(total_direction_difference / PI))
                singularity: Singularity
                match poincare_index:
                    case SingualirtyPoincareIndex.Core:
                        singularity = Core(block_column, block_row)
                    case SingualirtyPoincareIndex.Delta:
                        singularity = Delta(block_column, block_row)
                    case SingualirtyPoincareIndex.Whorl:
                        singularity = Whorl(block_column, block_row)
                    case _: continue
                self.all.append(singularity)

        self.filtered = []
        for singularity in self.all:
            if segmentation_mask_distance_map[singularity.row, singularity.column] <= min_distance_from_border:
                continue
            self.filtered.append(singularity)

class Minutiae:
    __slots__ = (
        "all",
        "filtered",
    )

    all: list[Minutia]
    filtered: list[Minutia]

    def __init__( # type: ignore
        self,
        thinned_fingerprint: NDArray[u8],
        segmentation_mask_distance_map: NDArray[f32],
        min_distance_from_border: int,
    ) -> None:
        thinned_binarized = numpy.where(thinned_fingerprint != 0, 1, 0).astype(u8)
        neighbors_values: NDArray[u8] = cv2.filter2D(
            thinned_binarized,
            ddepth = -1,
            kernel = CROSSING_NUMBER_FILTER,
            borderType = cv2.BORDER_CONSTANT
        ) # type: ignore

        crossing_numbers: NDArray[u8] = cv2.LUT(neighbors_values, CROSSING_NUMBER_LUT) # type: ignore

        fingerprint_rows, fingerprint_columns = thinned_fingerprint.shape
        self.all = []
        for minutia_row in range(fingerprint_rows):
            for minutia_column in range(fingerprint_columns):
                if thinned_fingerprint[minutia_row, minutia_column] == 0:
                    continue

                minutia: Minutia
                minutia_crossing_number: u8 = crossing_numbers[minutia_row, minutia_column]
                match minutia_crossing_number:
                    case MinutiaCrossingNumber.Termination: minutia = Termination(
                        row = minutia_row,
                        column = minutia_column,
                    )
                    case MinutiaCrossingNumber.Bifurcation: minutia = Bifurcation(
                        row = minutia_row,
                        column = minutia_column,
                    )
                    case _: continue

                self.all.append(minutia)

        self.filtered = []
        for minutia in self.all:
            if segmentation_mask_distance_map[minutia.row, minutia.column] <= min_distance_from_border:
                continue
            self.filtered.append(minutia)

def normalize_pixels(pixels: NDArray[u8]) -> NDArray[u8]:
    pixels_min: u8 = pixels.min()
    pixels_max: u8 = pixels.max()

    # normalized_pixels = (pixels - pixels_min) * (U8_MAX - U8_MIN) / (pixels_max - pixels_min) + U8_MIN
    # normalized_pixels = (pixels - pixels_min) * (255 - 0) / (pixels_max - pixels_min) + 0
    # normalized_pixels = (pixels - pixels_min) * 255 / (pixels_max - pixels_min)
    return ((pixels - pixels_min).astype(u16) * U8_MAX // (pixels_max - pixels_min)).astype(u8)

class FingerprintAcquisition:
    __slots__ = (
        "tag",
        "features",
    )

    tag: str
    features: FingerprintFeatures

    def __init__( # type: ignore
        self,
        tag: str,
        features: FingerprintFeatures,
    ) -> None:
        self.tag = tag
        self.features = features

class FingerprintFeatures:
    __slots__ = (
        "singularities",
        "minutiae",
        "mcc_reference_cell_coordinates",
        "local_structures"
    )

    singularities: list[Singularity]
    minutiae: list[MinutiaWithAngle]
    mcc_reference_cell_coordinates: MccReferenceCellCoordinates
    local_structures: NDArray[f32]

    def __init__( # type: ignore
        self,
        singularities: list[Singularity],
        minutiae: list[MinutiaWithAngle],
        mcc_reference_cell_coordinates: MccReferenceCellCoordinates,
        local_structures: NDArray[f32]
    ) -> None:
        self.singularities = singularities
        self.minutiae = minutiae
        self.local_structures = local_structures
        self.mcc_reference_cell_coordinates = mcc_reference_cell_coordinates

    def matching_score_local_structures(self, other: Self, mathing_pair_count: int) -> float:
        distances: NDArray[f32] = numpy.linalg.norm(
            self.local_structures[:, numpy.newaxis,:] - other.local_structures,
            axis = -1
        )
        distances /= numpy.linalg.norm(self.local_structures, axis = 1)[:, numpy.newaxis] + numpy.linalg.norm(other.local_structures, axis = 1)
        minutiae_matching_pairs: tuple[NDArray[i64], NDArray[i64]] = numpy.unravel_index(
            numpy.argpartition(distances, mathing_pair_count)[: mathing_pair_count],
            distances.shape
        ) # type: ignore
        matching_score = float(1 - numpy.mean(distances[minutiae_matching_pairs[0], minutiae_matching_pairs[1]]))
        return matching_score

    def matching_score_hough(
        self,
        other: Self,
        pixels_distance_threshold: int,
        # angles_distance_threshold: int,
    ) -> float:
        # def is_aligned(angle_0: float, angle_1: float, angles_distance_threshold: int) -> bool:
        #     threshold = math.radians(angles_distance_threshold)
        #     return angles_abs_difference(angle_0, angle_1) <= threshold

        accumulator: dict[PointWithAngle, int] = {}

        for other_minutia in other.minutiae:
            for self_minutia in self.minutiae:
                angles_difference = angles_abs_difference(self_minutia.angle, other_minutia.angle)
                angles_difference_cos = math.cos(angles_difference)
                angles_difference_sin = math.sin(angles_difference)

                translation_column = other_minutia.column - self_minutia.column * angles_difference_cos + self_minutia.row * angles_difference_sin
                translation_row = other_minutia.row - self_minutia.column * angles_difference_sin - self_minutia.row * angles_difference_cos

                translation_point = PointWithAngle(
                    column = int(round(translation_column)),
                    row = int(round(translation_row)),
                    angle = int(round(angles_difference))
                )
                if translation_point in accumulator:
                    accumulator[translation_point] += 1
                else:
                    accumulator[translation_point] = 1

        most_voted_translation: PointWithAngle = ... # type: ignore
        max_votes = 0
        for translation, votes in accumulator.items():
            if votes > max_votes:
                most_voted_translation = translation

        rotation_angle_cos = math.cos(most_voted_translation.angle)
        rotation_angle_sin = math.sin(most_voted_translation.angle)
        rotation_matrix: NDArray[f64] = numpy.array([
            [rotation_angle_cos, -rotation_angle_sin],
            [rotation_angle_sin, rotation_angle_cos],
        ])

        self_aligned_minutiae = (
            numpy.array([(minutia.column, minutia.row) for minutia in self.minutiae], dtype = i64)
            @ rotation_matrix + numpy.array([most_voted_translation.column, most_voted_translation.row], dtype = i64)
        ).round().astype(i64)
        self_aligned_minutiae = [
            Point(column = column, row = row) for column, row in self_aligned_minutiae
        ]

        matching_minutiae_count = 0
        for other_minutia in other.minutiae:
            for self_minutia in self.minutiae:
                # aligned_angle = other_minutia.angle - self_minutia.angle - most_voted_translation.angle
                # aligned_angle = min(aligned_angle, 360 - aligned_angle)
                self_aligned_minutia_column = other_minutia.column - self_minutia.column * rotation_angle_cos + self_minutia.row * rotation_angle_sin - most_voted_translation.column
                self_aligned_minutia_row = other_minutia.row - self_minutia.column * rotation_angle_sin - self_minutia.row * rotation_angle_cos - most_voted_translation.row
                self_aligned_minutia_column = int(round(self_aligned_minutia_column))
                self_aligned_minutia_row = int(round(self_aligned_minutia_row))

                minutiae_are_close = is_close(
                    self_aligned_minutia_row,
                    self_aligned_minutia_column,
                    other_minutia.row,
                    other_minutia.column,
                    pixels_distance_threshold,
                )
                # minutiae_are_aligned = is_aligned(
                #     aligned_angle,
                #     other_minutia.angle,
                #     angles_distance_threshold
                # )

                # if minutiae_are_close and minutiae_are_aligned:
                if minutiae_are_close:
                    matching_minutiae_count += 1
                    break

        matching_score = matching_minutiae_count / max(len(self_aligned_minutiae), len(other.minutiae))
        return matching_score

class MccReferenceCellCoordinates:
    __slots__ = (
        "coordinates"
    )

    coordinates: NDArray[f32]

    def __init__( # type: ignore
        self,
        total_radius: int,
        circles_radius: int,
    ) -> None:
        g = 2 * total_radius / circles_radius
        x = cast(NDArray[f32], numpy.arange(
            circles_radius, dtype = f32
        ) * g - total_radius + total_radius / circles_radius)
        y = x[..., numpy.newaxis]
        y_index, x_index = numpy.nonzero((x ** 2 + y ** 2) <= (total_radius ** 2))
        self.coordinates = numpy.column_stack((x[x_index], y[y_index]))

class Fingerprint:
    __slots__ = (
        "raw_fingerprint",
        "normalized_fingerprint",
        "normalized_negative_fingerprint",
        "enhanced_fingerprint",
        "binarized_fingerprint",
        "thinned_fingerprint",

        "gradient_x",
        "gradient_y",
        "gradient_x2",
        "gradient_y2",
        "gradient_module",
        "gradient_x2_filtered",
        "gradient_y2_filtered",
        "gradient_xy_filtered",
        "gradient_x2_minus_y2_filtered",
        "gradient_2xy_filtered",

        "segmentation_mask",
        "segmentation_mask_distance_map",

        "directional_map",

        "ridge_block",
        "ridge_block_row_start",
        "ridge_block_row_end",
        "ridge_block_column_start",
        "ridge_block_column_end",
        "ridge_frequency",

        "gabor_filters",
        "gabor_filters_angles",
        "fingerprint_with_gabor_filters",

        "acquisition",
    )

    raw_fingerprint: NDArray[u8]
    normalized_fingerprint: NDArray[u8]
    normalized_negative_fingerprint: NDArray[u8]
    enhanced_fingerprint: NDArray[u8]
    binarized_fingerprint: NDArray[u8]
    thinned_fingerprint: NDArray[u8]

    gradient_x: NDArray[f32]
    gradient_y: NDArray[f32]
    gradient_x2: NDArray[f32]
    gradient_y2: NDArray[f32]
    gradient_module: NDArray[f32]
    gradient_x2_filtered: NDArray[f32]
    gradient_y2_filtered: NDArray[f32]
    gradient_xy_filtered: NDArray[f32]
    gradient_x2_minus_y2_filtered: NDArray[f32]
    gradient_2xy_filtered: NDArray[f32]

    segmentation_mask: NDArray[u8]
    segmentation_mask_distance_map: NDArray[f32]

    directional_map: NDArray[f32]

    ridge_block: NDArray[u8]
    ridge_block_row_start: int
    ridge_block_row_end: int
    ridge_block_column_start: int
    ridge_block_column_end: int
    ridge_frequency: float

    gabor_filters: list[NDArray[f64]]
    gabor_filters_angles: list[float]
    fingerprint_with_gabor_filters: list[NDArray[u8]]

    acquisition: FingerprintAcquisition

    def __init__( # type: ignore
        self,
        file_path: str,
        acquisition_tag: str,
        *,
        gradient_sobel_filter_length: int,
        gradient_module_block_length: int,
        segmentation_mask_threshold_scale: float,
        directional_map_block_length: int,
        directional_map_blur_filter_length: int,
        local_ridge_block_rows: int,
        local_ridge_block_columns: int,
        gabor_filters_count: int,
        gabor_filters_sigma: float,
        gabor_filters_gamma: float,
        binarization_threshold: int,
        singularities_min_distance_from_border: int,
        minutiae_min_distance_from_border: float,
        minutiae_followed_length_min: int,
        minutiae_followed_length_max: int,
        mcc_reference_cell_coordinates: MccReferenceCellCoordinates | None,
        mcc_gaussian_std: float,
        mcc_sigmoid_tau: float,
        mcc_sigmoid_mu: float,
    ) -> None:
        self.raw_fingerprint: NDArray[u8] = cv2.imread(file_path, flags = cv2.IMREAD_GRAYSCALE) # type: ignore
        self.normalized_fingerprint = normalize_pixels(self.raw_fingerprint)
        self.normalized_negative_fingerprint: NDArray[u8] = U8_MAX - self.normalized_fingerprint # type: ignore

        # GRADIENTS
        self.gradient_x = cv2.Sobel(
            self.normalized_fingerprint,
            ddepth = cv2.CV_32F,
            dx = 1,
            dy = 0,
            ksize = gradient_sobel_filter_length
        ) # type: ignore
        self.gradient_y = cv2.Sobel(
            self.normalized_fingerprint,
            ddepth = cv2.CV_32F,
            dx = 0,
            dy = 1,
            ksize = gradient_sobel_filter_length
        ) # type: ignore
        self.gradient_x2: NDArray[f32] = self.gradient_x * self.gradient_x
        self.gradient_y2: NDArray[f32] = self.gradient_y * self.gradient_y
        self.gradient_module = numpy.sqrt(self.gradient_x2 + self.gradient_y2)
        self.gradient_module = cv2.boxFilter(
            numpy.sqrt(self.gradient_x2 + self.gradient_y2),
            ddepth = -1,
            ksize = (gradient_module_block_length, gradient_module_block_length),
            normalize = False,
        ) # type: ignore

        directional_map_block_size = (directional_map_block_length, directional_map_block_length)
        self.gradient_x2_filtered = cv2.boxFilter(
            self.gradient_x2,
            ddepth = -1,
            ksize = directional_map_block_size,
            normalize = False,
        ) # type: ignore
        self.gradient_y2_filtered = cv2.boxFilter(
            self.gradient_y2,
            ddepth = -1,
            ksize = directional_map_block_size,
            normalize = False,
        ) # type: ignore
        self.gradient_xy_filtered = cv2.boxFilter(
            self.gradient_x * self.gradient_y,
            ddepth = -1,
            ksize = directional_map_block_size,
            normalize = False,
        ) # type: ignore
        self.gradient_x2_minus_y2_filtered: NDArray[f32] = self.gradient_x2_filtered - self.gradient_y2_filtered
        self.gradient_2xy_filtered: NDArray[f32] = self.gradient_xy_filtered + self.gradient_xy_filtered

        # SEGMENTATION MASK
        _segmentation_mask_threshold, segmentation_mask = cv2.threshold(
            self.gradient_module,
            thresh = self.gradient_module.max() * segmentation_mask_threshold_scale,
            maxval = U8_MAX,
            type = cv2.THRESH_BINARY,
        )
        self.segmentation_mask = segmentation_mask.astype(u8)
        segmentation_mask_bordered = cv2.copyMakeBorder(
            self.segmentation_mask,
            top = 1,
            bottom = 1,
            left = 1,
            right = 1,
            borderType = cv2.BORDER_CONSTANT,
        )
        self.segmentation_mask_distance_map = cv2.distanceTransform(
            segmentation_mask_bordered,
            distanceType = cv2.DIST_C,
            maskSize = 3,
        ) # type: ignore
        self.segmentation_mask_distance_map = self.segmentation_mask_distance_map[1: -1, 1: -1]

        # DIRECTIONAL MAP
        directional_map_phase = cv2.phase(
            self.gradient_x2_minus_y2_filtered,
            -self.gradient_2xy_filtered,
            angleInDegrees = False
        )
        self.directional_map: NDArray[f32] = (directional_map_phase + PI) / 2
        self.directional_map[self.directional_map > PI] -= PI

        if directional_map_blur_filter_length >= 1:
            self.directional_map = cv2.GaussianBlur(
                self.directional_map,
                ksize = (directional_map_blur_filter_length, directional_map_blur_filter_length),
                sigmaX = 0,
                # sigmaY = 0
            ) # type: ignore

        # FREQUENCY ESTIMATION AND ENHANCMENT USING GABOR FILTERS
        fingerprint_rows, fingerprint_columns = self.normalized_fingerprint.shape
        self.ridge_block_row_start = (fingerprint_rows - local_ridge_block_rows) // 2
        self.ridge_block_row_end = self.ridge_block_row_start + local_ridge_block_rows
        self.ridge_block_column_start = (fingerprint_columns - local_ridge_block_columns) // 2
        self.ridge_block_column_end = self.ridge_block_column_start + local_ridge_block_columns
        self.ridge_block = self.normalized_fingerprint[
            self.ridge_block_row_start: self.ridge_block_row_end,
            self.ridge_block_column_start: self.ridge_block_column_end,
        ]

        x_signature: NDArray[u32] = self.ridge_block.sum(axis = 1, dtype = u32)
        x_signature_pairs_next_is_greater_than_previous = numpy.r_[
            False, x_signature[1:] > x_signature[:-1]
        ]
        x_signature_pairs_previous_is_greater_or_equals_than_next = numpy.r_[
            x_signature[:-1] >= x_signature[1:], False
        ]
        x_signature_local_maxima_indexes, = numpy.nonzero(
            x_signature_pairs_next_is_greater_than_previous
            & x_signature_pairs_previous_is_greater_or_equals_than_next
        )

        distances_between_consecutive_ridges: NDArray[i64] = (
            x_signature_local_maxima_indexes[1:] - x_signature_local_maxima_indexes[:-1]
        )
        frequency: f64 = numpy.average(distances_between_consecutive_ridges) # type: ignore
        self.ridge_frequency: float = round(float(frequency), ndigits = 2)

        gabor_kernel_length = int(round(self.ridge_frequency * 2 + 1))
        one_if_gabor_kernel_length_is_even = 1 - gabor_kernel_length & 1
        gabor_kernel_length += one_if_gabor_kernel_length_is_even
        gabor_kernel_size = (gabor_kernel_length, gabor_kernel_length)
        gabor_filters_sigma *= self.ridge_frequency

        self.gabor_filters = []
        self.gabor_filters_angles = []
        self.fingerprint_with_gabor_filters = []
        for gabor_kernel_angle in numpy.arange(0, PI, PI / gabor_filters_count, dtype = f64):
            gabor_kernel: NDArray[f64] = cv2.getGaborKernel(
                ksize = gabor_kernel_size,
                sigma = gabor_filters_sigma,
                theta = PI_DIV_2 - gabor_kernel_angle,
                lambd = self.ridge_frequency,
                gamma = gabor_filters_gamma,
                psi = 0 # making sure the filter is symetrical
            ) # type: ignore
            # kernel /= kernel.sum()
            # kernel -= kernel.mean()

            fingerprint_with_gabor_filter: NDArray[u8] = cv2.filter2D(
                self.normalized_negative_fingerprint,
                ddepth = -1,
                kernel = gabor_kernel
            ) # type: ignore

            self.gabor_filters.append(gabor_kernel)
            self.gabor_filters_angles.append(gabor_kernel_angle)
            self.fingerprint_with_gabor_filters.append(fingerprint_with_gabor_filter)

        fingerprints_with_gabor_filters = numpy.array(self.fingerprint_with_gabor_filters)

        closest_gabor_filter_angle_indices: NDArray[u8] = (
            ((self.directional_map % PI) * gabor_filters_count / PI).round().astype(u8) % gabor_filters_count
        ) # type: ignore

        row_coordinates_indices: NDArray[i64]
        column_coordinates_indices: NDArray[i64]
        row_coordinates_indices, column_coordinates_indices = numpy.indices(self.normalized_fingerprint.shape, dtype = u16)
        enhanced_fingerprint = fingerprints_with_gabor_filters[
            closest_gabor_filter_angle_indices,
            row_coordinates_indices,
            column_coordinates_indices
        ]
        self.enhanced_fingerprint: NDArray[u8] = self.segmentation_mask & enhanced_fingerprint

        # BINARIZATION AND THINNING
        self.binarized_fingerprint: NDArray[u8]
        _binarized_fingerprint_threshold, self.binarized_fingerprint = cv2.threshold( # type: ignore
            self.enhanced_fingerprint,
            thresh = binarization_threshold,
            maxval = U8_MAX,
            type = cv2.THRESH_BINARY,
        )
        self.thinned_fingerprint: NDArray[u8] = cv2.ximgproc.thinning(
            self.binarized_fingerprint, thinningType = cv2.ximgproc.THINNING_GUOHALL
        ) # type: ignore

        # POINCARÈ INDEX AND SINGULARITIES
        singularities: list[Singularity] = []

        half_directional_map_block_length = directional_map_block_length // 2
        block_start = directional_map_block_length + half_directional_map_block_length - 1
        block_row_end = fingerprint_rows - directional_map_block_length
        block_column_end = fingerprint_columns - directional_map_block_length
        for block_row in range(block_start, block_row_end, directional_map_block_length):
            for block_column in range(block_start, block_column_end, directional_map_block_length):
                if self.segmentation_mask_distance_map[block_row, block_column] <= singularities_min_distance_from_border:
                    continue

                total_direction_difference = 0.0
                neighbors = iter(NEIGHBORS)
                neighbor_direction = get_neighbor_direction(
                    self.directional_map,
                    directional_map_block_length,
                    block_row,
                    block_column,
                    next(neighbors)
                )
                last_neighbor_direction = neighbor_direction
                for next_neighbor in neighbors:
                    next_neighbor_direction = get_neighbor_direction(
                        self.directional_map,
                        directional_map_block_length,
                        block_row,
                        block_column,
                        next_neighbor
                    )
                    total_direction_difference += compute_direction_difference(next_neighbor_direction, neighbor_direction)
                    neighbor_direction = next_neighbor_direction
                total_direction_difference += compute_direction_difference(last_neighbor_direction, neighbor_direction)

                poincare_index = int(round(total_direction_difference / PI))
                singularity: Singularity
                match poincare_index:
                    case SingualirtyPoincareIndex.Core:
                        singularity = Core(block_column, block_row)
                    case SingualirtyPoincareIndex.Delta:
                        singularity = Delta(block_column, block_row)
                    case SingualirtyPoincareIndex.Whorl:
                        singularity = Whorl(block_column, block_row)
                    case _: continue
                singularities.append(singularity)

        # CROSSING NUMBERS AND MINUTIAE EXTRACTION
        thinned_binarized = numpy.where(self.thinned_fingerprint != 0, 1, 0).astype(u8)
        neighbors_counts: NDArray[u8] = cv2.filter2D(
            thinned_binarized,
            ddepth = -1,
            kernel = CROSSING_NUMBER_FILTER,
            borderType = cv2.BORDER_CONSTANT
        ) # type: ignore

        minutiae: list[MinutiaWithAngle] = []
        for minutia_row in range(fingerprint_rows):
            for minutia_column in range(fingerprint_columns):
                if self.thinned_fingerprint[minutia_row, minutia_column] == 0:
                    continue
                if self.segmentation_mask_distance_map[minutia_row, minutia_column] <= minutiae_min_distance_from_border:
                    continue

                neighbors_count: u8 = neighbors_counts[
                    minutia_row,
                    minutia_column,
                ]
                minutia_crossing_number: u8 = CROSSING_NUMBER_LUT[neighbors_count]
                if minutia_crossing_number == MinutiaCrossingNumber.Termination:
                    termination_angle = follow_ridge_and_compute_angle(
                        minutia_row,
                        minutia_column,
                        direction = 8,
                        neighbors_counts = neighbors_counts,
                        followed_length_min = minutiae_followed_length_min,
                        followed_length_max = minutiae_followed_length_max,
                    )
                    if termination_angle is not None:
                        minutiae.append(TerminationWithAngle(column = minutia_column, row = minutia_row, angle = termination_angle))
                elif minutia_crossing_number == MinutiaCrossingNumber.Bifurcation:
                    neighbor_value: u8 = neighbors_counts[minutia_row, minutia_column]
                    possible_directions = DIRECTIONS_DISTANCES_LUT[neighbor_value][8]
                    if len(possible_directions) != 3:
                        continue

                    directions: list[float | None] = [None, None, None]
                    for direction_index, direction in enumerate(possible_directions):
                        neighbor = NEIGHBORS[direction]
                        angle = follow_ridge_and_compute_angle(
                            minutia_row + neighbor.row_offset,
                            minutia_column + neighbor.column_offset,
                            direction = direction,
                            neighbors_counts = neighbors_counts,
                            followed_length_min = minutiae_followed_length_min,
                            followed_length_max = minutiae_followed_length_max,
                        )
                        directions[direction_index] = angle
                    if not all(directions):
                        continue

                    valid_directions: list[float] = directions # type: ignore

                    def angle_mean(a: float, b: float) -> float:
                        return math.atan2((math.sin(a) + math.sin(b)) / 2, (math.cos(a) + math.cos(b)) / 2)

                    angle_0, angle_1 = min(
                        (valid_directions[0], valid_directions[1]),
                        (valid_directions[0], valid_directions[2]),
                        (valid_directions[1], valid_directions[2]),
                        key = lambda angle_pair: angles_abs_difference(angle_pair[0], angle_pair[1])
                    )
                    bifurcation_angle = angle_mean(angle_0, angle_1)
                    minutiae.append(BifurcationWithAngle(column = minutia_column, row = minutia_row, angle = bifurcation_angle))

        # LOCAL STRUCTURES
        if mcc_reference_cell_coordinates is None:
            mcc_reference_cell_coordinates = MccReferenceCellCoordinates(
                FE_CONFIG.mcc_total_radius,
                FE_CONFIG.mcc_circles_radius,
            )

        local_structures: NDArray[f32]
        if len(minutiae) == 0:
            local_structures = numpy.array([], dtype = f32)
        else:
            minutiae_angles = numpy.fromiter((minutia.angle for minutia in minutiae), dtype = f32)
            minutiae_coordinates = numpy.array([(minutia.column, minutia.row) for minutia in minutiae], dtype = i32)
            minutiae_directions_cosines: NDArray[f32] = numpy.cos(minutiae_angles).reshape((-1, 1, 1))
            minutiae_directions_sines: NDArray[f32] = numpy.sin(minutiae_angles).reshape((-1, 1, 1))
            minutiae_rotation_matrix = numpy.block([
                [minutiae_directions_cosines, minutiae_directions_sines],
                [-minutiae_directions_sines, minutiae_directions_cosines],
            ])

            local_structure_cell_coordinates = cast(NDArray[f64], numpy.transpose(
                minutiae_rotation_matrix
                @ mcc_reference_cell_coordinates.coordinates.T
                + minutiae_coordinates[:, :, numpy.newaxis],
                axes = [0, 2, 1]
            ))

            local_structure_minutiae_distances: NDArray[f64] = numpy.sum(
                (local_structure_cell_coordinates[:, :, numpy.newaxis,:] - minutiae_coordinates) ** 2,
                axis = -1,
                dtype = f64
            ) # type: ignore

            minutiae_spatial_contributions = Mcc_Gaussian(
                local_structure_minutiae_distances,
                mcc_gaussian_std,
            )
            (
                minutiae_spatial_contributions_rows,
                _minutiae_spatial_contributions_columns,
                _minutiae_spatial_contribution_depth,
            ) = minutiae_spatial_contributions.shape

            diagonal_indices = numpy.arange(minutiae_spatial_contributions_rows, dtype = i32)
            minutiae_spatial_contributions[diagonal_indices,: , diagonal_indices] = 0

            local_structures: NDArray[f32] = Mcc_Sigmoid(
                numpy.sum(minutiae_spatial_contributions, axis = -1, dtype = f32), # type: ignore
                mcc_sigmoid_tau,
                mcc_sigmoid_mu,
            )

        self.acquisition = FingerprintAcquisition(
            tag = acquisition_tag,
            features = FingerprintFeatures(
                singularities,
                minutiae,
                mcc_reference_cell_coordinates,
                local_structures,
            )
        )

    @staticmethod
    def from_config_object(
        file_path: str,
        acquisition_tag: str,
        *,
        config: FeaturesExtractionConfig = FE_CONFIG,
        mcc_reference_cell_coordinates: MccReferenceCellCoordinates | None
    ) -> Fingerprint:
        if mcc_reference_cell_coordinates is None:
            mcc_reference_cell_coordinates = MccReferenceCellCoordinates(
                config.mcc_total_radius,
                config.mcc_circles_radius,
            )

        return Fingerprint(
            file_path = file_path,
            acquisition_tag = acquisition_tag,
            gradient_sobel_filter_length = config.gradient_sobel_filter_length.value,
            gradient_module_block_length = config.gradient_module_block_length.value,
            segmentation_mask_threshold_scale = config.segmentation_mask_threshold_scale.value,
            directional_map_block_length = config.directional_map_block_length.value,
            directional_map_blur_filter_length = config.directional_map_blur_filter_length.value,
            local_ridge_block_rows = config.local_ridge_block_rows.value,
            local_ridge_block_columns = config.local_ridge_block_columns.value,
            gabor_filters_count = config.gabor_filters_count.value,
            gabor_filters_gamma = config.gabor_filters_gamma,
            gabor_filters_sigma = config.gabor_filters_sigma,
            binarization_threshold = config.binarization_threshold.value,
            singularities_min_distance_from_border = config.singularities_min_distance_from_border.value,
            minutiae_min_distance_from_border = config.minutiae_min_distance_from_border.value,
            minutiae_followed_length_min = config.minutiae_followed_length.low,
            minutiae_followed_length_max = config.minutiae_followed_length.high,
            mcc_reference_cell_coordinates = mcc_reference_cell_coordinates,
            mcc_gaussian_std = config.mcc_gaussian_std,
            mcc_sigmoid_tau = config.mcc_sigmoid_tau,
            mcc_sigmoid_mu = config.mcc_sigmoid_mu,
        )

    @staticmethod
    def from_config(
        file_path: str,
        acquisition_tag: str,
        *,
        gradient_sobel_filter_length: int = GRADIENT_SOBEL_FILTER_LENGTH,
        gradient_module_block_length: int = GRADIENT_MODULE_BLOCK_LENGTH,
        segmentation_mask_threshold_scale: float = SEGMENTATION_MASK_THRESHOLD_SCALE,
        directional_map_block_length: int = DIRECTIONAL_MAP_BLOCK_LENGTH,
        directional_map_blur_filter_length: int = DIRECTIONAL_MAP_BLUR_FILTER_LENGTH,
        local_ridge_block_rows: int = LOCAL_RIDGE_BLOCK_ROWS,
        local_ridge_block_columns: int = LOCAL_RIDGE_BLOCK_COLUMNS,
        gabor_filters_count: int = GABOR_FILTERS_COUNT,
        gabor_filters_sigma: float = GABOR_FILTERS_GAMMA,
        gabor_filters_gamma: float = GABOR_FILTERS_SIGMA,
        binarization_threshold: int = BINARIZATION_THRESHOLD,
        singularities_min_distance_from_border: int = SINGULARITIES_MIN_DISTANCE_FROM_BORDER,
        minutiae_min_distance_from_border: float = MINUTIAE_MIN_DISTANCE_FROM_BORDER,
        minutiae_followed_length_min: int = MINUTIAE_FOLLOWED_LENGTH_MIN,
        minutiae_followed_length_max: int = MINUTIAE_FOLLOWED_LENGTH_MAX,
        mcc_reference_cell_coordinates: MccReferenceCellCoordinates | None,
        mcc_gaussian_std: float = MCC_GAUSSIAN_STD,
        mcc_sigmoid_tau: float = MCC_SIGMOID_TAU,
        mcc_sigmoid_mu: float = MCC_SIGMOID_MU,
    ) -> Fingerprint:
        if mcc_reference_cell_coordinates is None:
            mcc_reference_cell_coordinates = MccReferenceCellCoordinates(
                MCC_TOTAL_RADIUS,
                MCC_CIRCLES_RADIUS,
            )

        return Fingerprint(
            file_path = file_path,
            acquisition_tag = acquisition_tag,
            gradient_sobel_filter_length = gradient_sobel_filter_length,
            gradient_module_block_length = gradient_module_block_length,
            segmentation_mask_threshold_scale = segmentation_mask_threshold_scale,
            directional_map_block_length = directional_map_block_length,
            directional_map_blur_filter_length = directional_map_blur_filter_length,
            local_ridge_block_rows = local_ridge_block_rows,
            local_ridge_block_columns = local_ridge_block_columns,
            gabor_filters_count = gabor_filters_count,
            gabor_filters_gamma = gabor_filters_gamma,
            gabor_filters_sigma = gabor_filters_sigma,
            binarization_threshold = binarization_threshold,
            singularities_min_distance_from_border = singularities_min_distance_from_border,
            minutiae_min_distance_from_border = minutiae_min_distance_from_border,
            minutiae_followed_length_min = minutiae_followed_length_min,
            minutiae_followed_length_max = minutiae_followed_length_max,
            mcc_reference_cell_coordinates = mcc_reference_cell_coordinates,
            mcc_gaussian_std = mcc_gaussian_std,
            mcc_sigmoid_tau = mcc_sigmoid_tau,
            mcc_sigmoid_mu = mcc_sigmoid_mu,
        )
