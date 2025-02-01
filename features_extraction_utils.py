import ipywidgets # type: ignore
import numpy
import math
import cv2
from typing import Any, NamedTuple
from numpy.typing import NDArray
from numpy import (
    int64 as i64,
    uint8 as u8,
    float32 as f32,
    float64 as f64,
    floating
)
import html
import base64
import IPython.display

from fingerprint import FE_CONFIG, FM_CONFIG, Bifurcation, BifurcationWithAngle, Bounds, Core, Delta, FingerprintFeatures, Minutia, MinutiaWithAngle, Point, PointWithAngle, Range, Singularity, Termination, TerminationWithAngle, Whorl, angles_abs_difference

class ColorBGR(NamedTuple):
    b: int
    g: int
    r: int

PI = numpy.pi
PI_DIV_2 = PI / 2

U8_BITS = numpy.iinfo(u8).bits
U8_MIN = numpy.iinfo(u8).min
U8_MAX = numpy.iinfo(u8).max

RED   = ColorBGR(r = U8_MAX, g = 0,      b = 0)
GREEN = ColorBGR(r = 0,      g = U8_MAX, b = 0)
BLUE  = ColorBGR(r = 0,      g = 0,      b = U8_MAX)

DIRECTIONAL_MAP_LINES_LENGTH_SCALE = FE_CONFIG.directional_map_block_length.value * 0.5
DIRECTIONAL_MAP_LINES_COLOR = RED

def int_slider(range: Range[int], description: str) -> ipywidgets.IntSlider:
    return ipywidgets.IntSlider(
        min = range.min,
        max = range.max,
        step = range.step,
        value = range.value,
        description = description,
        layout = ipywidgets.Layout(width = "auto"),
        style = {"description_width": "initial"},
    )

def float_slider(range: Range[float], description: str) -> ipywidgets.FloatSlider:
    return ipywidgets.FloatSlider(
        min = range.min,
        max = range.max,
        step = range.step,
        value = range.value,
        description = description,
        layout = ipywidgets.Layout(width = "auto"),
        style = {"description_width": "initial"},
    )

def int_range_slider(bounds: Bounds[int], description: str) -> ipywidgets.IntRangeSlider:
    return ipywidgets.IntRangeSlider(
        min = bounds.min,
        max = bounds.max,
        step = bounds.step,
        value = (bounds.low, bounds.high),
        description = description,
        layout = ipywidgets.Layout(width = "auto"),
        style = {"description_width": "initial"}
    )

def float_text(value: float, description: str) -> ipywidgets.FloatText:
    return ipywidgets.FloatText(
        value = value,
        description = description,
        layout = ipywidgets.Layout(width = "auto"),
        style = {"description_width": "initial"}
    )

def show(
    *titles_and_images:     tuple[str, NDArray[Any]],
    enlarge_small_images: bool = True,
    max_images_per_row: int = -1,
    font_size: int = 0,
) -> None:
    def encode_base64(image: NDArray[Any]) -> str:
        if image.dtype != u8:
            image_min, image_max = image.min(), image.max()
            if image_min==image_max:
                offset, scale, d = 0, 0, 1
            elif image_min<0:
                offset, scale, d = 128, 127, max(-image_min, abs(image_max))
            else:
                offset, scale, d = 0, 255, image_max
            image = numpy.clip(offset + scale*(image.astype(float))/d, 0, 255).astype(u8)

        if enlarge_small_images:
            REF_SCALE = 100
            height, width, *_ = image.shape
            if height < REF_SCALE or width < REF_SCALE:
                scale = max(1, min(REF_SCALE // height, REF_SCALE // width))
                image = cv2.resize(
                    image,
                    dsize = (width * scale, height * scale),
                    interpolation = cv2.INTER_NEAREST
                )
        data = "data:image/png;base64," + base64.b64encode(
            cv2.imencode(".png", image)[1]
        ).decode("utf8")
        return data

    if max_images_per_row < 0:
        max_images_per_row = len(titles_and_images)

    font = f"font-size: {font_size}px;" if font_size > 0 else ""

    html_content = ""
    for row in range(0, len(titles_and_images), max_images_per_row):
        titles_and_images_row = titles_and_images[row: row + max_images_per_row]
        html_content += "<table>"

        html_content += "<tr>"
        for title, _image in titles_and_images_row:
            html_content += f"<td style='text-align:center;{font}'>{html.escape(title)}</td>"
        html_content += "</tr>"

        html_content += "<tr>"
        for _title, image in titles_and_images_row:
            image_base64 = encode_base64(image)
            html_content += f"<td style='text-align:center;'><img src='{image_base64}'></td>"
        html_content += "</tr>"

        html_content += "</table>"

    IPython.display.display(IPython.display.HTML(html_content)) # type: ignore

def convert_to_bgr_if_grayscale(image: NDArray[u8]) -> tuple[int, int, NDArray[u8]]:
    image_rows: int
    image_columns: int
    image_bgr: NDArray[u8]
    if len(image.shape) == 2:
        image_rows, image_columns = image.shape
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # type: ignore
    else:
        image_rows, image_columns, *_image_channels = image.shape
        image_bgr = image
    return (image_rows, image_columns, image_bgr)

def draw_directional_map_lines(
    fingerprint: NDArray[u8],
    directional_map: NDArray[f32],
    segmentation_mask: NDArray[u8],
    directional_map_block_length: int
) -> NDArray[u8]:
    fingerprint_rows, fingerprint_columns, fingerprint_with_directional_lines = convert_to_bgr_if_grayscale(fingerprint)

    half_directional_map_block_length = directional_map_block_length // 2
    directional_map_block_rows = fingerprint_rows // directional_map_block_length
    directional_map_block_columns = fingerprint_columns // directional_map_block_length
    for block_row in range(directional_map_block_rows):
        center_row = block_row * directional_map_block_length + half_directional_map_block_length
        for block_column in range(directional_map_block_columns):
            center_column = block_column * directional_map_block_length + half_directional_map_block_length

            if segmentation_mask[center_row, center_column] == 0:
                continue

            direction = float(directional_map[center_row, center_column])
            line_length_x = int(round(math.cos(direction) * DIRECTIONAL_MAP_LINES_LENGTH_SCALE))
            line_length_y = int(round(-math.sin(direction) * DIRECTIONAL_MAP_LINES_LENGTH_SCALE))
            _ = cv2.line(
                fingerprint_with_directional_lines,
                pt1 = (center_column - line_length_x, center_row - line_length_y),
                pt2 = (center_column + line_length_x, center_row + line_length_y),
                color = DIRECTIONAL_MAP_LINES_COLOR,
                thickness = 1,
                lineType = cv2.LINE_AA
            )
    return fingerprint_with_directional_lines

CORE_COLOR = RED
WHORL_COLOR = GREEN
DELTA_COLOR = BLUE

CORE_RADIUS = 5
WHORL_RADIUS = CORE_RADIUS
HALF_DELTA_HEIGHT = CORE_RADIUS
HALF_DELTA_BASE = HALF_DELTA_HEIGHT
def draw_singularities(
    fingerprint: NDArray[u8],
    singularities: list[Singularity],
) -> NDArray[u8]:
    _fingerprint_rows, _fingerprint_columns, fingerprint_with_singularities = convert_to_bgr_if_grayscale(fingerprint)

    for singularity in singularities:
        match singularity:
            case Core():
                _ = cv2.circle(
                    fingerprint_with_singularities,
                    center = (singularity.column - CORE_RADIUS, singularity.row - CORE_RADIUS),
                    radius = CORE_RADIUS,
                    color = CORE_COLOR,
                    thickness = 1,
                    lineType = cv2.LINE_AA
                )
            case Whorl():
                _ = cv2.circle(
                    fingerprint_with_singularities,
                    center = (singularity.column - WHORL_RADIUS, singularity.row - WHORL_RADIUS),
                    radius = WHORL_RADIUS,
                    color = WHORL_COLOR,
                    thickness = 1,
                    lineType = cv2.LINE_AA
                )
            case Delta():
                bottom_left_vertex = Point(singularity.column - HALF_DELTA_BASE, singularity.row + HALF_DELTA_HEIGHT)
                bottom_right_vertex = Point(singularity.column + HALF_DELTA_BASE, singularity.row + HALF_DELTA_HEIGHT)
                top_vertex = Point(singularity.column, singularity.row - HALF_DELTA_HEIGHT)
                _ = cv2.line(
                    fingerprint_with_singularities,
                    pt1 = (bottom_left_vertex.column - HALF_DELTA_BASE, bottom_left_vertex.row - HALF_DELTA_BASE),
                    pt2 = (bottom_right_vertex.column - HALF_DELTA_BASE, bottom_right_vertex.row - HALF_DELTA_BASE),
                    color = DELTA_COLOR,
                    thickness = 1,
                    lineType = cv2.LINE_AA
                )
                _ = cv2.line(
                    fingerprint_with_singularities,
                    pt1 = (bottom_right_vertex.column - HALF_DELTA_BASE, bottom_right_vertex.row - HALF_DELTA_BASE),
                    pt2 = (top_vertex.column - HALF_DELTA_BASE, top_vertex.row - HALF_DELTA_BASE),
                    color = DELTA_COLOR,
                    thickness = 1,
                    lineType = cv2.LINE_AA
                )
                _ = cv2.line(
                    fingerprint_with_singularities,
                    pt1 = (top_vertex.column - HALF_DELTA_BASE, top_vertex.row - HALF_DELTA_BASE),
                    pt2 = (bottom_left_vertex.column - HALF_DELTA_BASE, bottom_left_vertex.row - HALF_DELTA_BASE),
                    color = DELTA_COLOR,
                    thickness = 1,
                    lineType = cv2.LINE_AA
                )
            case _: assert False, "unreachable"
    return fingerprint_with_singularities

TERMINATION_COLOR = RED
BIFURCATION_COLOR = BLUE

def draw_minutiae(
    fingerprint: NDArray[u8],
    minutiae: list[Minutia],
) -> NDArray[u8]:
    _fingerprint_rows, _fingerprint_columns, fingerprint_with_minutiae = convert_to_bgr_if_grayscale(fingerprint)
    for minutia in minutiae:
        minutia_color: ColorBGR
        match minutia:
            case Termination(): minutia_color = TERMINATION_COLOR
            case Bifurcation(): minutia_color = BIFURCATION_COLOR
            case _: assert False, "unreachable"

        _ = cv2.drawMarker(
            fingerprint_with_minutiae,
            position = minutia,
            color = minutia_color,
            markerType = cv2.MARKER_CROSS,
            markerSize = 8
        )
    return fingerprint_with_minutiae

def draw_minutiae_with_angle(
    fingerprint: NDArray[u8],
    minutiae: list[MinutiaWithAngle],
) -> NDArray[u8]:
    _fingerprint_rows, _fingerprint_columns, fingerprint_with_minutiae = convert_to_bgr_if_grayscale(fingerprint)
    for minutia in minutiae:
        minutia_color: ColorBGR
        match minutia:
            case TerminationWithAngle(): minutia_color = TERMINATION_COLOR
            case BifurcationWithAngle(): minutia_color = BIFURCATION_COLOR
            case _: assert False, "unreachable"

        _ = cv2.circle(
            fingerprint_with_minutiae,
            center = (minutia.column, minutia.row),
            radius = 3,
            color = minutia_color,
            thickness = 1,
            lineType = cv2.LINE_AA
        )

        direction_line_width = int(round(math.cos(minutia.angle) * 10))
        direction_line_height = int(round(math.sin(minutia.angle) * 10))
        _ = cv2.line(
            fingerprint_with_minutiae,
            pt1 = (minutia.column, minutia.row),
            pt2 = (minutia.column + direction_line_width, minutia.row - direction_line_height),
            color = minutia_color,
            thickness = 1,
            lineType = cv2.LINE_AA
        )
    return fingerprint_with_minutiae

def draw_minutiae_and_cylinder(
    fingerprint: NDArray[u8],
    features: FingerprintFeatures,
    minutia_index: int
) -> NDArray[u8]:
    def compute_actual_cylinder_coordinates(minutia: MinutiaWithAngle) -> NDArray[floating]:
        minutia_angle_cosines = math.cos(minutia.angle)
        minutia_angle_sines = math.sin(minutia.angle)
        rotation_matrix = numpy.array([
            [minutia_angle_cosines, minutia_angle_sines],
            [-minutia_angle_sines, minutia_angle_cosines]
        ])
        return (rotation_matrix
            @ features.mcc_reference_cell_coordinates.coordinates.T
            + numpy.array([minutia.column, minutia.row])[:,numpy.newaxis]
        ).T

    _fingerprint_rows, _fingerprint_columns, fingerprint_with_minutiae_and_cylinder = convert_to_bgr_if_grayscale(fingerprint)
    minutia = features.minutiae[minutia_index]
    local_structure: NDArray[f64] = features.local_structures[minutia_index]
    for local_structure_intensity, (cilinder_center_column, cilinder_center_row) in zip(local_structure, compute_actual_cylinder_coordinates(minutia), strict = True):
        local_structure_intensity: f64
        cilinder_center_column: float
        cilinder_center_row: float
        _ = cv2.circle(
            img = fingerprint_with_minutiae_and_cylinder,
            center = (int(round(cilinder_center_column)), int(round(cilinder_center_row))),
            radius = 3,
            color = (0x80, int(round(local_structure_intensity * 0xff)), 0x80),
            thickness = 1,
            lineType = cv2.LINE_AA,
        )

    return fingerprint_with_minutiae_and_cylinder

def draw_match_pairs(
    fingerprint_1: NDArray[u8],
    features_1: FingerprintFeatures,
    fingerprint_2: NDArray[u8],
    features_2: FingerprintFeatures,
    pairs: tuple[NDArray[i64], NDArray[i64]],
    matching_pair_index: int
) -> NDArray[u8]:
    fingerprint_1_rows, fingerprint_1_columns, fingerprint_1 = convert_to_bgr_if_grayscale(fingerprint_1)
    fingerprint_2_rows, fingerprint_2_columns, fingerprint_2 = convert_to_bgr_if_grayscale(fingerprint_2)

    p1, p2 = pairs
    fingerprints_with_matching_local_structures = numpy.full(
        shape = (max(fingerprint_1_rows,fingerprint_2_rows), fingerprint_1_columns + fingerprint_2_columns, 3),
        fill_value = 255,
        dtype = u8
    )
    fingerprints_with_matching_local_structures[
        : fingerprint_1_rows,
        : fingerprint_1_columns
    ] = draw_minutiae_and_cylinder(fingerprint_1, features_1, p1[matching_pair_index])
    fingerprints_with_matching_local_structures[
        : fingerprint_2_rows,
        fingerprint_1_columns: fingerprint_1_columns + fingerprint_2_columns
    ] = draw_minutiae_and_cylinder(fingerprint_2, features_2, p2[matching_pair_index])

    for current_local_structure_index, (i1, i2) in enumerate(zip(p1, p2, strict = True)):
        i1: i64
        i2: i64
        minutiae_1_i1 = features_1.minutiae[i1]
        minutiae_2_i2 = features_2.minutiae[i2]
        if current_local_structure_index == matching_pair_index:
            _ = cv2.line(
                fingerprints_with_matching_local_structures,
                pt1 = (minutiae_1_i1.column, minutiae_1_i1.row),
                pt2 = (fingerprint_1_columns+minutiae_2_i2.column, minutiae_2_i2.row),
                color = GREEN,
                thickness = 1,
                lineType = cv2.LINE_AA
            )

    return fingerprints_with_matching_local_structures

def matching_score_local_structures_show_progress(
    self: FingerprintFeatures,
    other: FingerprintFeatures,
    self_image_path: str,
    other_image_path: str
) -> float:
    distances: NDArray[f64] = numpy.linalg.norm(
        self.local_structures[:, numpy.newaxis,:] - other.local_structures,
        axis = -1
    )
    distances /= numpy.linalg.norm(self.local_structures, axis = 1)[:, numpy.newaxis] + numpy.linalg.norm(other.local_structures, axis = 1)
    minutiae_matching_pairs: tuple[NDArray[i64], NDArray[i64]] = numpy.unravel_index(
        numpy.argpartition(distances, FM_CONFIG.local_structures_matching_minutiae_pair_count, None)[: FM_CONFIG.local_structures_matching_minutiae_pair_count],
        distances.shape
    ) # type: ignore
    matching_score = 1 - numpy.mean(distances[minutiae_matching_pairs[0], minutiae_matching_pairs[1]])

    self_raw_fingerprint: NDArray[u8] = cv2.imread(self_image_path, flags = cv2.IMREAD_GRAYSCALE) # type: ignore
    other_raw_fingerprint: NDArray[u8] = cv2.imread(other_image_path, flags = cv2.IMREAD_GRAYSCALE) # type: ignore

    self_with_minutiae = draw_minutiae_with_angle(self_raw_fingerprint, self.minutiae)
    other_with_minutiae = draw_minutiae_with_angle(other_raw_fingerprint, other.minutiae)

    match_pairs_images: list[tuple[str, NDArray[u8]]] = []
    for i in range(len(minutiae_matching_pairs[0])):
        match_pairs = draw_match_pairs(
            self_with_minutiae.copy(),
            self,
            other_with_minutiae.copy(),
            other,
            minutiae_matching_pairs,
            i,
        )

        match_pair_name = f"match pair {i}"
        match_pairs_images.append((match_pair_name, match_pairs))

    show(*match_pairs_images)
    return float(matching_score)

def matching_score_hough_show_progress(
    self: FingerprintFeatures,
    other: FingerprintFeatures,
    self_image_path: str,
    other_image_path: str
) -> float:
    def is_close(row_0: int, column_0: int, row_1: int, column_1: int, radius: int) -> bool:
        return (column_0 - column_1) ** 2 + (row_0 - row_1) ** 2 <= radius ** 2

    accumulator: dict[PointWithAngle, int] = {}

    for other_minutia in other.minutiae:
        for self_minutia in self.minutiae:
            angles_difference = angles_abs_difference(self_minutia.angle, other_minutia.angle)
            angles_difference_cos = math.cos(angles_difference)
            angles_difference_sin = math.sin(angles_difference)

            translation_column = other_minutia.column - (self_minutia.column * angles_difference_cos + self_minutia.row * angles_difference_sin)
            translation_row = other_minutia.row - (self_minutia.column * angles_difference_sin - self_minutia.row * angles_difference_cos)

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
            max_votes = votes

    rotation_angle_cos = math.cos(most_voted_translation.angle)
    rotation_angle_sin = math.sin(most_voted_translation.angle)
    rotation_matrix: NDArray[f64] = numpy.array([ # type: ignore
        [rotation_angle_cos, -rotation_angle_sin],
        [rotation_angle_sin, rotation_angle_cos],
    ])

    # self_aligned_minutiae = (
    #     numpy.array([(minutia.column, minutia.row) for minutia in self.minutiae], dtype = i64)
    #     @ rotation_matrix + numpy.array([most_voted_translation.column, most_voted_translation.row], dtype = i64)
    # ).round().astype(i64)
    self_aligned_minutiae = (
        numpy.array([(minutia.column, minutia.row) for minutia in self.minutiae], dtype = i64)
            + numpy.array([most_voted_translation.column, most_voted_translation.row], dtype = i64)
    ).round().astype(i64)
    self_aligned_minutiae = [
        Minutia(column = column, row = row) for column, row in self_aligned_minutiae
    ]

    self_raw_fingerprint: NDArray[u8] = cv2.imread(self_image_path, flags = cv2.IMREAD_GRAYSCALE) # type: ignore
    other_raw_fingerprint: NDArray[u8] = cv2.imread(other_image_path, flags = cv2.IMREAD_GRAYSCALE) # type: ignore

    self_raw_fingeprint_rows, self_raw_fingerprint_columns = self_raw_fingerprint.shape
    other_raw_fingerprint_rows, other_raw_fingerprint_columns = other_raw_fingerprint.shape
    res = numpy.empty(
        shape = (max(self_raw_fingeprint_rows,other_raw_fingerprint_rows), self_raw_fingerprint_columns+other_raw_fingerprint_columns, 3),
        dtype = u8
    )
    res[
        : self_raw_fingeprint_rows,
        :self_raw_fingerprint_columns
    ] = draw_minutiae_with_angle(
        self_raw_fingerprint,
        self.minutiae,
    )
    res[
        : other_raw_fingerprint_rows,
        self_raw_fingerprint_columns: self_raw_fingerprint_columns + other_raw_fingerprint_columns
    ] = draw_minutiae_with_angle(
        other_raw_fingerprint,
        other.minutiae,
    )

    matching_minutiae_count = 0
    for other_minutia in other.minutiae:
        for self_minutia, self_aligned_minutia in zip(self.minutiae, self_aligned_minutiae, strict = True):
            minutiae_are_close = is_close(
                self_aligned_minutia.row,
                self_aligned_minutia.column,
                other_minutia.row,
                other_minutia.column,
                radius = FM_CONFIG.hough_matching_minutiae_pixels_distance_threshold
            )
            # minutiae_are_aligned = is_aligned(aligned_angle, other_minutia.angle)

            if minutiae_are_close:
                _ = cv2.line(
                    res,
                    pt1 = (self_minutia.column, self_minutia.row),
                    pt2 = (self_raw_fingerprint_columns + other_minutia.column, other_minutia.row),
                    color = (0, 0, U8_MAX),
                    thickness = 1,
                    lineType = cv2.LINE_AA
                )

                matching_minutiae_count += 1
                break

    other_fingerprint_with_self_aligned_minutiae = draw_minutiae(
        other_raw_fingerprint,
        self_aligned_minutiae,
    )
    show(
        ("both minutiae", res),
        ("self aligned minutiae", other_fingerprint_with_self_aligned_minutiae),
    )

    matching_score = matching_minutiae_count / max(len(self.minutiae), len(other.minutiae))
    return matching_score
