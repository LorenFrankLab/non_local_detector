"""
regions/io.py
=============

Helper functions for importing *continuous* ROIs (``Region`` / ``Regions``)
from common labelling formats and exporting them to a shareable JSON schema.

"""

from __future__ import annotations

import json
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

import numpy as np
import pandas as pd
import shapely.geometry as shp
from numpy.typing import NDArray

from .core import Region, Regions

if TYPE_CHECKING:
    from shapely.geometry import Polygon

    from ..transforms import SpatialTransform

# --------------------------------------------------------------------------
# 1.  Generic JSON <--> Regions round-trip
# --------------------------------------------------------------------------
_SCHEMA_TAG = "Regions-v1"


def regions_to_json(
    regions: Regions, path: Union[str, Path], *, indent: int = 2
) -> None:
    """Write a list-of-dicts file that any language can read.

    Parameters
    ----------
    regions : Regions
        Collection of regions to write.
    path : str or Path
        Destination file path.
    indent : int, optional
        Indentation level for the JSON file. Default is 2 spaces.

    """
    payload = {
        "format": _SCHEMA_TAG,
        "regions": [reg.to_dict() for reg in regions.values()],
    }
    # Ensure the parent directory exists before writing
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=indent))


def regions_from_json(path: Union[str, Path]) -> Regions:
    """Load the schema written by :func:`regions_to_json`.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file containing regions.

    Returns
    -------
    Regions
        A collection of :class:`Region` objects representing the shapes in the JSON.
    """
    blob = json.loads(Path(path).read_text())
    if blob.get("format") != _SCHEMA_TAG:
        warnings.warn(
            f"Unrecognised format tag {blob.get('format')!r}; attempting best-effort load"
        )
    return Regions(Region.from_dict(d) for d in blob["regions"])


# --------------------------------------------------------------------------
# 2.  LabelMe / CVAT / VIA-style polygon JSON/XML  → Regions
# --------------------------------------------------------------------------
def load_labelme_json(
    json_path: Union[str, Path],
    *,
    pixel_to_world: Optional[SpatialTransform] = None,
    label_key: str = "label",
    points_key: str = "points",
) -> Regions:
    """
    Parse a *.json* file produced by many point-&-click ROI tools.

    Parameters
    ----------
    json_path : str or Path
        Path to the JSON file.
    pixel_to_world : SpatialTransform, optional
        Callable mapping `(N, 2)` **pixel** coordinates → `(N, 2)` **world**
        coordinates (e.g., centimeters). If `None`, the coordinates are
        assumed to be in pixels.
    label_key : str, optional
        Key in the JSON object that contains the region label.
        Default is "label".
    points_key : str, optional
        Key in the JSON object that contains the list of points.
        Default is "points".

    Returns
    -------
    Regions
        Every polygon becomes a :class:`Region` with ``kind="polygon"``.

    Raises
    ------
    ValueError
        If no shapes are found in the JSON file or if a shape is missing
        required keys.
    """
    data = json.loads(Path(json_path).read_text())
    regions_list: list[Region] = []  # Renamed to avoid conflict with Regions class

    # LabelMe puts shapes in a top-level list; CVAT stores under "shapes"
    shapes_data = data.get("shapes", data if isinstance(data, list) else [])
    if not isinstance(shapes_data, list):  # Ensure shapes_data is a list
        raise ValueError(
            f"Expected 'shapes' to be a list, but got {type(shapes_data).__name__} in {json_path}"
        )
    if not shapes_data:
        warnings.warn(
            f"No shapes found in {json_path}"
        )  # Changed to warning, or could be error
        return Regions([])  # Return empty Regions if no shapes

    for i, obj in enumerate(shapes_data):
        if not isinstance(obj, dict):
            warnings.warn(f"Shape at index {i} is not a dictionary, skipping.")
            continue

        name = obj.get(label_key)
        points_data = obj.get(points_key)

        if name is None:
            warnings.warn(
                f"Shape at index {i} is missing label (key: '{label_key}'), skipping."
            )
            continue
        if points_data is None:
            warnings.warn(
                f"Shape '{name}' (index {i}) is missing points (key: '{points_key}'), skipping."
            )
            continue

        try:
            pts_px: NDArray[np.float64] = np.asarray(points_data, dtype=float)  # (M, 2)
            if pts_px.ndim != 2 or pts_px.shape[1] != 2:
                warnings.warn(
                    f"Points for shape '{name}' (index {i}) are not in (M, 2) format, skipping."
                )
                continue
        except ValueError as e:
            warnings.warn(
                f"Could not parse points for shape '{name}' (index {i}): {e}, skipping."
            )
            continue

        pts_transformed = pixel_to_world(pts_px) if pixel_to_world else pts_px
        # Ensure polygon has at least 3 points for a valid polygon
        if len(pts_transformed) < 3:
            warnings.warn(
                f"Shape '{name}' (index {i}) has fewer than 3 points after processing, skipping."
            )
            continue
        try:
            poly = shp.Polygon(pts_transformed)
        except Exception as e:  # Catch potential shapely errors
            warnings.warn(
                f"Could not create polygon for shape '{name}' (index {i}): {e}, skipping."
            )
            continue

        regions_list.append(
            Region(
                name=str(name),  # Ensure name is a string
                kind="polygon",
                data=poly,
                metadata={"source_json": Path(json_path).name},
            )
        )
    return Regions(regions_list)


def _rle_to_mask(rle: str, height: int, width: int) -> np.ndarray:
    """
    Convert a Run-Length Encoded string to a binary mask.

    Parameters
    ----------
    rle : str
        The Run-Length Encoding string, e.g., "start1,length1,start2,length2,...".
        Values are 0-indexed and refer to the flattened image.
    height : int
        The height of the image (in pixels).
    width : int
        The width of the image (in pixels).

    Returns
    -------
    mask : NDArray[np.uint8], shape (height, width)
        A binary mask where 1 represents the mask area and 0 is background.

    Raises
    ------
    ValueError
        If RLE string is malformed or values are out of bounds.
    """
    try:
        rle_values = list(map(int, rle.split(",")))
    except ValueError:
        raise ValueError(f"RLE string contains non-integer values: {rle}")

    if len(rle_values) % 2 != 0:
        raise ValueError(f"RLE string has an odd number of values: {rle}")

    mask = np.zeros(height * width, dtype=np.uint8)

    # Unpack RLE values into the mask
    for start, length in zip(rle_values[::2], rle_values[1::2]):
        mask[start : start + length] = 1  # Set the corresponding region to 1

    return mask.reshape((height, width))  # Reshape to image dimensions


def _mask_to_polygon(mask: NDArray[np.uint8]) -> Polygon:
    """
    Converts a binary mask to a Shapely Polygon.

    Parameters
    ----------
    mask : np.ndarray, shape (H, W)
        A binary mask (2D array).

    Returns
    -------
    shapely.geometry.Polygon
        A Shapely Polygon representing the mask.
    """
    import cv2  # type: ignore

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        # Take the largest contour if there are multiple
        contour = max(contours, key=cv2.contourArea)
        return Polygon(contour.reshape(-1, 2))  # Convert to Polygon
    return Polygon()  # Return an empty polygon if no contour found


def _parse_cvat_points(
    points_str: str,
    shape_label: Optional[str] = None,
    shape_index: Optional[int] = None,
) -> NDArray[np.float64]:
    """Parse a CVAT-style points string into a NumPy array.

    CVAT points are formatted as "x1,y1;x2,y2;...;xn,yn".

    Parameters
    ----------
    points_str : str
        String containing points in the format "x1,y1;x2,y2;...;xn,yn".
    shape_label : str, optional
        Label of the shape these points belong to (for warning messages).
    shape_index : int, optional
        Index of the shape these points belong to (for warning messages).


    Returns
    -------
    parsed_points : NDArray[np.float64], shape (N, 2)
        Parsed points as floats.

    Raises
    ------
    ValueError
        If the points string is malformed.
    """
    parsed_points: list[list[float]] = []
    if not points_str.strip():  # Handle empty string case
        return np.empty((0, 2), dtype=float)

    point_pairs = points_str.strip().split(";")
    for i, pt_pair_str in enumerate(point_pairs):
        if not pt_pair_str.strip():  # Handle empty pair (e.g., "x,y;;x,y")
            warning_msg = f"Empty point pair found in points string"
            if shape_label and shape_index is not None:
                warning_msg += f" for shape '{shape_label}' (index {shape_index})"
            warning_msg += f" at pair index {i}."
            warnings.warn(warning_msg + " Skipping this pair.")
            continue
        try:
            x_str, y_str = pt_pair_str.split(",")
            parsed_points.append([float(x_str), float(y_str)])
        except ValueError:
            # More informative error message
            context = ""
            if shape_label and shape_index is not None:
                context = f" for shape '{shape_label}' (index {shape_index})"
            raise ValueError(
                f"Malformed point string{context}: "
                f"Could not parse '{pt_pair_str}' into two float coordinates. Full string: '{points_str}'"
            )
    return np.array(parsed_points, dtype=float)


def _create_cvat_region(
    name: str,
    shape_data: Union[Polygon, NDArray[np.float64]],  # Polygon or Point data
    kind: str,
    xml_path: Path,
    color: Optional[str],
    additional_metadata: Optional[Mapping[str, Any]] = None,
) -> Region:
    """Helper function to create a Region object for CVAT shapes."""
    metadata = {
        "source_xml": xml_path.name,
        "color": color,
        "plot_kwargs": {"color": color} if color else {},
    }
    if additional_metadata:
        metadata.update(additional_metadata)

    return Region(name=name, kind=kind, data=shape_data, metadata=metadata)


def load_cvat_xml(
    xml_path: Union[str, Path], *, pixel_to_world: Optional[SpatialTransform] = None
) -> Regions:
    """
    Parse a *.xml* file produced by CVAT.

    Region names are generated based on their label. If a label appears
    multiple times within an image, a running index is appended (e.g., tumor_0,
    tumor_1). If a label is unique within an image, no index is added (e.g., tumor).
    Unlabeled shapes follow the same logic (e.g., unlabeled or unlabeled_0).
    Open polylines are ignored. Other shapes are converted to Regions.

    Parameters
    ----------
    xml_path : str or Path
        Path to the CVAT XML file.
    pixel_to_world : SpatialTransform, optional
        Callable mapping `(N,2)` pixel coords → `(N,2)` world coords.

    Returns
    -------
    Regions
        A collection of :class:`Region` objects.

    Raises
    ------
    FileNotFoundError
        If `xml_path` does not exist.
    ET.ParseError
        If `xml_path` is not a valid XML file.
    """
    path_obj = Path(xml_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    try:
        tree = ET.parse(path_obj)
    except ET.ParseError as e:
        raise ET.ParseError(f"Error parsing XML file {xml_path}: {e}") from e

    root = tree.getroot()
    # This list will collect all regions from all images in the file
    all_regions_in_file: list[Region] = []

    label_colors: dict[str, str] = {}
    labels_elem = root.find(".//labels")
    if labels_elem is not None:
        for label_elem in labels_elem.findall("label"):
            name_elem = label_elem.find("name")
            color_elem = label_elem.find("color")
            if (
                name_elem is not None
                and name_elem.text
                and color_elem is not None
                and color_elem.text
            ):
                label_colors[name_elem.text.strip()] = color_elem.text.strip()

    def get_processed_label_and_color(
        raw_label_xml: Optional[str],
    ) -> tuple[str, Optional[str]]:
        """Processes raw label from XML and gets its color."""
        processed_label = (
            raw_label_xml.strip()
            if raw_label_xml and raw_label_xml.strip()
            else "unlabeled"
        )
        # Color is fetched based on the raw label (if it existed) before processing to "unlabeled"
        color = None
        if raw_label_xml and raw_label_xml.strip():
            color = label_colors.get(raw_label_xml.strip())
        return processed_label, color

    for image_idx, image in enumerate(root.findall("image")):
        image_id_str = image.get("id", f"image{image_idx}")

        # --- Pass 1 for current image: Collect shape data and count label occurrences ---
        # Stores tuples of (processed_label, geometric_data, kind_str, color_str)
        collected_shapes_data: list[tuple[str, Any, str, Optional[str]]] = []
        label_total_counts: dict[str, int] = {}

        # Polygons
        for elem_idx, polygon_elem in enumerate(image.findall("polygon")):
            raw_label = polygon_elem.get("label")
            points_str = polygon_elem.get("points")
            processed_label, color = get_processed_label_and_color(raw_label)

            if not points_str:  # Warn and skip if essential data missing
                warnings.warn(
                    f"Image '{image_id_str}', Polygon (raw_label: {raw_label}, "
                    f"xml_idx: {elem_idx}): missing 'points'. Skipping."
                )
                continue
            try:
                pts_px = _parse_cvat_points(points_str, raw_label, elem_idx)
                if pts_px.shape[0] < 3:
                    warnings.warn(
                        f"Image '{image_id_str}', Polygon (raw_label: {raw_label}, "
                        f"xml_idx: {elem_idx}): < 3 points. Skipping."
                    )
                    continue
                pts_transformed = pixel_to_world(pts_px) if pixel_to_world else pts_px
                geom = shp.Polygon(pts_transformed)
            except (ValueError, Exception) as e:  # Catch parsing or geometry errors
                warnings.warn(
                    f"Image '{image_id_str}', Polygon (raw_label: {raw_label}, "
                    f"xml_idx: {elem_idx}): error processing: {e}. Skipping."
                )
                continue

            label_total_counts[processed_label] = (
                label_total_counts.get(processed_label, 0) + 1
            )
            collected_shapes_data.append((processed_label, geom, "polygon", color))

        # Polylines (closed ones become polygons)
        for elem_idx, polyline_elem in enumerate(image.findall("polyline")):
            raw_label = polyline_elem.get("label")
            points_str = polyline_elem.get("points")
            processed_label, color = get_processed_label_and_color(raw_label)

            if not points_str:
                warnings.warn(
                    f"Image '{image_id_str}', Polyline (raw_label: {raw_label}, "
                    f"xml_idx: {elem_idx}): missing 'points'. Skipping."
                )
                continue
            try:
                pts_px = _parse_cvat_points(points_str, raw_label, elem_idx)
                if pts_px.shape[0] < 2:
                    warnings.warn(
                        f"Image '{image_id_str}', Polyline (raw_label: {raw_label}, "
                        f"xml_idx: {elem_idx}): < 2 points. Skipping."
                    )
                    continue
                pts_transformed = pixel_to_world(pts_px) if pixel_to_world else pts_px
                if pts_transformed.shape[0] >= 3 and np.allclose(
                    pts_transformed[0], pts_transformed[-1]
                ):
                    geom = shp.Polygon(pts_transformed)
                    label_total_counts[processed_label] = (
                        label_total_counts.get(processed_label, 0) + 1
                    )
                    collected_shapes_data.append(
                        (processed_label, geom, "polygon", color)
                    )
                else:
                    warnings.warn(
                        f"Image '{image_id_str}', Polyline (raw_label: {raw_label}, "
                        f"xml_idx: {elem_idx}): open or too few points for polygon. Skipping."
                    )
            except (ValueError, Exception) as e:
                warnings.warn(
                    f"Image '{image_id_str}', Polyline (raw_label: {raw_label}, "
                    f"xml_idx: {elem_idx}): error processing: {e}. Skipping."
                )
                continue

        # Points (each coordinate pair becomes a 'point' region)
        for elem_idx, points_elem in enumerate(image.findall("points")):
            raw_label_group = points_elem.get("label")  # Label for the group
            points_str = points_elem.get("points")
            # Process label and color for the group, will apply to all points from this element
            processed_label_group, color_group = get_processed_label_and_color(
                raw_label_group
            )

            if not points_str:
                warnings.warn(
                    f"Image '{image_id_str}', Points group (raw_label: {raw_label_group}, "
                    f"xml_idx: {elem_idx}): missing 'points'. Skipping."
                )
                continue
            try:
                pts_px = _parse_cvat_points(points_str, raw_label_group, elem_idx)
                if pts_px.shape[0] == 0:
                    warnings.warn(
                        f"Image '{image_id_str}', Points group (raw_label: {raw_label_group}, "
                        f"xml_idx: {elem_idx}): no valid points. Skipping."
                    )
                    continue
                pts_transformed = pixel_to_world(pts_px) if pixel_to_world else pts_px
                for pt_data in pts_transformed:  # pt_data is (2,) array
                    geom = shp.Point(pt_data)
                    label_total_counts[processed_label_group] = (
                        label_total_counts.get(processed_label_group, 0) + 1
                    )
                    collected_shapes_data.append(
                        (processed_label_group, geom, "point", color_group)
                    )
            except (ValueError, Exception) as e:
                warnings.warn(
                    f"Image '{image_id_str}', Points group (raw_label: {raw_label_group}, "
                    f"xml_idx: {elem_idx}): error processing: {e}. Skipping."
                )
                continue

        # Boxes (become polygons)
        for elem_idx, box_elem in enumerate(image.findall("box")):
            raw_label = box_elem.get("label")
            processed_label, color = get_processed_label_and_color(raw_label)
            try:
                xtl = float(box_elem.get("xtl"))
                ytl = float(box_elem.get("ytl"))
                xbr = float(box_elem.get("xbr"))
                ybr = float(box_elem.get("ybr"))
                pts_px = np.array([[xtl, ytl], [xbr, ytl], [xbr, ybr], [xtl, ybr]])
                pts_transformed = pixel_to_world(pts_px) if pixel_to_world else pts_px
                geom = shp.Polygon(pts_transformed)
            except (TypeError, ValueError, Exception) as e:
                warnings.warn(
                    f"Image '{image_id_str}', Box (raw_label: {raw_label}, "
                    f"xml_idx: {elem_idx}): error processing: {e}. Skipping."
                )
                continue

            label_total_counts[processed_label] = (
                label_total_counts.get(processed_label, 0) + 1
            )
            collected_shapes_data.append((processed_label, geom, "polygon", color))

        # Masks (RLE decoded to polygons)
        image_width_str = image.get("width")
        image_height_str = image.get("height")
        can_process_rle_for_image = False
        img_w, img_h = 0, 0
        if image_width_str and image_height_str:
            try:
                img_w, img_h = int(image_width_str), int(image_height_str)
                if img_w > 0 and img_h > 0:
                    can_process_rle_for_image = True
                else:
                    warnings.warn(
                        f"Image '{image_id_str}': non-positive width/height. Cannot process RLE."
                    )
            except ValueError:
                warnings.warn(
                    f"Image '{image_id_str}': invalid width/height. Cannot process RLE."
                )
        else:
            warnings.warn(
                f"Image '{image_id_str}': missing width/height. Cannot process RLE."
            )

        if can_process_rle_for_image:
            for elem_idx, mask_elem in enumerate(image.findall("mask")):
                raw_label = mask_elem.get("label")
                rle_str = mask_elem.get("rle")
                processed_label, color = get_processed_label_and_color(raw_label)

                if not rle_str:
                    warnings.warn(
                        f"Image '{image_id_str}', Mask (raw_label: {raw_label}, "
                        f"xml_idx: {elem_idx}): missing 'rle'. Skipping."
                    )
                    continue
                try:
                    mask_array = _rle_to_mask(rle_str, img_h, img_w)
                    mask_poly_px = _mask_to_polygon(mask_array)
                    if mask_poly_px.is_empty:
                        warnings.warn(
                            f"Image '{image_id_str}', Mask (raw_label: {raw_label}, "
                            f"xml_idx: {elem_idx}): empty polygon from mask. Skipping."
                        )
                        continue

                    final_geom: shp.Polygon
                    if pixel_to_world:
                        coords_px = np.array(mask_poly_px.exterior.coords)
                        coords_transformed = pixel_to_world(coords_px)
                        interiors_transformed = [
                            pixel_to_world(np.array(i.coords))
                            for i in mask_poly_px.interiors
                        ]
                        final_geom = shp.Polygon(
                            coords_transformed, holes=interiors_transformed
                        )
                    else:
                        final_geom = mask_poly_px
                except ImportError:  # cv2 missing
                    warnings.warn(
                        f"Image '{image_id_str}', Mask (raw_label: {raw_label}, "
                        f"xml_idx: {elem_idx}): OpenCV (cv2) missing. Skipping mask."
                    )
                    can_process_rle_for_image = (
                        False  # Stop trying masks for this image
                    )
                    break
                except (ValueError, Exception) as e:
                    warnings.warn(
                        f"Image '{image_id_str}', Mask (raw_label: {raw_label}, "
                        f"xml_idx: {elem_idx}): error processing: {e}. Skipping."
                    )
                    continue

                label_total_counts[processed_label] = (
                    label_total_counts.get(processed_label, 0) + 1
                )
                collected_shapes_data.append(
                    (processed_label, final_geom, "polygon", color)
                )

        # --- Pass 2 for current image: Generate names and create Region objects ---
        running_indices_for_multi_labels: dict[str, int] = {}
        for p_label, geom, kind, reg_color in collected_shapes_data:
            total_occurrences = label_total_counts[p_label]
            region_name: str
            if total_occurrences == 1:
                region_name = p_label
            else:
                current_idx = running_indices_for_multi_labels.get(p_label, 0)
                region_name = f"{p_label}_{current_idx}"
                running_indices_for_multi_labels[p_label] = current_idx + 1

            all_regions_in_file.append(
                _create_cvat_region(region_name, geom, kind, path_obj, reg_color)
            )

    return Regions(all_regions_in_file)


# --------------------------------------------------------------------------
# 3.  Binary mask  → single Region   (requires OpenCV)
# --------------------------------------------------------------------------
def mask_to_region(
    mask_img: NDArray[np.bool_],
    *,
    region_name: str,
    pixel_to_world: Optional[SpatialTransform] = None,
    approx_tol_px: float = 1.0,
) -> Region:
    """
    Trace the largest contour of a boolean mask into a polygon Region.

    * Requires **opencv-python**.
    * Assumes **2-D** image; Y axis is *not* flipped - handle upstream if needed.

    Parameters
    ----------
    mask_img : NDArray[np.bool_], shape (H, W)
        Binary mask image, where `True` indicates the region of interest.
    region_name : str
        Name for the resulting region.
    pixel_to_world : SpatialTransform, optional
        Callable mapping *(N,2)* **pixel** coords → centimeter coords.
        If `None`, the coordinates are assumed to be in pixels.
    approx_tol_px : float, optional
        Approximation tolerance for contour simplification in pixels.
        Default is 1.0 pixel.

    Returns
    -------
    Region
        A single :class:`Region` object representing the largest contour in the mask.
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("mask_to_region() needs opencv-python and shapely") from exc

    if mask_img.dtype != np.uint8:
        mask_img = mask_img.astype("uint8")

    cnts, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise ValueError("No contours found in mask")

    # Take the largest contour by area
    cnt = max(cnts, key=cv2.contourArea)[:, 0, :]  # (N,2) pixels
    cnt = cv2.approxPolyDP(cnt, approx_tol_px, True)[:, 0, :]

    pts_cm = pixel_to_world(cnt) if pixel_to_world else cnt
    poly = shp.Polygon(pts_cm)

    return Region(
        name=region_name, kind="polygon", data=poly, metadata={"source": "mask"}
    )


# --------------------------------------------------------------------------
# 4.  Utility: Regions → pandas DataFrame
# --------------------------------------------------------------------------
def regions_to_dataframe(regions: Regions) -> pd.DataFrame:
    """Return a *pandas* DataFrame summary.

    Parameters
    ----------
    regions : Regions
        Collection of regions to summarize.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns for region name, kind, area, and metadata.
    """

    records: list[Mapping[str, Any]] = []
    for reg in regions.values():
        rec = reg.to_dict()
        rec["area"] = reg.data.area if reg.kind == "polygon" else 0.0
        records.append(rec)

    return pd.DataFrame.from_records(records)
