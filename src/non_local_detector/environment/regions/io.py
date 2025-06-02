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
from typing import Any, Mapping

import numpy as np
import shapely.geometry as shp
from numpy.typing import NDArray

from ..transforms import SpatialTransform
from .core import Region, Regions

# --------------------------------------------------------------------------
# 1.  Generic JSON <--> Regions round-trip
# --------------------------------------------------------------------------
_SCHEMA_TAG = "Regions-v1"


def regions_to_json(regions: Regions, path: str | Path, *, indent: int = 2) -> None:
    """Write a list-of-dicts file that any language can read."""
    payload = {
        "format": _SCHEMA_TAG,
        "regions": [reg.to_dict() for reg in regions.values()],
    }
    Path(path).write_text(json.dumps(payload, indent=indent))


def regions_from_json(path: str | Path) -> Regions:
    """Load the schema written by :func:`regions_to_json`."""
    blob = json.loads(Path(path).read_text())
    if blob.get("format") != _SCHEMA_TAG:
        warnings.warn(
            f"Unrecognised format tag {blob.get('format')!r}; attempting best-effort load"
        )
    return Regions(Region.from_dict(d) for d in blob["regions"])


# --------------------------------------------------------------------------
# 2.  LabelMe / CVAT / VIA-style polygon JSON  → Regions
# --------------------------------------------------------------------------
def load_labelme_json(
    json_path: str | Path,
    *,
    pixel_to_world: SpatialTransform | None = None,
    label_key: str = "label",
    points_key: str = "points",
) -> Regions:
    """
    Parse a *.json* file produced by many point-&-click ROI tools.

    Parameters
    ----------
    pixel_to_world
        Callable mapping *(N,2)* **pixel** coords → centimetre coords.
        Pass ``None`` if the file is *already* in cm.
    label_key, points_key
        Keys used by the specific flavour (LabelMe default shown).

    Returns
    -------
    Regions
        Every polygon becomes a :class:`Region` with ``kind="polygon"``.
    """
    data = json.loads(Path(json_path).read_text())
    regions: list[Region] = []

    # LabelMe puts shapes in a top-level list; CVAT stores under "shapes"
    shapes = data.get("shapes", data if isinstance(data, list) else [])
    if not shapes:
        raise ValueError(f"No shapes found in {json_path}")

    for obj in shapes:
        name = obj[label_key]
        pts_px = np.asarray(obj[points_key], dtype=float)  # (M,2)

        pts_cm = pixel_to_world(pts_px) if pixel_to_world else pts_px
        poly = shp.Polygon(pts_cm)

        regions.append(
            Region(
                name=name,
                kind="polygon",
                data=poly,
                metadata={"source_json": Path(json_path).name},
            )
        )
    return Regions(regions)


def _parse_cvat_points(points_str: str) -> np.ndarray:
    points = []
    for pt in points_str.strip().split(";"):
        x_str, y_str = pt.split(",")
        points.append([float(x_str), float(y_str)])
    return np.array(points)


def load_cvat_xml(xml_path: str | Path, *, pixel_to_world=None) -> Regions:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    regions = []

    # Keep track of label counts to generate unique names
    label_counts = {}

    def unique_name(label: str) -> str:
        count = label_counts.get(label, 0)
        label_counts[label] = count + 1
        if count == 0:
            return label
        else:
            return f"{label}_{count}"

    for image in root.findall("image"):
        # Polygon shapes
        for polygon in image.findall("polygon"):
            label = polygon.get("label")
            points_str = polygon.get("points")
            color = polygon.get("color")  # CVAT color
            if not points_str:
                continue
            pts_px = _parse_cvat_points(points_str)
            pts_cm = pixel_to_world(pts_px) if pixel_to_world else pts_px
            poly = shp.Polygon(pts_cm)
            poly = shp.polygon.orient(poly, sign=1.0)
            # Skip polygons that are still invalid or empty
            if not poly.is_valid or poly.area == 0:
                # Try to fix invalid polygon by buffering zero
                poly = poly.buffer(0)
            regions.append(
                Region(
                    name=unique_name(label),
                    kind="polygon",
                    data=poly,
                    metadata={
                        "source_xml": Path(xml_path).name,
                        "plot_kwargs": dict(color=color) if color else {},
                    },
                )
            )

        # Polyline shapes
        for polyline in image.findall("polyline"):
            label = polyline.get("label")
            points_str = polyline.get("points")
            color = polyline.get("color")  # CVAT color
            if not points_str:
                continue
            pts_px = _parse_cvat_points(points_str)
            pts_cm = pixel_to_world(pts_px) if pixel_to_world else pts_px
            if np.allclose(pts_cm[0], pts_cm[-1]):
                poly = shp.Polygon(pts_cm)
                kind = "polygon"
            else:
                continue
            regions.append(
                Region(
                    name=unique_name(label),
                    kind=kind,
                    data=poly,
                    metadata={
                        "source_xml": Path(xml_path).name,
                        "plot_kwargs": dict(color=color) if color else {},
                    },
                )
            )

        # Points shapes
        for points in image.findall("points"):
            label = points.get("label")
            points_str = points.get("points")
            if not points_str:
                continue
            pts_px = _parse_cvat_points(points_str)
            pts_cm = pixel_to_world(pts_px) if pixel_to_world else pts_px
            for i, pt in enumerate(pts_cm):
                regions.append(
                    Region(
                        name=unique_name(f"{label}_{i}"),
                        kind="point",
                        data=pt,
                        metadata={"source_xml": Path(xml_path).name},
                    )
                )

        # Box shapes
        for box in image.findall("box"):
            label = box.get("label")
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))
            pts_px = np.array(
                [
                    [xtl, ytl],
                    [xbr, ytl],
                    [xbr, ybr],
                    [xtl, ybr],
                    [xtl, ytl],
                ]
            )
            pts_cm = pixel_to_world(pts_px) if pixel_to_world else pts_px
            poly = shp.Polygon(pts_cm)
            regions.append(
                Region(
                    name=unique_name(label),
                    kind="polygon",
                    data=poly,
                    metadata={"source_xml": Path(xml_path).name},
                )
            )

        # Skip masks for now
        for mask in image.findall("mask"):
            continue

    return Regions(regions)


# --------------------------------------------------------------------------
# 3.  Binary mask  → single Region   (requires OpenCV)
# --------------------------------------------------------------------------
def mask_to_region(
    mask_img: NDArray[np.bool_],
    *,
    region_name: str,
    pixel_to_world: SpatialTransform | None = None,
    approx_tol_px: float = 1.0,
) -> Region:
    """
    Trace the largest contour of a boolean mask into a polygon Region.

    * Requires **opencv-python**.
    * Assumes **2-D** image; Y axis is *not* flipped – handle upstream if needed.
    """
    try:
        import cv2  # heavy import
        import shapely.geometry as shp
    except ModuleNotFoundError as exc:
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
# 4.  Utility: Regions → pandas DataFrame  (optional dependency)
# --------------------------------------------------------------------------
def regions_to_dataframe(regions: Regions):
    """Return a *pandas* DataFrame summary (no heavy deps in caller)."""
    import pandas as pd  # optional import

    records: list[Mapping[str, Any]] = []
    for reg in regions.values():
        rec = reg.to_dict()
        rec["area"] = reg.data.area if reg.kind == "polygon" else 0.0
        records.append(rec)

    return pd.DataFrame.from_records(records)
