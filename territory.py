"""Local image territories and operator-confirmed planar restrictions.

Image analysis is deliberately a heuristic proposal, never a flight restriction.
Coordinates are normalized image coordinates: x right, y down. The simulator uses
the same abstract local plane; images do not establish geographic coordinates.
"""
from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import re
import threading
from typing import Annotated, Literal
import uuid
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import ndimage

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAP_ID = re.compile(r"^[a-f0-9]{12}$")
ANALYSIS_WARNING = ("Эвристика по цвету и контрасту, не распознавание объектов. "
                    "Проверьте кандидатов и подтвердите запретные зоны вручную.")


class BoundedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MapUpload(BoundedModel):
    name: str = Field(min_length=1, max_length=100)
    image_data_url: str = Field(min_length=30, max_length=6_000_000)
    width_m: float = Field(default=80, ge=10, le=10000)
    height_m: float = Field(default=60, ge=10, le=10000)


class MapAnalysis(BoundedModel):
    method: Literal["dark", "water", "contrast"] = "dark"
    sensitivity: float = Field(default=.5, ge=0, le=1)


class MapCalibration(BoundedModel):
    width_m: float = Field(ge=10, le=10000)
    height_m: float = Field(ge=10, le=10000)


class Point(BoundedModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class RectangleZone(Point):
    kind: Literal["rectangle"]
    id: str | None = Field(default=None, max_length=64)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def bounded(self):
        if self.x + self.width > 1 + 1e-9 or self.y + self.height > 1 + 1e-9:
            raise ValueError("Rectangle must fit inside the territory")
        return self


class CircleZone(Point):
    kind: Literal["circle"]
    id: str | None = Field(default=None, max_length=64)
    radius: float = Field(gt=0, le=.5)

def _orientation(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a, b, c, d):
    # Includes touching and collinear edges, needed for conservative cell coverage.
    o = [_orientation(a, b, c), _orientation(a, b, d),
         _orientation(c, d, a), _orientation(c, d, b)]
    if o[0] * o[1] < 0 and o[2] * o[3] < 0:
        return True
    return any(abs(v) < 1e-12 and min(p[0], q[0]) - 1e-12 <= r[0] <= max(p[0], q[0]) + 1e-12
               and min(p[1], q[1]) - 1e-12 <= r[1] <= max(p[1], q[1]) + 1e-12
               for v, p, q, r in ((o[0], a, b, c), (o[1], a, b, d),
                                  (o[2], c, d, a), (o[3], c, d, b)))


class PolygonZone(BoundedModel):
    kind: Literal["polygon"]
    id: str | None = Field(default=None, max_length=64)
    points: list[Point] = Field(min_length=3, max_length=64)

    @model_validator(mode="after")
    def simple_polygon(self):
        points = [(p.x, p.y) for p in self.points]
        if len(set(points)) != len(points):
            raise ValueError("Polygon vertices must be distinct; do not repeat the first vertex")
        area = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))) / 2
        if area < 1e-8:
            raise ValueError("Polygon area must be nonzero")
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue
                if _segments_intersect(points[i], points[(i + 1) % n], points[j], points[(j + 1) % n]):
                    raise ValueError("Polygon edges must not intersect")
        return self


RestrictedZone = Annotated[RectangleZone | CircleZone | PolygonZone, Field(discriminator="kind")]


def _inside_polygon(point, vertices):
    x, y = point
    inside = False
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
            inside = not inside
    return inside


def restricted_mask(zones: list[dict], n=24, width=80., height=60.) -> np.ndarray:
    """Conservatively block every grid cell touched by a confirmed shape."""
    mask = np.zeros((n, n), dtype=bool)
    for zone in zones:
        kind = zone["kind"]
        vertices = [(p["x"], p["y"]) for p in zone.get("points", [])]
        for row in range(n):
            for col in range(n):
                x0, y0, x1, y1 = col/n, row/n, (col+1)/n, (row+1)/n
                if kind == "rectangle":
                    hit = (x0 < zone["x"] + zone["width"] - 1e-12 and x1 > zone["x"] + 1e-12
                           and y0 < zone["y"] + zone["height"] - 1e-12 and y1 > zone["y"] + 1e-12)
                elif kind == "circle":
                    cx = np.clip(zone["x"], x0, x1)
                    cy = np.clip(zone["y"], y0, y1)
                    hit = (((cx-zone["x"])*width)**2 + ((cy-zone["y"])*height)**2
                           <= (zone["radius"]*min(width, height))**2 + 1e-14)
                else:
                    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                    hit = (any(x0 <= x <= x1 and y0 <= y <= y1 for x, y in vertices)
                           or any(_inside_polygon(corner, vertices) for corner in corners)
                           or any(_segments_intersect(a, b, c, d)
                                  for a, b in zip(vertices, vertices[1:] + vertices[:1])
                                  for c, d in zip(corners, corners[1:] + corners[:1])))
                mask[row, col] |= bool(hit)
    return mask


def mask_rectangles(mask: np.ndarray, width: float, height: float) -> list[tuple]:
    """Lossless cover of blocked cells, merging identical row runs vertically."""
    n, m = mask.shape
    finished, active = [], {}
    for row in range(n + 1):
        runs = []
        if row < n:
            start = None
            for col in range(m + 1):
                occupied = col < m and mask[row, col]
                if occupied and start is None:
                    start = col
                elif not occupied and start is not None:
                    runs.append((start, col))
                    start = None
        current = set(runs)
        for run in list(active):
            if run not in current:
                first = active.pop(run)
                finished.append((run[0]/m*width, first/n*height, run[1]/m*width, row/n*height))
        for run in runs:
            active.setdefault(run, row)
    if len(finished) > 256:
        raise ValueError("Restrictions exceed the 256 rectangle planning budget; simplify the shapes")
    return finished


def free_slots(slots: np.ndarray, rectangles: list, width: float, height: float, n=24) -> tuple[np.ndarray, int]:
    """Project blocked target slots to the nearest safe grid centre explicitly."""
    def free(point):
        return (.15 <= point[0] <= width-.15 and .15 <= point[1] <= height-.15
                and not any(x0-.16 <= point[0] <= x1+.16 and y0-.16 <= point[1] <= y1+.16
                            for x0, y0, x1, y1 in rectangles))
    candidates = np.asarray([((x+.5)*width/n, (y+.5)*height/n)
                             for y in range(n) for x in range(n)])
    candidates = candidates[[free(p) for p in candidates]]
    if not len(candidates):
        raise ValueError("Restrictions leave no safe launch or target cell")
    result, moved = slots.copy(), 0
    for index, point in enumerate(result):
        if not free(point):
            result[index] = candidates[np.argmin(np.linalg.norm(candidates-point, axis=1))]
            moved += 1
    return result, moved


def initial_free_positions(count: int, rectangles: list, width: float, height: float, seed: int) -> np.ndarray:
    """Deterministic launch points on available cells, preferring the left edge."""
    candidates = np.asarray([((x+.5)*width/24, (y+.5)*height/24)
                             for x in range(24) for y in range(24)])
    safe = [p for p in candidates if not any(x0-.16 <= p[0] <= x1+.16 and y0-.16 <= p[1] <= y1+.16
                                            for x0, y0, x1, y1 in rectangles)]
    if not safe:
        raise ValueError("Restrictions leave no safe launch cells")
    # Cycling is only needed for densely restricted maps. Duplicate launch points
    # would be misleading, so reject instead of silently violating placement.
    if len(safe) < count:
        raise ValueError(f"Only {len(safe)} safe launch cells remain for {count} vehicles")
    return np.asarray(safe[:count], dtype=float)


def analyze_image(image: Image.Image, method="dark", sensitivity=.5) -> tuple[list, dict]:
    sample = image.copy()
    sample.thumbnail((256, 256))
    rgb = np.asarray(sample.convert("RGB"), dtype=float) / 255
    gray = rgb @ np.asarray([.2126, .7152, .0722])
    if method == "water":
        red, green, blue = np.moveaxis(rgb, -1, 0)
        selected = (blue > red + .10 - .08*sensitivity) & (green > red + .025) & (blue > .22)
    elif method == "contrast":
        selected = np.abs(gray - ndimage.gaussian_filter(gray, 4)) > .18 - .13*sensitivity
    else:
        selected = gray < .12 + .42*sensitivity
    selected = ndimage.binary_opening(selected, iterations=1)
    labels, count = ndimage.label(selected)
    sizes = np.bincount(labels.ravel())
    candidates = []
    for label, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None or sizes[label] < max(5, gray.size*.001) or sizes[label] > gray.size*.65:
            continue
        sy, sx = box
        h, w = gray.shape
        candidates.append({"id": f"candidate-{label}", "kind": "rectangle",
                           "x": sx.start/w, "y": sy.start/h,
                           "width": (sx.stop-sx.start)/w, "height": (sy.stop-sy.start)/h,
                           "source": "automatic", "method": method, "confidence": None})
    candidates.sort(key=lambda z: z["width"]*z["height"], reverse=True)
    return candidates[:20], {"method": method, "sensitivity": sensitivity,
                             "warning": ANALYSIS_WARNING, "requires_confirmation": True,
                             "candidate_count": min(20, len(candidates)), "recognized_objects": False}


class MapStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def get(self, identifier: str) -> dict:
        if not MAP_ID.fullmatch(identifier):
            raise FileNotFoundError("Territory not found")
        path = self.directory / f"{identifier}.json"
        if not path.is_file() or path.stat().st_size > 256_000:
            raise FileNotFoundError("Territory not found")
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("id") != identifier or not self.image_path(identifier).is_file():
                raise ValueError("Invalid stored territory")
            return record
        except (ValueError, OSError) as error:
            raise FileNotFoundError("Territory not found") from error

    def image_path(self, identifier: str) -> Path:
        if not MAP_ID.fullmatch(identifier):
            raise FileNotFoundError("Territory not found")
        return self.directory / f"{identifier}.png"

    def list(self) -> list[dict]:
        records = []
        for path in sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:100]:
            try:
                records.append(self.get(path.stem))
            except FileNotFoundError:
                continue
        return records

    def _persist(self, record: dict):
        destination = self.directory / f"{record['id']}.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(destination)

    def create(self, upload: MapUpload) -> dict:
        match = re.fullmatch(r"data:image/(png|jpeg|webp);base64,([A-Za-z0-9+/=]+)", upload.image_data_url)
        if not match:
            raise ValueError("Upload a PNG, JPEG or WEBP image as a base64 data URL")
        encoded = match[2]
        if len(encoded) > 4 * math.ceil(MAX_IMAGE_BYTES / 3):
            raise ValueError("Image exceeds the 4 MiB decoded limit")
        try:
            data = base64.b64decode(encoded, validate=True)
        except binascii.Error as error:
            raise ValueError("Invalid base64 image") from error
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("Image exceeds the 4 MiB decoded limit")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as source:
                    expected = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}[match[1]]
                    if source.format != expected:
                        raise ValueError("Image content does not match its MIME type")
                    if source.width * source.height > 16_777_216 or max(source.size) > 8192:
                        raise ValueError("Image dimensions exceed the bounded 16 megapixel input")
                    source.load()
                    oriented = ImageOps.exif_transpose(source)
                    if "A" in oriented.getbands() or "transparency" in oriented.info:
                        rgba = oriented.convert("RGBA")
                        backdrop = Image.new("RGBA", rgba.size, (245, 244, 235, 255))
                        image = Image.alpha_composite(backdrop, rgba).convert("RGB")
                    else:
                        image = oriented.convert("RGB")
                    image.thumbnail((4096, 4096))
                    sanitized = io.BytesIO()
                    image.save(sanitized, format="PNG", optimize=True)
                    if sanitized.tell() > 16 * 1024 * 1024:
                        image.thumbnail((1600, 1600))
                        sanitized = io.BytesIO()
                        image.save(sanitized, format="PNG", optimize=True)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            raise ValueError("Unreadable or unsafe image; upload a valid PNG, JPEG or WEBP") from error
        suggestions, analysis = analyze_image(image)
        identifier = uuid.uuid4().hex[:12]
        record = {"id": identifier, "name": upload.name.strip() or "Территория",
                  "width_m": upload.width_m, "height_m": upload.height_m,
                  "pixel_width": image.width, "pixel_height": image.height,
                  "image_url": f"/api/maps/{identifier}/image", "suggestions": suggestions,
                  "analysis": analysis, "coordinate_system": "image_normalized_x_right_y_down",
                  "created_at": datetime.now(timezone.utc).isoformat()}
        with self.lock:
            if len(list(self.directory.glob("*.json"))) >= 100:
                raise ValueError("At most 100 uploaded territories may be stored")
            path = self.image_path(identifier)
            path.write_bytes(sanitized.getvalue())
            try:
                self._persist(record)
            except OSError:
                path.unlink(missing_ok=True)
                raise
        return record

    def analyze(self, identifier: str, settings: MapAnalysis) -> dict:
        with self.lock:
            record = self.get(identifier)
            with Image.open(self.image_path(identifier)) as image:
                record["suggestions"], record["analysis"] = analyze_image(image, settings.method, settings.sensitivity)
            self._persist(record)
            return record

    def calibrate(self, identifier: str, dimensions: MapCalibration) -> dict:
        with self.lock:
            record = self.get(identifier)
            record.update(dimensions.model_dump())
            self._persist(record)
            return record
