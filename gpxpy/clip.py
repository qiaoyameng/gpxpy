"""
Clip GPX tracks against polygonal regions (geofences) with optional holes.

Unlike point-by-point filtering, clipping keeps the parts of a track that
pass through a region even when both endpoints of a track edge lie outside
it: edges are cut at the polygon boundary and new intersection points are
inserted. The result may therefore consist of more (and shorter) segments
than the original track.

The clip is computed in a local equirectangular (plane) projection centered
on a caller-chosen origin. This is a *local planar approximation*, not a
geodesically exact global calculation: it is only meaningful while every
polygon vertex and track point stays close to the origin (see
``DEFAULT_MAX_SPAN_M``). Inputs that span a larger area, or polygons that
are self-intersecting, are rejected with :class:`ClipError` instead of
being silently clipped with a degraded approximation.

Boundary conventions:

* Points and track edges lying exactly on the polygon boundary (including
  hole boundaries) are kept.
* Intersection points created on the boundary get their time and elevation
  linearly interpolated along the track edge they fall on. All other
  attributes (extensions, speed, ...) are left empty rather than invented.
* Original points keep all their data (including extensions); the input
  GPX objects are never modified.
"""

import copy as mod_copy
import math as mod_math

from typing import List, Optional, Sequence, Tuple

from . import geo as mod_geo
from . import gpx as mod_gpx

LatLon = Tuple[float, float]
"""A (latitude, longitude) pair in decimal degrees."""

_Point = Tuple[float, float]
"""A projected (x, y) pair in meters."""

_Ring = List[_Point]
"""A closed polygon ring in projected coordinates (last != first)."""

_PointRef = Tuple[str, int, float]
"""
Reference to an output point: ('orig', point_index, 0.0) reuses an original
track point, ('cut', edge_index, ratio) is a synthesized boundary point on
the edge between points edge_index and edge_index + 1.
"""

DEFAULT_MAX_SPAN_M: float = 100_000.0
"""
Default maximum distance (in meters) allowed between the projection origin
and any polygon vertex or track point. Beyond a few tens of kilometers the
local planar approximation used here becomes visibly inaccurate, so larger
spans are rejected unless the caller explicitly raises ``max_span_m``.
"""

_MAX_ORIGIN_LATITUDE: float = 85.0
"""The equirectangular projection degenerates near the poles."""

_CUT_TOLERANCE: float = 1e-9
"""Relative tolerance when deciding whether a cut falls on a track edge."""

_CUT_DEDUP_TOLERANCE: float = 1e-7
"""
Two cut ratios closer than this are considered the same cut (e.g. a track edge
crossing exactly through a polygon vertex is reported by both polygon edges
meeting there).
"""


class ClipError(mod_gpx.GPXException):
    """
    Raised when a track cannot be clipped: the polygon (or a hole) is
    self-intersecting or otherwise invalid, or the input spans too large an
    area around the projection origin for a local planar approximation.
    """


class _LocalProjection:
    """
    Equirectangular projection centered on an explicit origin, converting
    between (latitude, longitude) in degrees and local (x, y) in meters.
    """

    def __init__(self, origin: LatLon) -> None:
        latitude, longitude = origin
        if not -_MAX_ORIGIN_LATITUDE <= latitude <= _MAX_ORIGIN_LATITUDE:
            raise ClipError(
                f'projection origin latitude {latitude} is too close to a pole; '
                f'the local planar approximation requires |latitude| <= {_MAX_ORIGIN_LATITUDE}')
        self.latitude = latitude
        self.longitude = longitude
        self._cos_latitude = mod_math.cos(mod_math.radians(latitude))

    def to_xy(self, latitude: float, longitude: float) -> _Point:
        x = mod_math.radians(longitude - self.longitude) * self._cos_latitude * mod_geo.EARTH_RADIUS
        y = mod_math.radians(latitude - self.latitude) * mod_geo.EARTH_RADIUS
        return (x, y)

    def to_latitude_longitude(self, x: float, y: float) -> LatLon:
        latitude = self.latitude + mod_math.degrees(y / mod_geo.EARTH_RADIUS)
        longitude = self.longitude + mod_math.degrees(x / (mod_geo.EARTH_RADIUS * self._cos_latitude))
        return (latitude, longitude)


def _cross(a_xy: _Point, b_xy: _Point, c_xy: _Point) -> float:
    """Z-component of (b - a) x (c - a); sign is the turn direction from ab to ac."""
    return (b_xy[0] - a_xy[0]) * (c_xy[1] - a_xy[1]) - (b_xy[1] - a_xy[1]) * (c_xy[0] - a_xy[0])


def _point_on_segment(p_xy: _Point, a_xy: _Point, b_xy: _Point) -> bool:
    cross = _cross(a_xy, b_xy, p_xy)
    scale = (abs(b_xy[0] - a_xy[0]) + abs(b_xy[1] - a_xy[1])) \
        * (abs(p_xy[0] - a_xy[0]) + abs(p_xy[1] - a_xy[1]))
    if abs(cross) > 1e-9 * max(scale, 1.0):
        return False
    epsilon = 1e-9 * (abs(b_xy[0] - a_xy[0]) + abs(b_xy[1] - a_xy[1]) + 1.0)
    return min(a_xy[0], b_xy[0]) - epsilon <= p_xy[0] <= max(a_xy[0], b_xy[0]) + epsilon \
        and min(a_xy[1], b_xy[1]) - epsilon <= p_xy[1] <= max(a_xy[1], b_xy[1]) + epsilon


def _segments_intersect(a_xy: _Point, b_xy: _Point, c_xy: _Point, d_xy: _Point) -> bool:
    """True if segments ab and cd share at least one point (touching counts)."""
    cross_1 = _cross(c_xy, d_xy, a_xy)
    cross_2 = _cross(c_xy, d_xy, b_xy)
    cross_3 = _cross(a_xy, b_xy, c_xy)
    cross_4 = _cross(a_xy, b_xy, d_xy)
    if (cross_1 > 0) != (cross_2 > 0) and (cross_3 > 0) != (cross_4 > 0):
        return True
    if cross_1 == 0 and _point_on_segment(a_xy, c_xy, d_xy):
        return True
    if cross_2 == 0 and _point_on_segment(b_xy, c_xy, d_xy):
        return True
    if cross_3 == 0 and _point_on_segment(c_xy, a_xy, b_xy):
        return True
    if cross_4 == 0 and _point_on_segment(d_xy, a_xy, b_xy):
        return True
    return False


def _cut_ratio(p0_xy: _Point, p1_xy: _Point, a_xy: _Point, b_xy: _Point) -> Optional[float]:
    """
    Ratio along p0 -> p1 where it crosses segment a -> b, or None.

    Parallel and collinear pairs return None: a track edge running along a
    polygon edge needs no cut because boundary points count as inside.
    """
    direction_x = p1_xy[0] - p0_xy[0]
    direction_y = p1_xy[1] - p0_xy[1]
    edge_x = b_xy[0] - a_xy[0]
    edge_y = b_xy[1] - a_xy[1]
    denominator = direction_x * edge_y - direction_y * edge_x
    if denominator == 0:
        return None
    offset_x = a_xy[0] - p0_xy[0]
    offset_y = a_xy[1] - p0_xy[1]
    ratio = (offset_x * edge_y - offset_y * edge_x) / denominator
    edge_ratio = (offset_x * direction_y - offset_y * direction_x) / denominator
    if -_CUT_TOLERANCE <= ratio <= 1 + _CUT_TOLERANCE \
            and -_CUT_TOLERANCE <= edge_ratio <= 1 + _CUT_TOLERANCE:
        return min(max(ratio, 0.0), 1.0)
    return None


def _locate_in_ring(p_xy: _Point, ring: _Ring) -> int:
    """0 = outside, 1 = exactly on the boundary, 2 = strictly inside."""
    inside = False
    size = len(ring)
    for index in range(size):
        a_xy = ring[index]
        b_xy = ring[(index + 1) % size]
        if _point_on_segment(p_xy, a_xy, b_xy):
            return 1
        if (a_xy[1] > p_xy[1]) != (b_xy[1] > p_xy[1]):
            x_crossing = a_xy[0] + (p_xy[1] - a_xy[1]) * (b_xy[0] - a_xy[0]) / (b_xy[1] - a_xy[1])
            if x_crossing > p_xy[0]:
                inside = not inside
    return 2 if inside else 0


def _ring_area(ring: _Ring) -> float:
    """Signed area; positive for counter-clockwise rings."""
    area = 0.0
    size = len(ring)
    for index in range(size):
        a_xy = ring[index]
        b_xy = ring[(index + 1) % size]
        area += a_xy[0] * b_xy[1] - b_xy[0] * a_xy[1]
    return area / 2


def _validate_simple_ring(ring: _Ring, description: str) -> None:
    """Reject rings that are degenerate or self-intersecting."""
    size = len(ring)
    if size < 3:
        raise ClipError(f'{description} must have at least 3 distinct vertices')
    if _ring_area(ring) == 0:
        raise ClipError(f'{description} has zero area')
    for index in range(size):
        if ring[index] == ring[(index + 1) % size]:
            raise ClipError(f'{description} has a zero-length edge at vertex {index}')
    for first in range(size):
        a_xy = ring[first]
        b_xy = ring[(first + 1) % size]
        for second in range(first + 1, size):
            c_xy = ring[second]
            d_xy = ring[(second + 1) % size]
            if second == first + 1 or (first == 0 and second == size - 1):
                # Adjacent edges share one vertex; folding back onto the
                # shared edge would make the ring touch itself.
                if _cross(a_xy, b_xy, d_xy) == 0 and _cross(a_xy, b_xy, c_xy) == 0 \
                        and (_point_on_segment(d_xy, a_xy, b_xy) or _point_on_segment(a_xy, c_xy, d_xy)):
                    raise ClipError(f'{description} folds back onto itself at vertex {(first + 1) % size}')
                continue
            if _segments_intersect(a_xy, b_xy, c_xy, d_xy):
                raise ClipError(f'{description} is self-intersecting (edges {first} and {second} cross)')


def _project_ring(ring: Sequence[LatLon], description: str,
                  projection: _LocalProjection, max_span_m: float) -> _Ring:
    vertices = list(ring)
    if len(vertices) > 1 and vertices[0] == vertices[-1]:
        vertices = vertices[:-1]
    if len(vertices) < 3:
        raise ClipError(f'{description} must have at least 3 distinct vertices')
    projected: List[_Point] = []
    for latitude, longitude in vertices:
        x, y = projection.to_xy(latitude, longitude)
        if mod_math.hypot(x, y) > max_span_m:
            raise ClipError(
                f'{description} vertex ({latitude}, {longitude}) is more than {max_span_m} m '
                f'from the projection origin; the local planar approximation is not valid '
                f'over such a span (choose a closer origin or raise max_span_m)')
        projected.append((x, y))
    return projected


def _prepare_rings(polygon: Sequence[LatLon], holes: Optional[Sequence[Sequence[LatLon]]],
                   projection: _LocalProjection, max_span_m: float) -> Tuple[_Ring, List[_Ring]]:
    """Project and validate the outer ring and all holes."""
    outer = _project_ring(polygon, 'polygon', projection, max_span_m)
    _validate_simple_ring(outer, 'polygon')
    hole_rings: List[_Ring] = []
    for hole_no, hole in enumerate(holes or []):
        description = f'hole {hole_no}'
        hole_ring = _project_ring(hole, description, projection, max_span_m)
        _validate_simple_ring(hole_ring, description)
        hole_rings.append(hole_ring)
    for hole_no, hole_ring in enumerate(hole_rings):
        for index in range(len(hole_ring)):
            a_xy = hole_ring[index]
            b_xy = hole_ring[(index + 1) % len(hole_ring)]
            for outer_index in range(len(outer)):
                c_xy = outer[outer_index]
                d_xy = outer[(outer_index + 1) % len(outer)]
                if _segments_intersect(a_xy, b_xy, c_xy, d_xy):
                    raise ClipError(f'hole {hole_no} touches or crosses the polygon boundary')
        if _locate_in_ring(hole_ring[0], outer) != 2:
            raise ClipError(f'hole {hole_no} is not strictly inside the polygon')
    for first in range(len(hole_rings)):
        for second in range(first + 1, len(hole_rings)):
            first_ring = hole_rings[first]
            second_ring = hole_rings[second]
            for index in range(len(first_ring)):
                a_xy = first_ring[index]
                b_xy = first_ring[(index + 1) % len(first_ring)]
                for second_index in range(len(second_ring)):
                    c_xy = second_ring[second_index]
                    d_xy = second_ring[(second_index + 1) % len(second_ring)]
                    if _segments_intersect(a_xy, b_xy, c_xy, d_xy):
                        raise ClipError(f'holes {first} and {second} touch or cross each other')
            if _locate_in_ring(first_ring[0], second_ring) == 2 \
                    or _locate_in_ring(second_ring[0], first_ring) == 2:
                raise ClipError(f'holes {first} and {second} are nested')
    return outer, hole_rings


def _contains(outer: _Ring, holes: List[_Ring], p_xy: _Point) -> bool:
    """Inside the polygon: inside the outer ring and not strictly inside a hole."""
    if _locate_in_ring(p_xy, outer) == 0:
        return False
    for hole in holes:
        if _locate_in_ring(p_xy, hole) == 2:
            return False
    return True


def _edge_cut_ratios(p0_xy: _Point, p1_xy: _Point, outer: _Ring, holes: List[_Ring]) -> List[float]:
    """Sorted, deduplicated cut ratios (including 0 and 1) along p0 -> p1."""
    ratios = [0.0, 1.0]
    for ring in [outer] + holes:
        size = len(ring)
        for index in range(size):
            ratio = _cut_ratio(p0_xy, p1_xy, ring[index], ring[(index + 1) % size])
            if ratio is not None and _CUT_TOLERANCE < ratio < 1 - _CUT_TOLERANCE:
                ratios.append(ratio)
    ratios.sort()
    deduplicated = [ratios[0]]
    for ratio in ratios[1:]:
        if ratio - deduplicated[-1] > _CUT_DEDUP_TOLERANCE:
            deduplicated.append(ratio)
    return deduplicated


def _point_ref(edge_index: int, ratio: float) -> _PointRef:
    if ratio <= 0.0:
        return ('orig', edge_index, 0.0)
    if ratio >= 1.0:
        return ('orig', edge_index + 1, 0.0)
    return ('cut', edge_index, ratio)


def _inside_runs(xy: List[_Point], outer: _Ring, holes: List[_Ring]) -> List[List[_PointRef]]:
    """
    Split the polyline into maximal runs of point references that lie inside
    the polygon. Every outside interval starts a new run, so a track crossing
    a hole (or leaving and re-entering the polygon) yields several runs.
    """
    runs: List[List[_PointRef]] = []
    current: List[_PointRef] = []
    last_ref: Optional[_PointRef] = None

    def emit(ref: _PointRef) -> None:
        nonlocal last_ref
        if current and last_ref == ref:
            return
        current.append(ref)
        last_ref = ref

    def close() -> None:
        nonlocal current, last_ref
        if current:
            runs.append(current)
        current = []
        last_ref = None

    if len(xy) == 1:
        if _contains(outer, holes, xy[0]):
            runs.append([('orig', 0, 0.0)])
        return runs

    for edge_index in range(len(xy) - 1):
        p0_xy = xy[edge_index]
        p1_xy = xy[edge_index + 1]
        ratios = _edge_cut_ratios(p0_xy, p1_xy, outer, holes)
        for start, end in zip(ratios, ratios[1:]):
            middle_ratio = (start + end) / 2
            middle = (p0_xy[0] + (p1_xy[0] - p0_xy[0]) * middle_ratio,
                      p0_xy[1] + (p1_xy[1] - p0_xy[1]) * middle_ratio)
            if _contains(outer, holes, middle):
                emit(_point_ref(edge_index, start))
                emit(_point_ref(edge_index, end))
            else:
                close()
    close()
    return runs


def _materialize(ref: _PointRef, segment: mod_gpx.GPXTrackSegment,
                 xy: List[_Point], projection: _LocalProjection) -> mod_gpx.GPXTrackPoint:
    kind, index, ratio = ref
    if kind == 'orig':
        # Deepcopy so the caller can mutate the result without affecting the
        # input; all original data (including extensions) is preserved.
        return mod_copy.deepcopy(segment.points[index])
    first = segment.points[index]
    second = segment.points[index + 1]
    x = xy[index][0] + (xy[index + 1][0] - xy[index][0]) * ratio
    y = xy[index][1] + (xy[index + 1][1] - xy[index][1]) * ratio
    latitude, longitude = projection.to_latitude_longitude(x, y)
    elevation = None
    if first.elevation is not None and second.elevation is not None:
        elevation = first.elevation + (second.elevation - first.elevation) * ratio
    time = None
    if first.time is not None and second.time is not None:
        time = first.time + (second.time - first.time) * ratio
    # Synthesized boundary points carry no extensions or other metadata:
    # there is no defensible way to invent them for an interpolated position.
    return mod_gpx.GPXTrackPoint(latitude=latitude, longitude=longitude,
                                 elevation=elevation, time=time)


def _clip_segment(segment: mod_gpx.GPXTrackSegment, projection: _LocalProjection,
                  outer: _Ring, holes: List[_Ring], max_span_m: float) -> List[mod_gpx.GPXTrackSegment]:
    xy: List[_Point] = []
    for point in segment.points:
        x, y = projection.to_xy(point.latitude, point.longitude)
        if mod_math.hypot(x, y) > max_span_m:
            raise ClipError(
                f'track point ({point.latitude}, {point.longitude}) is more than {max_span_m} m '
                f'from the projection origin; the local planar approximation is not valid '
                f'over such a span (choose a closer origin or raise max_span_m)')
        xy.append((x, y))
    result: List[mod_gpx.GPXTrackSegment] = []
    for run in _inside_runs(xy, outer, holes):
        clipped_segment = mod_gpx.GPXTrackSegment()
        clipped_segment.extensions = mod_copy.deepcopy(segment.extensions)
        clipped_segment.points = [_materialize(ref, segment, xy, projection) for ref in run]
        result.append(clipped_segment)
    return result


def clip_segment_to_polygon(segment: mod_gpx.GPXTrackSegment, polygon: Sequence[LatLon],
                            holes: Optional[Sequence[Sequence[LatLon]]]=None, *,
                            origin: LatLon,
                            max_span_m: float=DEFAULT_MAX_SPAN_M) -> List[mod_gpx.GPXTrackSegment]:
    """
    Clip a single track segment against a polygon (with optional holes).

    Parameters
    ----------
    segment : GPXTrackSegment
        The segment to clip. It is not modified.
    polygon : sequence of (latitude, longitude)
        The outer ring of the region to keep. May be given in either winding
        order; a repeated first vertex at the end is ignored.
    holes : sequence of rings, optional
        Rings (same format as ``polygon``) describing holes inside the
        region. Each hole must be simple and strictly inside the polygon,
        and holes must not touch, cross or contain each other.
    origin : (latitude, longitude)
        Required. Origin of the local planar projection used for the clip;
        choose it inside (or next to) the region of interest.
    max_span_m : float
        Maximum allowed distance in meters between ``origin`` and any
        polygon vertex or track point. Larger spans are rejected because
        the local planar approximation would not be trustworthy.

    Returns
    -------
    list of GPXTrackSegment
        Zero or more new segments covering the parts of ``segment`` inside
        the region. Original points are deep copies with all their data;
        points synthesized on the boundary have interpolated time and
        elevation only.
    """
    projection = _LocalProjection(origin)
    outer, hole_rings = _prepare_rings(polygon, holes, projection, max_span_m)
    return _clip_segment(segment, projection, outer, hole_rings, max_span_m)


def clip_track_to_polygon(track: mod_gpx.GPXTrack, polygon: Sequence[LatLon],
                          holes: Optional[Sequence[Sequence[LatLon]]]=None, *,
                          origin: LatLon,
                          max_span_m: float=DEFAULT_MAX_SPAN_M) -> mod_gpx.GPXTrack:
    """
    Clip every segment of a track against a polygon (with optional holes).

    Returns a new GPXTrack with the same metadata as ``track`` whose
    segments are the clipped parts of the original segments (a segment that
    leaves and re-enters the region produces several segments). The input
    track is not modified. See :func:`clip_segment_to_polygon` for the
    meaning of the other parameters.
    """
    projection = _LocalProjection(origin)
    outer, hole_rings = _prepare_rings(polygon, holes, projection, max_span_m)
    clipped_track = mod_gpx.GPXTrack(name=track.name, description=track.description,
                                     number=track.number)
    clipped_track.comment = track.comment
    clipped_track.source = track.source
    clipped_track.link = track.link
    clipped_track.link_text = track.link_text
    clipped_track.link_type = track.link_type
    clipped_track.type = track.type
    clipped_track.extensions = mod_copy.deepcopy(track.extensions)
    for segment in track.segments:
        clipped_track.segments.extend(_clip_segment(segment, projection, outer, hole_rings, max_span_m))
    return clipped_track
