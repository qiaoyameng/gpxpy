"""
Clip tracks against a planar polygon.

The clipping happens in a *local equirectangular projection* chosen by the
caller through :class:`LocalProjection`.  The projection is a simple flat
approximation that is accurate only within a limited area around the base
point; it is **not** a globally exact geodesic computation.  The polygon and
the track are rejected with :class:`ClipError` when they span too large an
area for the approximation to be trustworthy.

The polygon consists of one simple (non-self-intersecting) exterior ring and
zero or more simple, non-intersecting holes.  Self-intersecting input rings,
holes outside the exterior ring or rings that touch or intersect are rejected.

A track polyline is cut at every point where it enters or leaves the region.
Portions inside the region -- including paths running along the exterior ring
or along a hole boundary -- are returned, possibly as several independent
:class:`gpxpy.gpx.GPXTrackSegment` objects.  Newly created intersection points
get their time and elevation interpolated linearly along the intersected
track edge; every other attribute (extensions, speed, course, ...) is left at
its default empty value instead of being invented.  Original points are
reused unchanged and therefore keep all of their data.
"""

import copy as mod_copy
import datetime as mod_datetime
import math as mod_math
from typing import (
    TYPE_CHECKING,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from . import geo as mod_geo
    from . import gpx as mod_gpx

# WGS84 semi-major axis, meters (kept local so this module is self contained):
_EARTH_RADIUS = 6378.137 * 1000

# Coincidence tolerances in the *projected* plane (meters):
_EPS = 1e-7
_EPS_SCALE = 1e-10
_EPS_T = 1e-12

# Coincidence tolerance when normalizing rings given in lat/lon (degrees):
_LL_EPS = 1e-12

# Default maximum distance from the projection base point, meters.  Beyond
# this the equirectangular approximation is no longer offered as accurate.
DEFAULT_MAX_EXTENT = 50_000.0

# The planar projection is meaningless near the poles:
_MAX_BASE_LATITUDE = 85.0

XY = Tuple[float, float]
LatLon = Tuple[float, float]
PointLike = Union["mod_gpx.GPXTrackPoint", "mod_geo.Location", LatLon]
RingLike = Sequence[PointLike]


class ClipError(Exception):
    """Raised when a track cannot be clipped as requested.

    This covers polygons that are self-intersecting or otherwise invalid for
    clipping, and data that spans too large an area for the local planar
    approximation.
    """


class LocalProjection:
    """A local equirectangular projection centered on a base point.

    The caller must explicitly choose the base point
    (``base_latitude``/``base_longitude``), preferably inside or near the
    polygon.  Coordinates are mapped to meters with::

        x = R * cos(lat0) * (lon - lon0)
        y = R * (lat - lat0)

    This is a planar *approximation* suitable for a limited local area only;
    it is not a globally exact calculation.  ``max_extent`` (default
    :data:`DEFAULT_MAX_EXTENT`, 50 km) declares how far from the base point,
    in meters, data may be; clipping refuses to process anything farther.
    """

    def __init__(self, base_latitude: float, base_longitude: float,
                 max_extent: float = DEFAULT_MAX_EXTENT) -> None:
        for name, value in (('base_latitude', base_latitude),
                            ('base_longitude', base_longitude),
                            ('max_extent', max_extent)):
            if not _is_finite(value):
                raise ClipError(f'{name} must be a finite number')
        if not -_MAX_BASE_LATITUDE <= base_latitude <= _MAX_BASE_LATITUDE:
            raise ClipError(
                f'base_latitude must be within +-{_MAX_BASE_LATITUDE} degrees; '
                'the planar projection is not defined near the poles')
        if max_extent <= 0:
            raise ClipError('max_extent must be positive')
        self.base_latitude = base_latitude
        self.base_longitude = base_longitude
        self.max_extent = max_extent
        self._lat0 = mod_math.radians(base_latitude)
        self._lon0 = mod_math.radians(base_longitude)
        self._cos_lat0 = mod_math.cos(self._lat0)

    def project(self, latitude: float, longitude: float) -> XY:
        """Project a lat/lon point to ``(x, y)`` meters from the base point."""
        x = _EARTH_RADIUS * self._cos_lat0 * (mod_math.radians(longitude) - self._lon0)
        y = _EARTH_RADIUS * (mod_math.radians(latitude) - self._lat0)
        return x, y

    def unproject(self, x: float, y: float) -> LatLon:
        """Inverse of :meth:`project`: ``(x, y)`` meters back to lat/lon."""
        latitude = mod_math.degrees(y / _EARTH_RADIUS + self._lat0)
        longitude = mod_math.degrees(
            x / (_EARTH_RADIUS * self._cos_lat0) + self._lon0)
        return latitude, longitude

    def check_extent(self, x: float, y: float) -> None:
        """Raise :class:`ClipError` if a projected point is too far away."""
        distance = mod_math.hypot(x, y)
        if distance > self.max_extent + _EPS:
            raise ClipError(
                f'data is {distance:.1f} m from the projection base point; '
                f'exceeds max_extent={self.max_extent:.1f} m. The local planar '
                'approximation is not valid for such spans; choose a closer '
                'base point or raise max_extent consciously.')


class Polygon:
    """A simple planar polygon with optional holes, given in lat/lon.

    ``exterior`` is the outer ring; ``holes`` is a list of zero or more inner
    rings.  Each ring is a sequence of ``(latitude, longitude)`` pairs or of
    :class:`gpxpy.geo.Location` (e.g. :class:`gpxpy.gpx.GPXTrackPoint`)
    objects.  Rings may be open or closed (a last point equal to the first is
    accepted and ignored).

    Ring winding direction does not matter.  A ring needs at least three
    distinct vertices.  Topological validation -- self-intersections, holes
    outside the exterior ring, holes touching or overlapping each other -- is
    performed when the polygon is clipped, because it depends on the
    explicitly selected :class:`LocalProjection`.
    """

    def __init__(self, exterior: RingLike, holes: Optional[Sequence[RingLike]] = None) -> None:
        self.exterior = self._normalize_ring(exterior, 'exterior')
        self.holes: List[List[LatLon]] = []
        if holes:
            for hole_no, hole in enumerate(holes):
                self.holes.append(self._normalize_ring(hole, f'hole {hole_no}'))

    @staticmethod
    def _normalize_ring(ring: RingLike, name: str) -> List[LatLon]:
        if ring is None:
            raise ClipError(f'{name} ring is missing')
        points = [_as_latlon(point) for point in ring]
        result: List[LatLon] = []
        for point in points:
            if not result or not _latlon_equal(point, result[-1]):
                result.append(point)
        if len(result) >= 2 and _latlon_equal(result[0], result[-1]):
            result.pop()
        if len(result) < 3:
            raise ClipError(f'{name} ring must contain at least 3 distinct points')
        return result


class _Contact(NamedTuple):
    point: XY
    t: float
    other_point: Optional[XY]


class _PreparedPolygon:
    """Polygon projected to meters and validated for one clipping call."""

    def __init__(self, polygon: Polygon, projection: LocalProjection) -> None:
        self.projection = projection
        self.exterior = [projection.project(*point) for point in polygon.exterior]
        self.holes = [[projection.project(*point) for point in hole]
                      for hole in polygon.holes]
        for ring in [self.exterior] + self.holes:
            for x, y in ring:
                projection.check_extent(x, y)
        _validate_simple_ring(self.exterior, 'exterior')
        for hole_no, hole in enumerate(self.holes):
            _validate_simple_ring(hole, f'hole {hole_no}')
            _validate_hole_in_exterior(hole, self.exterior, hole_no)
        for hole_no, hole in enumerate(self.holes):
            for other_no, other in enumerate(self.holes):
                if other_no <= hole_no:
                    continue
                if _rings_contact(hole, other):
                    raise ClipError(
                        f'holes {hole_no} and {other_no} touch or intersect')
                if _point_in_ring(hole[0], other) or _point_in_ring(other[0], hole):
                    raise ClipError(
                        f'hole {hole_no} is nested inside hole {other_no}; '
                        'nested holes are not supported')

    @property
    def rings(self) -> List[List[XY]]:
        return [self.exterior] + self.holes


def clip_segment(segment: "mod_gpx.GPXTrackSegment", polygon: Polygon,
                 projection: LocalProjection) -> List["mod_gpx.GPXTrackSegment"]:
    """Clip one track segment against a polygon.

    Returns a list of new :class:`gpxpy.gpx.GPXTrackSegment` objects holding
    the portions of ``segment`` that lie inside the region.  The list may be
    empty or contain several segments.  The input segment is never modified.
    """
    prepared = _PreparedPolygon(polygon, projection)
    points = segment.points
    for point in points:
        prepared.projection.check_extent(*projection.project(point.latitude,
                                                              point.longitude))
    return _build_segments(points, prepared)


def clip_track(track: "mod_gpx.GPXTrack", polygon: Polygon,
               projection: LocalProjection) -> "mod_gpx.GPXTrack":
    """Clip all segments of a track against a polygon.

    Returns a *new* :class:`gpxpy.gpx.GPXTrack` with the same metadata as
    ``track`` and one segment per inside portion (a single input segment can
    yield several output segments).  Empty portions are omitted.  The input
    track is never modified.
    """
    from .gpx import GPXTrack
    result = GPXTrack(name=track.name, description=track.description,
                      number=track.number)
    result.comment = track.comment
    result.source = track.source
    result.link = track.link
    result.link_text = track.link_text
    result.link_type = track.link_type
    result.type = track.type
    result.extensions = mod_copy.deepcopy(track.extensions)
    prepared = _PreparedPolygon(polygon, projection)
    for segment in track.segments:
        for point in segment.points:
            prepared.projection.check_extent(
                *projection.project(point.latitude, point.longitude))
        for new_segment in _build_segments(segment.points, prepared):
            new_segment.extensions = mod_copy.deepcopy(segment.extensions)
            result.segments.append(new_segment)
    return result


def _build_segments(points: Sequence["mod_gpx.GPXTrackPoint"],
                    prepared: _PreparedPolygon) -> List["mod_gpx.GPXTrackSegment"]:
    from .gpx import GPXTrackSegment

    rings = prepared.rings

    # Collapse consecutive duplicate points; zero length edges carry no
    # geometry and would only produce meaningless intersection parameters.
    clean: List["mod_gpx.GPXTrackPoint"] = []
    for point in points:
        if not clean or (point.latitude != clean[-1].latitude
                         or point.longitude != clean[-1].longitude):
            clean.append(point)

    def xy_of(point: "mod_gpx.GPXTrackPoint") -> XY:
        return prepared.projection.project(point.latitude, point.longitude)

    def point_at(previous: "mod_gpx.GPXTrackPoint",
                 current: "mod_gpx.GPXTrackPoint", t: float
                 ) -> "mod_gpx.GPXTrackPoint":
        if t <= _EPS_T:
            return previous
        if t >= 1.0 - _EPS_T:
            return current
        return _interpolated_point(previous, current, t, prepared.projection)

    # Cut the polyline at every boundary crossing.  The resulting tessellation
    # pieces each connect two vertices and lie entirely inside (or on) the
    # region or entirely outside.  A piece whose midpoint sits on a ring counts
    # as inside, so an edge that runs along the exterior or a hole boundary is
    # retained.
    pieces: List[Tuple["mod_gpx.GPXTrackPoint", "mod_gpx.GPXTrackPoint", bool]] = []
    for index, current in enumerate(clean[1:], 1):
        previous = clean[index - 1]
        a = xy_of(previous)
        b = xy_of(current)
        for t_start, t_end in _edge_intervals(a, b, rings):
            midpoint = _edge_point(a, b, (t_start + t_end) / 2.0)
            inside = _classify(midpoint, rings) or _point_on_any_ring(midpoint, rings)
            pieces.append((point_at(previous, current, t_start),
                           point_at(previous, current, t_end), inside))

    # Build an ordered vertex walk.  Each vertex occurrence is "kept" when it
    # belongs to an inside piece or lies on a boundary (the latter covers
    # standalone touches between two outside pieces).  Each link between two
    # consecutive walk vertices records whether its spanning piece is inside;
    # two kept vertices connected by an outside piece still separate segments.
    class _Occurrence:
        __slots__ = ('point', 'kept')

        point: "mod_gpx.GPXTrackPoint"
        kept: bool

        def __init__(self, point: "mod_gpx.GPXTrackPoint") -> None:
            self.point = point
            self.kept = False

    occurrences: List[_Occurrence] = []

    def occurrence_for(point: "mod_gpx.GPXTrackPoint") -> _Occurrence:
        for occurrence in occurrences:
            if _xy_equal(xy_of(occurrence.point), xy_of(point)):
                return occurrence
        occurrence = _Occurrence(point)
        occurrences.append(occurrence)
        return occurrence

    prepared_pieces: List[Tuple[_Occurrence, _Occurrence, bool]] = []
    for start_point, end_point, inside in pieces:
        prepared_pieces.append(
            (occurrence_for(start_point), occurrence_for(end_point), inside))

    for start_occ, end_occ, inside in prepared_pieces:
        if inside:
            start_occ.kept = True
            end_occ.kept = True
        else:
            if _point_on_any_ring(xy_of(start_occ.point), rings):
                start_occ.kept = True
            if _point_on_any_ring(xy_of(end_occ.point), rings):
                end_occ.kept = True

    walk: List[_Occurrence] = []
    linked: List[bool] = []

    def append_vertex(occurrence: _Occurrence, link_to_previous: bool) -> None:
        if walk and _xy_equal(xy_of(walk[-1].point), xy_of(occurrence.point)):
            # Same geometric vertex reached via tessellation; a vertex is
            # linked if either spanning piece is inside.
            linked[-1] = linked[-1] or link_to_previous
            return
        walk.append(occurrence)
        linked.append(link_to_previous)

    if prepared_pieces:
        append_vertex(prepared_pieces[0][0], False)
    previous_inside = False
    for start_occ, end_occ, inside in prepared_pieces:
        append_vertex(start_occ, previous_inside)
        append_vertex(end_occ, inside)
        previous_inside = inside

    segments: List[GPXTrackSegment] = []
    run: List["mod_gpx.GPXTrackPoint"] = []

    def flush() -> None:
        nonlocal run
        if run:
            segments.append(GPXTrackSegment(run))
        run = []

    for index, occurrence in enumerate(walk):
        if not occurrence.kept:
            flush()
            continue
        if run and not linked[index]:
            flush()
        if not run or not _xy_equal(xy_of(run[-1]), xy_of(occurrence.point)):
            run.append(occurrence.point)
    flush()
    return segments


def _edge_intervals(a: XY, b: XY,
                    rings: Sequence[Sequence[XY]]) -> List[Tuple[float, float]]:
    """Non-degenerate [t_start, t_end] intervals covering edge a->b."""
    boundaries = [0.0] + _edge_events(a, b, rings) + [1.0]
    intervals: List[Tuple[float, float]] = []
    for index, t_start in enumerate(boundaries[:-1]):
        t_end = boundaries[index + 1]
        if t_end - t_start >= _EPS_T:
            intervals.append((t_start, t_end))
    return intervals


def _point_on_any_ring(point: XY, rings: Sequence[Sequence[XY]]) -> bool:
    return any(_point_on_ring(point, ring) for ring in rings)


def _interpolated_point(previous: "mod_gpx.GPXTrackPoint",
                        current: "mod_gpx.GPXTrackPoint", t: float,
                        projection: LocalProjection) -> "mod_gpx.GPXTrackPoint":
    """Create a new track point on the edge previous->current at parameter t.

    Only time and elevation are interpolated.  Everything else (extensions,
    recorded speed/course, name, accuracy fields, ...) cannot be meaningfully
    derived and is therefore left empty rather than fabricated.
    """
    from .gpx import GPXTrackPoint

    x1, y1 = projection.project(previous.latitude, previous.longitude)
    x2, y2 = projection.project(current.latitude, current.longitude)
    latitude, longitude = projection.unproject(x1 + (x2 - x1) * t,
                                               y1 + (y2 - y1) * t)

    elevation: Optional[float] = None
    if previous.elevation is not None and current.elevation is not None:
        elevation = previous.elevation + (current.elevation - previous.elevation) * t

    time: Optional[mod_datetime.datetime] = None
    if previous.time is not None and current.time is not None:
        time = previous.time + (current.time - previous.time) * t

    return GPXTrackPoint(latitude=latitude, longitude=longitude,
                         elevation=elevation, time=time)


def _edge_events(a: XY, b: XY, rings: Sequence[Sequence[XY]]) -> List[float]:
    """Sorted edge parameters t in (0, 1) where edge a->b meets any ring.

    For a collinear overlap both overlap ends are reported, so the overlapping
    stretch becomes its own piece.  Duplicate parameters are merged.
    """
    events = set()
    for ring in rings:
        for c, d in _ring_edges(ring):
            contact = _segment_contact(a, b, c, d)
            if contact is None:
                continue
            if contact.other_point is None:
                t = contact.t
                if _EPS_T < t < 1.0 - _EPS_T:
                    events.add(_clamp01(t))
            else:
                t1 = contact.t
                t2 = _parameter(contact.other_point, a, b)
                for t in (t1, t2):
                    if _EPS_T < t < 1.0 - _EPS_T:
                        events.add(_clamp01(t))
    # An edge grazing exactly through a polygon vertex is reported once per
    # adjacent polygon edge; merge events that coincide geometrically so they
    # do not form an ultra-thin piece whose midpoint is numerically unstable.
    merged: List[float] = []
    for t in sorted(events):
        if merged and _xy_equal(_edge_point(a, b, t), _edge_point(a, b, merged[-1])):
            continue
        merged.append(t)
    return merged


def _edge_point(a: XY, b: XY, t: float) -> XY:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _classify(point: XY, rings: Sequence[Sequence[XY]]) -> bool:
    """True when the point belongs to the clipped region.

    Points on the exterior ring or on a hole boundary are considered inside:
    a track that runs along a boundary, including around a hole, is kept.
    """
    exterior = rings[0]
    if _point_on_ring(point, exterior):
        return True
    if not _point_in_ring(point, exterior):
        return False
    for hole in rings[1:]:
        if _point_on_ring(point, hole):
            return True
        if _point_in_ring(point, hole):
            return False
    return True


def _point_in_ring(point: XY, ring: Sequence[XY]) -> bool:
    """Standard ray casting; points exactly on the ring return False here.

    Uses a half-open rule at vertices and a small tolerance so that a point
    numerically indistinguishable from the boundary is not flipped to the
    wrong side by floating-point round-off.
    """
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in _ring_edges(ring):
        if y1 == y2:
            continue
        lo_y, hi_y = (y1, y2) if y1 < y2 else (y2, y1)
        if not (lo_y <= y < hi_y):
            continue
        cross_x = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
        if cross_x > x + _EPS:
            inside = not inside
        elif abs(cross_x - x) <= _EPS:
            return False
    return inside


def _point_on_ring(point: XY, ring: Sequence[XY]) -> bool:
    for c, d in _ring_edges(ring):
        if _point_on_segment(point, c, d):
            return True
    return False


def _point_on_segment(point: XY, a: XY, b: XY) -> bool:
    x, y = point
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    tolerance = _EPS + _EPS_SCALE * max(1.0, abs(dx), abs(dy))
    if length_sq <= tolerance * tolerance:
        return mod_math.hypot(x - ax, y - ay) <= tolerance
    t = ((x - ax) * dx + (y - ay) * dy) / length_sq
    if t < -_EPS_T or t > 1.0 + _EPS_T:
        return False
    projected_x = ax + t * dx
    projected_y = ay + t * dy
    return mod_math.hypot(x - projected_x, y - projected_y) <= tolerance


def _ring_edges(ring: Sequence[XY]) -> List[Tuple[XY, XY]]:
    return [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]


def _segment_contact(a: XY, b: XY, c: XY, d: XY) -> Optional[_Contact]:
    """Describe how segment a->b touches segment c->d.

    Returns ``None`` when they do not touch.  For a single contact point the
    result carries that point, t along a->b and ``other_point=None``.  For a
    collinear overlap it carries one overlap end as ``point``/``t`` and the
    other end as ``other_point``.
    """
    r = (b[0] - a[0], b[1] - a[1])
    s = (d[0] - c[0], d[1] - c[1])
    r_length_sq = r[0] * r[0] + r[1] * r[1]
    s_length_sq = s[0] * s[0] + s[1] * s[1]
    if r_length_sq <= _EPS * _EPS or s_length_sq <= _EPS * _EPS:
        return None

    scale = max(1.0, abs(r[0]), abs(r[1]), abs(s[0]), abs(s[1]))
    epsilon = _EPS_SCALE * scale * scale
    r_cross_s = r[0] * s[1] - r[1] * s[0]
    c_minus_a = (c[0] - a[0], c[1] - a[1])
    t_numerator = c_minus_a[0] * s[1] - c_minus_a[1] * s[0]
    u_numerator = c_minus_a[0] * r[1] - c_minus_a[1] * r[0]

    if abs(r_cross_s) > epsilon:
        t = t_numerator / r_cross_s
        u = u_numerator / r_cross_s
        if -_EPS_T <= t <= 1.0 + _EPS_T and -_EPS_T <= u <= 1.0 + _EPS_T:
            t = _clamp01(t)
            return _Contact(_edge_point(a, b, t), t, None)
        return None

    if abs(t_numerator) > epsilon or abs(u_numerator) > epsilon:
        return None

    # Collinear: find overlap interval along a->b.
    t_c = _parameter(c, a, b)
    t_d = _parameter(d, a, b)
    lo = max(0.0, min(t_c, t_d))
    hi = min(1.0, max(t_c, t_d))
    if hi + _EPS_T < lo:
        return None
    lo = _clamp01(lo)
    hi = _clamp01(hi)
    if hi - lo <= _EPS_T:
        return _Contact(_edge_point(a, b, lo), lo, None)
    return _Contact(_edge_point(a, b, lo), lo, _edge_point(a, b, hi))


def _parameter(point: XY, a: XY, b: XY) -> float:
    """Parameter t of point's projection on line a->b (segments are collinear)."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= _EPS * _EPS:
        return 0.0
    return _clamp01(((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length_sq)


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _validate_simple_ring(ring: Sequence[XY], name: str) -> None:
    """Reject zero length edges and self-touching/self-intersecting rings."""
    count = len(ring)
    for i in range(count):
        a, b = ring[i], ring[(i + 1) % count]
        if _xy_equal(a, b):
            raise ClipError(f'{name} ring contains a zero length edge')

    for i in range(count):
        a = ring[i]
        b = ring[(i + 1) % count]
        for j in range(i + 1, count):
            c = ring[j]
            d = ring[(j + 1) % count]
            adjacent = ((i + 1) % count == j) or (i == 0 and j == count - 1)
            contact = _segment_contact(a, b, c, d)
            if contact is None:
                continue
            if adjacent and contact.other_point is None:
                # The shared endpoint of two consecutive edges is allowed.
                shared = b if (i + 1) % count == j else a
                if _xy_equal(contact.point, shared):
                    continue
            raise ClipError(
                f'{name} ring is self-intersecting (edges {i} and {j} '
                'touch or cross)')


def _validate_hole_in_exterior(hole: Sequence[XY], exterior: Sequence[XY],
                               hole_no: int) -> None:
    for vertex in hole:
        if _point_on_ring(vertex, exterior) or not _point_in_ring(vertex, exterior):
            raise ClipError(
                f'hole {hole_no} is not strictly inside the exterior ring '
                '(a hole vertex is on or outside the exterior boundary)')


def _rings_contact(ring_1: Sequence[XY], ring_2: Sequence[XY]) -> bool:
    for a, b in _ring_edges(ring_1):
        for c, d in _ring_edges(ring_2):
            if _segment_contact(a, b, c, d) is not None:
                return True
    return False


def _as_latlon(point: PointLike) -> LatLon:
    if isinstance(point, (tuple, list)):
        latitude, longitude = point
    else:
        latitude = point.latitude
        longitude = point.longitude
    if latitude is None or longitude is None:
        raise ClipError('all polygon points must have latitude and longitude')
    if not _is_finite(latitude) or not _is_finite(longitude):
        raise ClipError('polygon coordinates must be finite numbers')
    return float(latitude), float(longitude)


def _latlon_equal(point_1: LatLon, point_2: LatLon) -> bool:
    return (abs(point_1[0] - point_2[0]) <= _LL_EPS
            and abs(point_1[1] - point_2[1]) <= _LL_EPS)


def _xy_equal(point_1: XY, point_2: XY) -> bool:
    return abs(point_1[0] - point_2[0]) <= _EPS and abs(point_1[1] - point_2[1]) <= _EPS


def _is_finite(value: float) -> bool:
    return isinstance(value, (int, float)) and mod_math.isfinite(float(value))
