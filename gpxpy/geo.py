# Copyright 2011 Tomo Krajina
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging as mod_logging
import math as mod_math

from typing import *

log = mod_logging.getLogger(__name__)

# Generic geo related function and class(es)

# latitude/longitude in GPX files is always in WGS84 datum
# WGS84 defined the Earth semi-major axis with 6378.137 km
EARTH_RADIUS = 6378.137 * 1000

# One degree in meters:
ONE_DEGREE = (2*mod_math.pi*EARTH_RADIUS) / 360  # ==> 111.319 km

HAVERSINE_THRESHOLD: float = .1
"""
Threshold for switching from equirectangular to haversine distance in ``distance()``
when ``haversine=False``. If the absolute difference in latitude or longitude between
two points exceeds this value (in degrees), the haversine formula is used; otherwise
the equirectangular approximation is used.

This is a public, user-tunable knob. Assigning a new value to it changes the behavior
of :func:`distance` for the entire process.
"""

_SPHERICAL_EPSILON: float = 1e-14
_ANTIPODAL_EDGE_EPSILON: float = 1e-12
_EQUAL_DISTANCE_EPSILON: float = 1e-12

_SphericalVector = Tuple[float, float, float]

def haversine_distance(latitude_1: float, longitude_1: float, latitude_2: float, longitude_2: float) -> float:
    """
    Haversine distance between two points, expressed in meters.

    Implemented from http://www.movable-type.co.uk/scripts/latlong.html
    """
    d_lon = mod_math.radians(longitude_1 - longitude_2)
    lat1 = mod_math.radians(latitude_1)
    lat2 = mod_math.radians(latitude_2)
    d_lat = lat1 - lat2

    a = mod_math.pow(mod_math.sin(d_lat/2),2) + \
        mod_math.pow(mod_math.sin(d_lon/2),2) * mod_math.cos(lat1) * mod_math.cos(lat2)
    c = 2 * mod_math.asin(mod_math.sqrt(a))
    d = EARTH_RADIUS * c

    return d


def get_course(latitude_1: float, longitude_1: float, latitude_2: float, longitude_2: float,
               loxodromic: bool=True) -> float:
    """
    The initial course from one point to another,
    expressed in decimal degrees clockwise from true North
    (not magnetic)
    (0.0 <= value < 360.0)

    Use the default loxodromic model in most cases
    (except when visualizing the long routes of maritime transport and aeroplanes)

    Implemented from http://www.movable-type.co.uk/scripts/latlong.html
    (sections 'Bearing' and 'Rhumb lines')
    """

    d_lon = mod_math.radians(longitude_2 - longitude_1)
    lat1 = mod_math.radians(latitude_1)
    lat2 = mod_math.radians(latitude_2)

    if not loxodromic:
        y = mod_math.sin(d_lon) * mod_math.cos(lat2)
        x = mod_math.cos(lat1) * mod_math.sin(lat2) - \
            mod_math.sin(lat1) * mod_math.cos(lat2) * mod_math.cos(d_lon)
    else:
        radian_circle = 2*mod_math.pi

        if abs(d_lon) > mod_math.pi:
            if d_lon > 0:
                d_lon = - (radian_circle - d_lon)
            else:
                d_lon = radian_circle + d_lon

        y = d_lon

        delta = mod_math.pi/4
        x = mod_math.log(mod_math.tan(delta + 0.5*lat2)
                         / mod_math.tan(delta + 0.5*lat1))

    course = mod_math.degrees(mod_math.atan2(y, x))
    return course % 360


def length(locations: List["Location"]=[], _3d: bool=False) -> float:
    """
    Compute the total length between locations.

    Parameters
    ----------
    locations: list
        List with Location objects to calculate the length between.
    _3d: bool
        True: use the 3d distance: with taking elevation in account.
        False: use the 2d distance: without taking elevation in account.

    Returns
    ----------
    length: float
        Sum of the length between consecutive locations in meters.
    """
    if not locations:
        return 0
    length: float = 0
    for previous_location, location in zip(locations, locations[1:]):
        if _3d:
            d = location.distance_3d(previous_location)
        else:
            d = location.distance_2d(previous_location)
        if d:
            length += d
    return length


def length_2d(locations: List["Location"]=[]) -> float:
    """ 2-dimensional length (meters) of locations (only latitude and longitude, no elevation). """
    return length(locations, False)


def length_3d(locations: List["Location"]=[]) -> float:
    """ 3-dimensional length (meters) of locations (it uses latitude, longitude, and elevation). """
    return length(locations, True)


def calculate_max_speed(speeds_and_distances: List[Tuple[float, float]], extremes_percentile: float, ignore_nonstandard_distances: bool) -> Optional[float]:
    """
    Compute average distance and standard deviation for distance. Extremes
    in distances are usually extremes in speeds, so we will ignore them,
    here.

    speeds_and_distances must be a list containing pairs of (speed, distance)
    for every point in a track segment.

    In many cases the top speeds are measurement errors. For that reason extreme speeds can be removed
    with the extremes_percentile (for example, a value of 0.05 will remove top 5%).
    """
    assert speeds_and_distances
    if len(speeds_and_distances) > 0:
        assert len(speeds_and_distances[0]) == 2
        # ...
        assert len(speeds_and_distances[-1]) == 2

    if not ignore_nonstandard_distances:
        return max(x[0] or 0 for x in speeds_and_distances)

    size = len(speeds_and_distances)

    if size < 2:
        # log.debug('Segment too small to compute speed, size=%s', size)
        return None

    distances = [x[1] for x in speeds_and_distances]
    average_distance = sum(distances) / size
    standard_distance_deviation = mod_math.sqrt(sum((distance - average_distance) ** 2 for distance in distances) / size)

    # Ignore items where the distance is too big:
    filtered_speeds_and_distances = [x for x in speeds_and_distances if abs(x[1] - average_distance) <= standard_distance_deviation * 1.5]

    # sort by speed:
    speeds = [x[0] for x in filtered_speeds_and_distances]
    if not speeds:
        return None
    speeds.sort()

    # Even here there may be some extremes => ignore the last 5%:
    index = int(len(speeds) * (1-extremes_percentile))
    if index >= len(speeds):
        index = -1

    return speeds[index]


def calculate_uphill_downhill(elevations: List[Optional[float]]) -> Tuple[float, float]:
    """
    Compute the total uphill and downhill elevation for a list of elevations.
    Note that the result is smoothened.

    Parameters
    ----------
    elevations: list
        List of elevations to calculate the total up-/downhill elevation between.
        If elevation is missing for a point, None can be used.

    Returns
    ----------
    uphill_downhill: tuple
        Tuple of total (uphill, downhill) elevation. (smoothened)
    """
    if not elevations:
        return 0, 0

    elevations = list(filter(lambda e: e is not None, elevations))
    size = len(elevations)

    def __filter(n: int) -> float:
        current_ele = elevations[n]
        if current_ele is None:
            return False
        if 0 < n < size - 1:
            previous_ele = elevations[n-1]
            next_ele = elevations[n+1]
            if previous_ele is not None and current_ele is not None and next_ele is not None:
                return previous_ele*.3 + current_ele*.4 + next_ele*.3
        return current_ele

    smoothed_elevations = list(map(__filter, range(size)))

    uphill, downhill = 0., 0.
    for prev, cur in zip(smoothed_elevations, smoothed_elevations[1:]):
        if prev is not None and cur is not None:
            d = cur - prev
            if d > 0:
                uphill += d
            else:
                downhill -= d
    return uphill, downhill


def distance(latitude_1: float, longitude_1: float, elevation_1: Optional[float],
            latitude_2: float, longitude_2: float, elevation_2: Optional[float],
            haversine: bool=False) -> float:
    """
    Distance between two points. If elevation is None compute a 2d distance

    if haversine==True -- haversine will be used for every computations,
    otherwise...

    Haversine distance will be used for distant points where elevation makes a
    small difference, so it is ignored. That's because haversine is 5-6 times
    slower than the dummy distance algorithm (which is OK for most GPS tracks).
    """

    # If points too distant -- compute haversine distance:
    if haversine or (abs(latitude_1 - latitude_2) > HAVERSINE_THRESHOLD or abs(longitude_1 - longitude_2) > HAVERSINE_THRESHOLD):
        return haversine_distance(latitude_1, longitude_1, latitude_2, longitude_2)

    coef = mod_math.cos(mod_math.radians(latitude_1))
    x = latitude_1 - latitude_2
    y = (longitude_1 - longitude_2) * coef

    distance_2d = mod_math.sqrt(x * x + y * y) * ONE_DEGREE

    if elevation_1 is None or elevation_2 is None or elevation_1 == elevation_2:
        return distance_2d

    return mod_math.sqrt(distance_2d ** 2 + (elevation_1 - elevation_2) ** 2)


def elevation_angle(location1: "Location", location2: "Location", radians: bool=False) -> Optional[float]:
    """ Uphill/downhill angle between two locations. """
    if location1.elevation is None or location2.elevation is None:
        return None

    b = location2.elevation - location1.elevation
    a = location2.distance_2d(location1)

    if not a:
        return 0

    angle = mod_math.atan(b / a)
    
    return angle if radians else mod_math.degrees(angle)


def distance_from_line(point: "Location", line_point_1: "Location", line_point_2: "Location") -> Optional[float]:
    """ Distance of point from a line given with two points. """
    assert point, point
    assert line_point_1, line_point_1
    assert line_point_2, line_point_2

    a = line_point_1.distance_2d(line_point_2)

    if not a:
        return line_point_1.distance_2d(point)

    b = line_point_1.distance_2d(point)
    c = line_point_2.distance_2d(point)

    if a is not None and b is not None and c is not None:
        s = (a + b + c) / 2
        return 2 * mod_math.sqrt(abs(s * (s - a) * (s - b) * (s - c))) / a
    return None


def get_line_equation_coefficients(location1: "Location", location2: "Location") -> Iterable[float]:
    """
    Get line equation coefficients for:
        latitude * a + longitude * b + c = 0

    This is a normal cartesian line (not spherical!)
    """
    if location1.longitude == location2.longitude:
        # Vertical line:
        return 0, 1, -location1.longitude
    else:
        a = (location1.latitude - location2.latitude) / (location1.longitude - location2.longitude)
        b = location1.latitude - location1.longitude * a
        return 1, -a, -b


def simplify_polyline(points: List["Location"], max_distance: Optional[float]) -> List["Location"]:
    """Does Ramer-Douglas-Peucker algorithm for simplification of polyline """

    _max_distance = max_distance if max_distance is not None else 10

    if len(points) < 3:
        return points

    begin, end = points[0], points[-1]

    # Use a "normal" line just to detect the most distant point (not its real distance)
    # this is because this is faster to compute than calling distance_from_line() for
    # every point.
    #
    # This is an approximation and may have some errors near the poles and if
    # the points are too distant, but it should be good enough for most use
    # cases...
    a, b, c = get_line_equation_coefficients(begin, end)

    # Initialize to safe values
    tmp_max_distance: float = 0
    tmp_max_distance_position = 1
    
    # Check distance of all points between begin and end, exclusive
    for point_no, point in enumerate(points[1:-1], 1):
        d = abs(a * point.latitude + b * point.longitude + c)
        if d > tmp_max_distance:
            tmp_max_distance = d
            tmp_max_distance_position = point_no

    # Now that we have the most distance point, compute its real distance:
    real_max_distance = distance_from_line(points[tmp_max_distance_position], begin, end)

    # If furthest point is less than max_distance, remove all points between begin and end
    if real_max_distance is not None and real_max_distance < _max_distance:
        return [begin, end]
    
    # If furthest point is more than max_distance, use it as anchor and run
    # function again using (begin to anchor) and (anchor to end), remove extra anchor
    return (simplify_polyline(points[:tmp_max_distance_position + 1], _max_distance) +
            simplify_polyline(points[tmp_max_distance_position:], _max_distance)[1:])


class PolylineNearestLocation(NamedTuple):
    location: "Location"
    polyline_index: int
    edge_index: int
    edge_ratio: float
    distance_from_polyline_start: float
    distance: float

    @property
    def ratio(self) -> float:
        return self.edge_ratio

    @property
    def distance_from_start(self) -> float:
        return self.distance_from_polyline_start


class _PolylineEdge(NamedTuple):
    start: _SphericalVector
    end: _SphericalVector
    angular_length: float
    length: float
    distance_from_start: float
    polyline_index: int
    edge_index: int


class _BoundingCap(NamedTuple):
    center: _SphericalVector
    radius: float


class _BVHNode(NamedTuple):
    cap: _BoundingCap
    left: Optional["_BVHNode"]
    right: Optional["_BVHNode"]
    edge: Optional[_PolylineEdge]


def _unit_vector(location: "Location") -> _SphericalVector:
    latitude = mod_math.radians(location.latitude)
    longitude = mod_math.radians(location.longitude)
    cos_latitude = mod_math.cos(latitude)
    return (
        cos_latitude * mod_math.cos(longitude),
        cos_latitude * mod_math.sin(longitude),
        mod_math.sin(latitude),
    )


def _vector_dot(vector_1: _SphericalVector, vector_2: _SphericalVector) -> float:
    return sum(value_1 * value_2 for value_1, value_2 in zip(vector_1, vector_2))


def _vector_cross(vector_1: _SphericalVector, vector_2: _SphericalVector) -> _SphericalVector:
    return (
        vector_1[1] * vector_2[2] - vector_1[2] * vector_2[1],
        vector_1[2] * vector_2[0] - vector_1[0] * vector_2[2],
        vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0],
    )


def _vector_norm(vector: Tuple[float, ...]) -> float:
    return mod_math.sqrt(sum(value * value for value in vector))


def _normalize_vector(vector: Tuple[float, ...]) -> _SphericalVector:
    norm = _vector_norm(vector)
    if norm <= _SPHERICAL_EPSILON:
        raise ValueError('Cannot normalize a zero-length spherical vector')
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _angular_distance(vector_1: _SphericalVector, vector_2: _SphericalVector) -> float:
    cosine = min(1.0, max(-1.0, _vector_dot(vector_1, vector_2)))
    sine_half = mod_math.sqrt(max(0.0, (1.0 - cosine) / 2.0))
    cosine_half = mod_math.sqrt(max(0.0, (1.0 + cosine) / 2.0))
    return 2.0 * mod_math.atan2(sine_half, cosine_half)


def _vector_to_location(vector: _SphericalVector) -> "Location":
    x, y, z = vector
    return Location(mod_math.degrees(mod_math.atan2(z, mod_math.hypot(x, y))),
                    mod_math.degrees(mod_math.atan2(y, x)))


def _edge_cap(edge: _PolylineEdge) -> _BoundingCap:
    if edge.angular_length <= _SPHERICAL_EPSILON:
        return _BoundingCap(edge.start, 0.0)
    center = _normalize_vector((
        edge.start[0] + edge.end[0],
        edge.start[1] + edge.end[1],
        edge.start[2] + edge.end[2],
    ))
    return _BoundingCap(center, edge.angular_length / 2.0)


def _combine_caps(cap_1: _BoundingCap, cap_2: _BoundingCap) -> _BoundingCap:
    distance = _angular_distance(cap_1.center, cap_2.center)
    if distance + cap_1.radius <= cap_2.radius + _SPHERICAL_EPSILON:
        return cap_2
    if distance + cap_2.radius <= cap_1.radius + _SPHERICAL_EPSILON:
        return cap_1
    return _BoundingCap(cap_1.center,
                        min(mod_math.pi, max(cap_1.radius, distance + cap_2.radius)))


def _build_edges(polylines: Sequence[Sequence["Location"]]) -> List[_PolylineEdge]:
    edges: List[_PolylineEdge] = []
    for polyline_index, points in enumerate(polylines):
        distance_from_start = 0.0
        previous_end: Optional[_SphericalVector] = None
        edge_range = range(1) if len(points) == 1 else range(max(0, len(points) - 1))
        for edge_index in edge_range:
            point = points[edge_index]
            start = _unit_vector(point)
            end = _unit_vector(points[edge_index + 1]) if len(points) > 1 else start
            angular_length = _angular_distance(start, end)
            if mod_math.pi - angular_length < _ANTIPODAL_EDGE_EPSILON:
                raise ValueError(f'Antipodal polyline edge {edge_index} has no unique shortest arc')
            length = angular_length * EARTH_RADIUS
            if previous_end is not None:
                distance_from_start += _angular_distance(previous_end, start) * EARTH_RADIUS
            edges.append(_PolylineEdge(start, end, angular_length, length,
                                       distance_from_start, polyline_index, edge_index))
            previous_end = end
    return edges


def _project_to_edge(location_vector: _SphericalVector, edge: _PolylineEdge) -> Tuple[_SphericalVector, float, float, float]:
    if edge.angular_length <= _SPHERICAL_EPSILON:
        distance = _angular_distance(location_vector, edge.start)
        return edge.start, 0.0, edge.distance_from_start, distance * EARTH_RADIUS

    sine_length = mod_math.sin(edge.angular_length)
    cross = _vector_cross(edge.start, edge.end)
    normal = _normalize_vector((
        cross[0] / sine_length,
        cross[1] / sine_length,
        cross[2] / sine_length,
    ))
    tangent = _normalize_vector(tuple(
        (edge.end[index] - edge.start[index] * mod_math.cos(edge.angular_length)) / sine_length
        for index in range(3)
    ))

    along_axis = _vector_dot(location_vector, tangent)
    across_axis = _vector_dot(location_vector, normal)
    start_axis = _vector_dot(location_vector, edge.start)
    along_angle = mod_math.atan2(along_axis, start_axis)
    along_angle = min(max(along_angle, 0.0), edge.angular_length)

    projected = _normalize_vector((
        edge.start[0] * mod_math.cos(along_angle) + tangent[0] * mod_math.sin(along_angle),
        edge.start[1] * mod_math.cos(along_angle) + tangent[1] * mod_math.sin(along_angle),
        edge.start[2] * mod_math.cos(along_angle) + tangent[2] * mod_math.sin(along_angle),
    ))
    angular_distance = _angular_distance(location_vector, projected)
    ratio = along_angle / edge.angular_length
    distance_from_start = edge.distance_from_start + edge.length * ratio
    return projected, ratio, distance_from_start, angular_distance * EARTH_RADIUS


def _result_from_edge(location_vector: _SphericalVector, edge: _PolylineEdge) -> Tuple[PolylineNearestLocation, float]:
    projected, ratio, distance_from_start, distance = _project_to_edge(location_vector, edge)
    result = PolylineNearestLocation(
        _vector_to_location(projected),
        edge.polyline_index,
        edge.edge_index,
        ratio,
        distance_from_start,
        distance,
    )
    return result, distance / EARTH_RADIUS


def _nearest_edge_scan(edges: List[_PolylineEdge], location_vector: _SphericalVector,
                       max_distance: Optional[float]) -> Optional[PolylineNearestLocation]:
    best_result: Optional[PolylineNearestLocation] = None
    best_angular = mod_math.inf
    for edge in edges:
        result, angular_distance = _result_from_edge(location_vector, edge)
        within_max_distance = max_distance is None or angular_distance * EARTH_RADIUS <= max_distance
        if within_max_distance and angular_distance < best_angular - _EQUAL_DISTANCE_EPSILON:
            best_result = result
            best_angular = angular_distance
    return best_result


def nearest_location_on_polyline(points: Sequence["Location"], location: "Location",
                                 max_distance: Optional[float] = None) -> Optional[PolylineNearestLocation]:
    """
    Find the nearest point on spherical great-circle segments of one polyline.
    This is the reference full scan and builds no persistent index.
    """
    return nearest_location_on_polylines([points], location, max_distance)


def nearest_location_on_polylines(polylines: Sequence[Sequence["Location"]], location: "Location",
                                  max_distance: Optional[float] = None) -> Optional[PolylineNearestLocation]:
    """ Reference full scan over spherical great-circle polyline edges. """
    if max_distance is not None and max_distance < 0:
        raise ValueError('max_distance must not be negative')
    edges = _build_edges(polylines)
    return _nearest_edge_scan(edges, _unit_vector(location), max_distance)


class SphericalPolylineIndex:
    """ Reusable BVH index for one or more spherical polylines. """

    __slots__ = ('edges', 'root', 'polyline_count')

    def __init__(self, polylines: Sequence[Sequence["Location"]]) -> None:
        self.edges = _build_edges(polylines)
        self.polyline_count = len(polylines)
        self.root = self._build_node(self.edges) if self.edges else None

    def _build_node(self, edges: List[_PolylineEdge]) -> _BVHNode:
        if len(edges) == 1:
            edge = edges[0]
            return _BVHNode(_edge_cap(edge), None, None, edge)

        middle = len(edges) // 2
        left = self._build_node(edges[:middle])
        right = self._build_node(edges[middle:])
        return _BVHNode(_combine_caps(left.cap, right.cap), left, right, None)

    def find_nearest_location(self, location: "Location",
                              max_distance: Optional[float] = None) -> Optional[PolylineNearestLocation]:
        if self.root is None:
            return None
        if max_distance is not None and max_distance < 0:
            raise ValueError('max_distance must not be negative')

        location_vector = _unit_vector(location)
        best_angular = mod_math.inf
        best_edge: Optional[_PolylineEdge] = None
        best_projected: Optional[_SphericalVector] = None
        best_ratio = 0.0
        best_distance_from_start = 0.0

        def search(node: _BVHNode) -> None:
            nonlocal best_angular, best_edge, best_projected, best_ratio, best_distance_from_start
            lower_bound = max(0.0, _angular_distance(location_vector, node.cap.center) - node.cap.radius)
            if best_angular != mod_math.inf and lower_bound > best_angular + _EQUAL_DISTANCE_EPSILON:
                return
            if node.edge is not None:
                projected, ratio, distance_from_start, distance = _project_to_edge(location_vector, node.edge)
                angular_distance = distance / EARTH_RADIUS
                if max_distance is not None and distance > max_distance:
                    return
                if angular_distance < best_angular - _EQUAL_DISTANCE_EPSILON:
                    best_angular = angular_distance
                    best_edge = node.edge
                    best_projected = projected
                    best_ratio = ratio
                    best_distance_from_start = distance_from_start
                return
            if node.left is not None:
                search(node.left)
            if node.right is not None:
                search(node.right)

        search(self.root)
        if best_edge is None or best_projected is None:
            return None
        return PolylineNearestLocation(
            _vector_to_location(best_projected),
            best_edge.polyline_index,
            best_edge.edge_index,
            best_ratio,
            best_distance_from_start,
            best_angular * EARTH_RADIUS,
        )


class Location:
    """ Generic geographical location """

    def __init__(self, latitude: float, longitude: float, elevation: Optional[float]=None) -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation

    def has_elevation(self) -> bool:
        """Returns if this location contains elevation data."""
        return self.elevation is not None

    def remove_elevation(self) -> None:
        """ Remove the elevation data from this location."""
        self.elevation = None

    def distance_2d(self, location: "Location") -> Optional[float]:
        """
        Calculate the distance between self and location in meters.
        Does not take elevation in account.

        Parameters
        ----------
        location: Location
            Location to calculate the distance to.

        Returns
        ----------
        distance : float
            Distance returned in meters.
        """
        if not location:
            return None

        return distance(self.latitude, self.longitude, None, location.latitude, location.longitude, None)

    def distance_3d(self, location: "Location") -> Optional[float]:
        """
        Calculate the distance between self and location in meters.
        Takes elevation in account.

        Parameters
        ----------
        location: Location
            Location to calculate the distance to.

        Returns
        ----------
        distance : float
            Distance returned in meters.
        """
        if not location:
            return None

        return distance(self.latitude, self.longitude, self.elevation, location.latitude, location.longitude, location.elevation)

    def elevation_angle(self, location: "Location", radians: bool=False) -> Optional[float]:
        """
        Computes the uphill/downhill angle towards locations.

        Parameters
        ----------
        location: Location
            Location to calculate the angle to.
        radians: bool, default=False
            If True: return result in radians.
            If False: return result in degrees.

        Returns
        ----------
        angle : float
            Angle returned in degrees or radians.
        """
        return elevation_angle(self, location, radians)

    def move(self, location_delta: "LocationDelta") -> None:
        """
        Move this location with the given LocationDelta.

        Parameters
        ----------
        location_delta: LocationDelta
            Delta (distance/angle or lat/lon) offset to apply to this location.
        """
        self.latitude, self.longitude = location_delta.move(self)

    def __add__(self, location_delta: "LocationDelta") -> "Location":
        latitude, longitude = location_delta.move(self)
        return Location(latitude, longitude)

    def __str__(self) -> str:
        return f'[loc:{self.latitude},{self.longitude}@{self.elevation}]'

    def __repr__(self) -> str:
        if self.elevation is None:
            return f'Location({self.latitude}, {self.longitude})'
        else:
            return f'Location({self.latitude}, {self.longitude}, {self.elevation})'


class LocationDelta:
    """
    Intended to use similar to timestamp.timedelta, but for Locations.
    """

    NORTH = 0
    EAST = 90
    SOUTH = 180
    WEST = 270

    def __init__(self, distance: Optional[float]=None, angle: Optional[float]=None,
                 latitude_diff: Optional[float]=None,longitude_diff: Optional[float]=None) -> None:
        """
        Version 1:
            Distance (in meters).
            angle_from_north *clockwise*.
            ...must be given
        Version 2:
            latitude_diff and longitude_diff
            ...must be given
        """
        if (distance is not None) and (angle is not None):
            if (latitude_diff is not None) or (longitude_diff is not None):
                raise Exception('No lat/lon diff if using distance and angle!')
            self.distance = distance
            self.angle_from_north = angle
            self.move_function = self.move_by_angle_and_distance
        elif (latitude_diff is not None) and (longitude_diff is not None):
            if (distance is not None) or (angle is not None):
                raise Exception('No distance/angle if using lat/lon diff!')
            self.latitude_diff  = latitude_diff
            self.longitude_diff = longitude_diff
            self.move_function = self.move_by_lat_lon_diff

    def move(self, location: Location) -> Tuple[float, float]:
        """
        Move location by this LocationDelta.
        """
        return self.move_function(location)

    def move_by_angle_and_distance(self, location: Location) -> Tuple[float, float]:
        coef = mod_math.cos(mod_math.radians(location.latitude))
        vertical_distance_diff   = mod_math.sin(mod_math.radians(90 - self.angle_from_north)) / ONE_DEGREE
        horizontal_distance_diff = mod_math.cos(mod_math.radians(90 - self.angle_from_north)) / ONE_DEGREE
        lat_diff = self.distance * vertical_distance_diff
        lon_diff = self.distance * horizontal_distance_diff / coef
        return location.latitude + lat_diff, location.longitude + lon_diff

    def move_by_lat_lon_diff(self, location: "Location") -> Tuple[float, float]:
        return location.latitude + self.latitude_diff, location.longitude + self.longitude_diff

    def __repr__(self) -> str:
        if hasattr(self, 'distance'):
            return f'LocationDelta: {self.distance}, {self.angle_from_north}, {None}, {None}'
        return f'LocationDelta: {None}, {None}, {self.latitude_diff}, {self.longitude_diff}'

    def __str__(self) -> str:
        if hasattr(self, 'distance'):
            return f'LocationDelta(distance={self.distance}, ' \
                   f'angle={self.angle_from_north})'
        return f'LocationDelta(latitude_diff={self.latitude_diff}, ' \
               f'longitude_diff={self.longitude_diff})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if (hasattr(self, 'distance') and not hasattr(other, 'distance')) \
                or (not hasattr(self, "distance") and hasattr(other, "distance")):
            # self and other are both a different kind of LocationDelta (angle&distance vs. lat-&longitude_diff)
            return False
        if hasattr(self, 'distance'):
            return self.distance == other.distance and self.angle_from_north == other.angle_from_north
        return self.latitude_diff == other.latitude_diff and self.longitude_diff == other.longitude_diff
