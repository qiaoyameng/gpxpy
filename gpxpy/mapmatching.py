"""
Offline map matching of GPS observations to a caller-provided road network.

The road network is a small directed graph: every edge has a polyline
geometry and a direction (one-way streets are simply directed edges). No
online map service is used, the caller controls both the graph size and the
candidate limits.

For every observation a limited set of projection candidates (points on
nearby edges) is found. Then the best sequence of candidates is computed
with a Hidden-Markov-Model / Viterbi approach (similar to Newson & Krumm,
"Hidden Markov Map Matching Through Noise and Sparseness"):

* emission score: how far the observation is from the candidate,
* transition score: how well the legal road-network travel distance between
  two consecutive candidates (shortest directed path, candidates may lie in
  the middle of an edge) matches the straight-line distance between the two
  observations.

If no legal route connects two consecutive observations, the point is
reported as unmatched -- the matcher never fakes a straight-line jump and
pretends it was a road. Large time gaps between observations start a new
segment instead of guessing the route in between.

Example:

    import gpxpy.mapmatching as mod_mapmatching

    graph = mod_mapmatching.RoadGraph()
    graph.add_edge('main_st', 'n0', 'n1', [(45.0, 14.0), (45.0, 14.01)])
    graph.add_edge('side_st', 'n1', 'n2', [(45.0, 14.01), (45.001, 14.01)])

    matcher = mod_mapmatching.MapMatcher(graph, max_candidates=5)
    result = matcher.match(mod_mapmatching.observations_from_points(segment.points))
    for point in result.points:
        if point.matched:
            print(point.edge_id, point.latitude, point.longitude)
        else:
            print('unmatched:', point.reason)
"""

import datetime as mod_datetime
import heapq as mod_heapq
import logging as mod_logging
import math as mod_math

from enum import Enum
from typing import *

from . import geo as mod_geo

log = mod_logging.getLogger(__name__)


class Observation(NamedTuple):
    """A single GPS observation to be matched to the road network."""
    latitude: float
    longitude: float
    time: Optional[mod_datetime.datetime] = None


class Candidate(NamedTuple):
    """
    A projection of an observation onto a road edge.

    offset is the distance in meters along the edge polyline from the start
    of the edge (candidates may lie in the middle of an edge), distance is
    the distance in meters between the observation and the projection.
    """
    edge_id: str
    offset: float
    latitude: float
    longitude: float
    distance: float


class UnmatchedReason(Enum):
    """Why an observation could not be matched to the road network."""
    #: No edge within max_candidate_distance of the observation
    NO_CANDIDATE = 'no_candidate'
    #: Candidates exist, but no legal (directed) route connects them with
    #: the rest of the matched track
    NO_ROUTE = 'no_route'


class MatchedPoint(NamedTuple):
    """
    The matching result for a single observation.

    For matched points, latitude/longitude are the projected position on the
    edge, edge_offset the distance along the edge and distance the distance
    between the observation and the projection. connecting_edges lists the
    edge ids of the legal route from the previous matched point to this one.

    segment_break is True for the first point of a new segment (after an
    unmatched point or a large time gap).
    """
    index: int
    matched: bool
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    edge_id: Optional[str] = None
    edge_offset: Optional[float] = None
    distance: Optional[float] = None
    reason: Optional[UnmatchedReason] = None
    segment_break: bool = False
    connecting_edges: Tuple[str, ...] = ()


class RoadEdge:
    """A directed edge of the road network with a polyline geometry."""

    __slots__ = ('edge_id', 'from_node', 'to_node', 'polyline', 'length')

    def __init__(self, edge_id: str, from_node: str, to_node: str,
                 polyline: Sequence[Tuple[float, float]]) -> None:
        if len(polyline) < 2:
            raise ValueError('Edge polyline needs at least two points')
        self.edge_id = edge_id
        self.from_node = from_node
        self.to_node = to_node
        self.polyline: List[Tuple[float, float]] = list(polyline)
        self.length = 0.0
        for first, second in zip(self.polyline, self.polyline[1:]):
            self.length += mod_geo.haversine_distance(first[0], first[1], second[0], second[1])

    def __repr__(self) -> str:
        return 'RoadEdge({0!r}, {1!r} -> {2!r}, length={3:.1f}m)'.format(
            self.edge_id, self.from_node, self.to_node, self.length)


class RoadGraph:
    """
    A small directed road network provided by the caller.

    Nodes are plain string ids, edges are directed (use two edges for a
    two-way road) and carry a polyline of (latitude, longitude) points.
    """

    def __init__(self) -> None:
        self._edges: Dict[str, RoadEdge] = {}
        self._adjacency: Dict[str, List[Tuple[str, str, float]]] = {}
        self._shortest_path_cache: Dict[Tuple[str, str], Optional[Tuple[float, List[str]]]] = {}

    def add_edge(self, edge_id: str, from_node: str, to_node: str,
                 polyline: Sequence[Tuple[float, float]]) -> RoadEdge:
        """Add a directed edge. Returns the created RoadEdge."""
        if edge_id in self._edges:
            raise ValueError('Duplicate edge id: {0}'.format(edge_id))
        edge = RoadEdge(edge_id, from_node, to_node, polyline)
        self._edges[edge_id] = edge
        self._adjacency.setdefault(from_node, []).append((edge_id, to_node, edge.length))
        self._adjacency.setdefault(to_node, [])
        self._shortest_path_cache.clear()
        return edge

    @property
    def edges(self) -> Dict[str, RoadEdge]:
        return self._edges

    def edge(self, edge_id: str) -> RoadEdge:
        return self._edges[edge_id]

    def shortest_path(self, from_node: str, to_node: str) -> Optional[Tuple[float, List[str]]]:
        """
        Shortest directed path between two nodes (Dijkstra).

        Returns a tuple (distance in meters, list of edge ids) or None if
        to_node cannot be reached from from_node following edge directions.
        """
        key = (from_node, to_node)
        if key not in self._shortest_path_cache:
            self._shortest_path_cache[key] = self._compute_shortest_path(from_node, to_node)
        return self._shortest_path_cache[key]

    def _compute_shortest_path(self, from_node: str, to_node: str) -> Optional[Tuple[float, List[str]]]:
        if from_node == to_node:
            return 0.0, []
        distances: Dict[str, float] = {from_node: 0.0}
        previous: Dict[str, Tuple[str, str]] = {}
        heap: List[Tuple[float, str]] = [(0.0, from_node)]
        while heap:
            distance, node = mod_heapq.heappop(heap)
            if node == to_node:
                break
            if distance > distances.get(node, mod_math.inf):
                continue
            for edge_id, next_node, length in self._adjacency.get(node, []):
                new_distance = distance + length
                if new_distance < distances.get(next_node, mod_math.inf):
                    distances[next_node] = new_distance
                    previous[next_node] = (node, edge_id)
                    mod_heapq.heappush(heap, (new_distance, next_node))
        if to_node not in distances:
            return None
        edge_ids: List[str] = []
        node = to_node
        while node != from_node:
            previous_node, edge_id = previous[node]
            edge_ids.append(edge_id)
            node = previous_node
        edge_ids.reverse()
        return distances[to_node], edge_ids


class MatchResult:
    """The result of MapMatcher.match(), one MatchedPoint per observation."""

    def __init__(self, points: List[MatchedPoint]) -> None:
        self.points = points

    @property
    def matched_points(self) -> List[MatchedPoint]:
        return [point for point in self.points if point.matched]

    @property
    def unmatched_points(self) -> List[MatchedPoint]:
        return [point for point in self.points if not point.matched]

    @property
    def matched_edges(self) -> List[str]:
        """Edge ids used by the matched track, in order, without consecutive duplicates."""
        result: List[str] = []
        for point in self.points:
            if point.matched and point.edge_id is not None:
                if not result or result[-1] != point.edge_id:
                    result.append(point.edge_id)
        return result


class _ChainState:
    """Internal Viterbi chain state used by MapMatcher.match()."""

    def __init__(self) -> None:
        self.obs_indices: List[int] = []
        self.layers: List[List[Candidate]] = []
        self.scores: List[List[float]] = []
        self.parents: List[List[Optional[int]]] = []
        self.routes: List[List[Optional[List[str]]]] = []

    @property
    def active(self) -> bool:
        return bool(self.obs_indices)

    def start(self, index: int, candidates: List[Candidate], scores: List[float]) -> None:
        self.obs_indices = [index]
        self.layers = [candidates]
        self.scores = [scores]
        self.parents = [[None] * len(candidates)]
        self.routes = [[None] * len(candidates)]

    def add_layer(self, index: int, candidates: List[Candidate], scores: List[float],
                  parents: List[Optional[int]], routes: List[Optional[List[str]]]) -> None:
        self.obs_indices.append(index)
        self.layers.append(candidates)
        self.scores.append(scores)
        self.parents.append(parents)
        self.routes.append(routes)

    def clear(self) -> None:
        self.obs_indices = []
        self.layers = []
        self.scores = []
        self.parents = []
        self.routes = []


class MapMatcher:
    """
    Offline map matcher for a caller-provided RoadGraph.

    Parameters (all limits are controlled by the caller):

    * max_candidate_distance: observations further than this (in meters)
      from every edge are reported as unmatched (UnmatchedReason.NO_CANDIDATE),
    * max_candidates: keep at most this many best candidates per observation,
    * gps_sigma: expected GPS error in meters (emission score),
    * transition_beta: scale in meters for the difference between the
      road-network travel distance and the straight-line observation
      distance (transition score),
    * max_time_gap: if set (seconds), observations separated by a longer
      gap start a new segment instead of being connected by a guessed route.
    """

    def __init__(self, graph: RoadGraph, max_candidate_distance: float = 50.0,
                 max_candidates: int = 8, gps_sigma: float = 20.0,
                 transition_beta: float = 30.0,
                 max_time_gap: Optional[float] = None) -> None:
        if max_candidate_distance <= 0:
            raise ValueError('max_candidate_distance must be > 0')
        if max_candidates < 1:
            raise ValueError('max_candidates must be >= 1')
        if gps_sigma <= 0:
            raise ValueError('gps_sigma must be > 0')
        if transition_beta <= 0:
            raise ValueError('transition_beta must be > 0')
        self.graph = graph
        self.max_candidate_distance = max_candidate_distance
        self.max_candidates = max_candidates
        self.gps_sigma = gps_sigma
        self.transition_beta = transition_beta
        self.max_time_gap = max_time_gap
        self._route_cache: Dict[Tuple[Candidate, Candidate], Optional[Tuple[float, List[str]]]] = {}

    def get_candidates(self, latitude: float, longitude: float) -> List[Candidate]:
        """
        Projections of a position onto nearby edges, sorted by distance,
        limited by max_candidate_distance and max_candidates.
        """
        candidates: List[Candidate] = []
        for edge in self.graph.edges.values():
            candidate = self._project(latitude, longitude, edge)
            if candidate is not None and candidate.distance <= self.max_candidate_distance:
                candidates.append(candidate)
        candidates.sort(key=lambda candidate: candidate.distance)
        return candidates[:self.max_candidates]

    def match(self, observations: Sequence[Observation]) -> MatchResult:
        """
        Match a sequence of observations to the road network.

        Returns a MatchResult with one MatchedPoint per observation. Points
        that cannot be matched (no nearby candidate, or no legal route from
        the rest of the track) are reported with matched=False and a reason;
        the matcher never connects such points with a fake straight line.
        """
        points: List[Optional[MatchedPoint]] = [None] * len(observations)
        chain = _ChainState()
        previous_observation: Optional[Observation] = None

        for index, observation in enumerate(observations):
            candidates = self.get_candidates(observation.latitude, observation.longitude)

            if not candidates:
                self._close_chain(chain, points)
                points[index] = MatchedPoint(index=index, matched=False,
                                             reason=UnmatchedReason.NO_CANDIDATE)
                previous_observation = observation
                continue

            if chain.active and self.max_time_gap is not None \
                    and previous_observation is not None \
                    and previous_observation.time is not None \
                    and observation.time is not None:
                gap = (observation.time - previous_observation.time).total_seconds()
                if gap > self.max_time_gap:
                    # Too much time passed: the route in between is unknown,
                    # start a new segment instead of guessing it.
                    self._close_chain(chain, points)

            if not chain.active:
                chain.start(index, candidates,
                            [self._emission_log_score(candidate) for candidate in candidates])
            else:
                previous = observations[chain.obs_indices[-1]]
                linear_distance = mod_geo.haversine_distance(
                    previous.latitude, previous.longitude,
                    observation.latitude, observation.longitude)
                layer_candidates: List[Candidate] = []
                layer_scores: List[float] = []
                layer_parents: List[Optional[int]] = []
                layer_routes: List[Optional[List[str]]] = []
                for candidate in candidates:
                    best_score: Optional[float] = None
                    best_parent: Optional[int] = None
                    best_route: Optional[List[str]] = None
                    for parent_index, parent_candidate in enumerate(chain.layers[-1]):
                        route = self._route_distance(parent_candidate, candidate)
                        if route is None:
                            continue
                        route_distance, route_edges = route
                        score = chain.scores[-1][parent_index] \
                            + self._transition_log_score(route_distance, linear_distance) \
                            + self._emission_log_score(candidate)
                        if best_score is None or score > best_score:
                            best_score = score
                            best_parent = parent_index
                            best_route = route_edges
                    if best_score is not None:
                        layer_candidates.append(candidate)
                        layer_scores.append(best_score)
                        layer_parents.append(best_parent)
                        layer_routes.append(best_route)
                if not layer_candidates:
                    # No legal route from the current chain to any candidate:
                    # report the point as unmatched instead of faking a
                    # straight-line jump across the road network.
                    self._close_chain(chain, points)
                    points[index] = MatchedPoint(index=index, matched=False,
                                                 reason=UnmatchedReason.NO_ROUTE)
                    previous_observation = observation
                    continue
                chain.add_layer(index, layer_candidates, layer_scores, layer_parents, layer_routes)

            previous_observation = observation

        self._close_chain(chain, points)

        return MatchResult([
            point if point is not None else MatchedPoint(index=index, matched=False,
                                                         reason=UnmatchedReason.NO_CANDIDATE)
            for index, point in enumerate(points)
        ])

    def _close_chain(self, chain: _ChainState, points: List[Optional[MatchedPoint]]) -> None:
        if not chain.active:
            return
        best_index = 0
        best_score = chain.scores[-1][0]
        for candidate_index, score in enumerate(chain.scores[-1]):
            if score > best_score:
                best_score = score
                best_index = candidate_index
        for layer_position in range(len(chain.obs_indices) - 1, -1, -1):
            obs_index = chain.obs_indices[layer_position]
            candidate = chain.layers[layer_position][best_index]
            route_edges = chain.routes[layer_position][best_index]
            points[obs_index] = MatchedPoint(
                index=obs_index,
                matched=True,
                latitude=candidate.latitude,
                longitude=candidate.longitude,
                edge_id=candidate.edge_id,
                edge_offset=candidate.offset,
                distance=candidate.distance,
                segment_break=layer_position == 0 and obs_index > 0,
                connecting_edges=tuple(route_edges) if route_edges else (),
            )
            parent = chain.parents[layer_position][best_index]
            if parent is None:
                break
            best_index = parent
        chain.clear()

    def _project(self, latitude: float, longitude: float, edge: RoadEdge) -> Optional[Candidate]:
        polyline = edge.polyline
        if len(polyline) < 2:
            return None
        cos_latitude = mod_math.cos(mod_math.radians(latitude))
        if abs(cos_latitude) < 1e-12:
            cos_latitude = 1e-12

        def to_xy(point: Tuple[float, float]) -> Tuple[float, float]:
            # Local equirectangular projection (meters) around the observation
            return (mod_math.radians(point[1] - longitude) * cos_latitude * mod_geo.EARTH_RADIUS,
                    mod_math.radians(point[0] - latitude) * mod_geo.EARTH_RADIUS)

        best_distance = mod_math.inf
        best_offset = 0.0
        best_x = 0.0
        best_y = 0.0
        cumulative = 0.0
        previous_x, previous_y = to_xy(polyline[0])
        for point in polyline[1:]:
            next_x, next_y = to_xy(point)
            dx = next_x - previous_x
            dy = next_y - previous_y
            segment_length_squared = dx * dx + dy * dy
            segment_length = mod_math.sqrt(segment_length_squared)
            if segment_length_squared > 0:
                # The observation is at (0, 0) in this local frame
                t = -(previous_x * dx + previous_y * dy) / segment_length_squared
                t = max(0.0, min(1.0, t))
            else:
                t = 0.0
            projection_x = previous_x + t * dx
            projection_y = previous_y + t * dy
            distance = mod_math.hypot(projection_x, projection_y)
            if distance < best_distance:
                best_distance = distance
                best_offset = cumulative + t * segment_length
                best_x = projection_x
                best_y = projection_y
            cumulative += segment_length
            previous_x, previous_y = next_x, next_y

        projected_latitude = latitude + mod_math.degrees(best_y / mod_geo.EARTH_RADIUS)
        projected_longitude = longitude + mod_math.degrees(best_x / (mod_geo.EARTH_RADIUS * cos_latitude))
        return Candidate(edge_id=edge.edge_id, offset=best_offset,
                         latitude=projected_latitude, longitude=projected_longitude,
                         distance=best_distance)

    def _route_distance(self, a: Candidate, b: Candidate) -> Optional[Tuple[float, List[str]]]:
        key = (a, b)
        if key not in self._route_cache:
            self._route_cache[key] = self._compute_route_distance(a, b)
        return self._route_cache[key]

    def _compute_route_distance(self, a: Candidate, b: Candidate) -> Optional[Tuple[float, List[str]]]:
        """
        Legal road-network travel distance from candidate a to candidate b,
        following edge directions. Returns (distance in meters, edge ids) or
        None if b cannot be reached from a by a directed route.
        """
        edge_a = self.graph.edge(a.edge_id)
        edge_b = self.graph.edge(b.edge_id)
        if a.edge_id == b.edge_id and b.offset >= a.offset - 1e-6:
            # Forward along the same (one-way) edge
            return max(0.0, b.offset - a.offset), [a.edge_id]
        middle = self.graph.shortest_path(edge_a.to_node, edge_b.from_node)
        if middle is None:
            return None
        middle_distance, middle_edges = middle
        distance = (edge_a.length - a.offset) + middle_distance + b.offset
        edges = [a.edge_id] + middle_edges + [b.edge_id]
        deduped_edges: List[str] = []
        for edge_id in edges:
            if not deduped_edges or deduped_edges[-1] != edge_id:
                deduped_edges.append(edge_id)
        return distance, deduped_edges

    def _emission_log_score(self, candidate: Candidate) -> float:
        return -0.5 * (candidate.distance / self.gps_sigma) ** 2

    def _transition_log_score(self, route_distance: float, linear_distance: float) -> float:
        return -abs(route_distance - linear_distance) / self.transition_beta


def observations_from_points(points: Iterable[Any]) -> List[Observation]:
    """
    Build observations from gpxpy.gpx.GPXTrackPoint objects (or any objects
    with latitude/longitude attributes and an optional time attribute).
    """
    result: List[Observation] = []
    for point in points:
        result.append(Observation(latitude=point.latitude, longitude=point.longitude,
                                  time=getattr(point, 'time', None)))
    return result
