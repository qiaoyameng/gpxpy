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

"""
Offline map matching of GPS observations against a caller-supplied road
network. No online map service is used: the caller builds a small directed
:class:`RoadGraph` (nodes plus edges with polylines) and decides its size.

For every observation a limited set of projection candidates on nearby edges
is computed (emission probability from the distance to the edge, controlled by
``gps_sigma``). Between consecutive observations only *legal* network routes
are considered: the shortest directed path along edges, including partial edge
traversal when a candidate lies in the middle of an edge. One-way streets are
enforced because edges are directed. The best candidate sequence for the whole
segment is found with a Viterbi dynamic program (Newson-Krumm style HMM),
which is more robust than per-point greedy nearest-road snapping.

When no legal connection exists between two consecutive observations the
sequence is *broken* instead of drawing a fake straight line across the map;
the break (and its reason) is reported in the result. Large time gaps between
observations also break the sequence. Observations too far from every edge are
reported as unmatched with a reason.

The candidate fan-out is bounded by ``search_radius`` and ``max_candidates``,
both controlled by the caller.
"""

import heapq as mod_heapq
import math as mod_math

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from . import geo as mod_geo

LatLon = Tuple[float, float]


class UnmatchedReason(Enum):
    """Why an observation could not be matched to the road network."""
    NO_CANDIDATE_IN_RADIUS = 'no_candidate_in_radius'


class BreakReason(Enum):
    """Why two consecutive observations are not in the same matched sequence."""
    NO_CANDIDATE_IN_RADIUS = 'no_candidate_in_radius'
    NO_FEASIBLE_CONNECTION = 'no_feasible_connection'
    TIME_GAP = 'time_gap'


@dataclass
class RoadEdge:
    """A directed edge of the road network."""
    edge_id: str
    from_node: str
    to_node: str
    polyline: List[LatLon]
    length: float
    cumulative: List[float]
    """cumulative[i] is the length in meters of the polyline up to vertex i."""


class RoadGraph:
    """
    A small directed road network provided by the caller.

    Nodes carry coordinates; edges are directed (a one-way street is a single
    edge, a two-way street is two opposite edges) and carry the polyline
    geometry used for projecting observations. The caller controls the graph
    size.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, LatLon] = {}
        self.edges: Dict[str, RoadEdge] = {}
        self._adjacency: Dict[str, List[Tuple[str, str]]] = {}
        self._shortest_paths: Dict[str, Dict[str, float]] = {}

    def add_node(self, node_id: str, latitude: float, longitude: float) -> None:
        if node_id in self.nodes:
            raise ValueError(f'Duplicate node id: {node_id}')
        self.nodes[node_id] = (latitude, longitude)

    def add_edge(self, edge_id: str, from_node: str, to_node: str,
                 polyline: Sequence[LatLon]) -> None:
        """
        Add a directed edge from ``from_node`` to ``to_node``. ``polyline`` is
        the edge geometry (at least two points); it should start at
        ``from_node`` and end at ``to_node``.
        """
        if edge_id in self.edges:
            raise ValueError(f'Duplicate edge id: {edge_id}')
        if from_node not in self.nodes:
            raise ValueError(f'Unknown node: {from_node}')
        if to_node not in self.nodes:
            raise ValueError(f'Unknown node: {to_node}')
        if len(polyline) < 2:
            raise ValueError('Edge polyline needs at least two points')
        cumulative = [0.0]
        for i in range(1, len(polyline)):
            lat1, lon1 = polyline[i - 1]
            lat2, lon2 = polyline[i]
            cumulative.append(cumulative[-1] + mod_geo.haversine_distance(lat1, lon1, lat2, lon2))
        self.edges[edge_id] = RoadEdge(
            edge_id=edge_id,
            from_node=from_node,
            to_node=to_node,
            polyline=list(polyline),
            length=cumulative[-1],
            cumulative=cumulative,
        )
        self._adjacency.setdefault(from_node, []).append((edge_id, to_node))
        self._shortest_paths.clear()

    def shortest_paths_from(self, from_node: str) -> Dict[str, float]:
        """Dijkstra single-source shortest network distances in meters (cached)."""
        cached = self._shortest_paths.get(from_node)
        if cached is not None:
            return cached
        dist: Dict[str, float] = {from_node: 0.0}
        heap: List[Tuple[float, str]] = [(0.0, from_node)]
        while heap:
            distance, node = mod_heapq.heappop(heap)
            if distance > dist.get(node, mod_math.inf):
                continue
            for edge_id, neighbour in self._adjacency.get(node, []):
                new_distance = distance + self.edges[edge_id].length
                if new_distance < dist.get(neighbour, mod_math.inf):
                    dist[neighbour] = new_distance
                    mod_heapq.heappush(heap, (new_distance, neighbour))
        self._shortest_paths[from_node] = dist
        return dist


@dataclass
class Candidate:
    """A possible projection of one observation onto one edge."""
    edge_id: str
    offset: float
    """Distance in meters along the edge polyline from its start."""
    distance: float
    """Distance in meters between the observation and the projected point."""
    latitude: float
    longitude: float


@dataclass
class PointMatch:
    """The matched position of one observation."""
    latitude: float
    longitude: float
    edge_id: str
    edge_offset: float
    distance_to_edge: float


@dataclass
class MapMatchResult:
    """
    Result of matching a sequence of observations.

    ``matches[i]`` is the matched position of observation ``i`` or ``None``
    when unmatched (see ``unmatched`` for the reason). ``breaks`` maps the
    index of the first observation of a new sub-sequence to the reason the
    previous sequence could not be extended (time gap, missing candidates, or
    no legal network connection -- the matcher never connects such gaps with
    a fake straight line).
    """
    matches: List[Optional[PointMatch]]
    unmatched: Dict[int, UnmatchedReason]
    breaks: Dict[int, BreakReason]

    @property
    def matched_edges(self) -> List[Optional[str]]:
        """Edge id used by every observation (``None`` when unmatched)."""
        return [match.edge_id if match is not None else None for match in self.matches]


class PointLike(Protocol):
    """Anything with WGS84 ``latitude``/``longitude`` attributes (e.g. GPXTrackPoint)."""
    latitude: float
    longitude: float


def _point_time(point: PointLike) -> Optional[datetime]:
    value = getattr(point, 'time', None)
    return value if isinstance(value, datetime) else None


class MapMatcher:
    """
    Matches GPS observations to a :class:`RoadGraph` with a hidden Markov
    model: Gaussian emission on the distance to the edge (``gps_sigma``) and
    an exponential transition on the difference between the legal network
    route length and the great-circle distance (``transition_beta``).
    """

    def __init__(self, graph: RoadGraph, search_radius: float = 50.0,
                 max_candidates: int = 5, gps_sigma: float = 10.0,
                 transition_beta: float = 50.0,
                 max_time_gap_seconds: Optional[float] = None) -> None:
        if search_radius <= 0:
            raise ValueError('search_radius must be positive')
        if max_candidates < 1:
            raise ValueError('max_candidates must be at least 1')
        if gps_sigma <= 0:
            raise ValueError('gps_sigma must be positive')
        if transition_beta <= 0:
            raise ValueError('transition_beta must be positive')
        self.graph = graph
        self.search_radius = search_radius
        self.max_candidates = max_candidates
        self.gps_sigma = gps_sigma
        self.transition_beta = transition_beta
        self.max_time_gap_seconds = max_time_gap_seconds

    def candidates(self, latitude: float, longitude: float) -> List[Candidate]:
        """
        Projections of one observation onto nearby edges, sorted by distance
        and truncated to ``max_candidates``. May be empty when no edge is
        within ``search_radius``.
        """
        result: List[Candidate] = []
        for edge in self.graph.edges.values():
            candidate = self._project_on_edge(edge, latitude, longitude)
            if candidate.distance <= self.search_radius:
                result.append(candidate)
        result.sort(key=lambda candidate: candidate.distance)
        return result[:self.max_candidates]

    def match(self, points: Sequence[PointLike]) -> MapMatchResult:
        """
        Match a sequence of observations (anything with ``latitude`` and
        ``longitude`` attributes, e.g. ``GPXTrackPoint``) to the road graph.
        """
        latitudes = [point.latitude for point in points]
        longitudes = [point.longitude for point in points]
        times = [_point_time(point) for point in points]
        count = len(points)

        candidate_lists: List[Optional[List[Candidate]]] = []
        unmatched: Dict[int, UnmatchedReason] = {}
        for i in range(count):
            candidates = self.candidates(latitudes[i], longitudes[i])
            if candidates:
                candidate_lists.append(candidates)
            else:
                candidate_lists.append(None)
                unmatched[i] = UnmatchedReason.NO_CANDIDATE_IN_RADIUS

        matches: List[Optional[PointMatch]] = [None] * count
        breaks: Dict[int, BreakReason] = {}

        # Split into segments of consecutive observations that all have
        # candidates and no too-large time gap:
        i = 0
        while i < count:
            if candidate_lists[i] is None:
                i += 1
                continue
            start = i
            if start > 0:
                breaks[start] = (BreakReason.NO_CANDIDATE_IN_RADIUS
                                 if candidate_lists[start - 1] is None
                                 else BreakReason.TIME_GAP)
            i += 1
            while i < count and candidate_lists[i] is not None \
                    and not self._time_gap_exceeded(times[i - 1], times[i]):
                i += 1
            self._match_segment(latitudes, longitudes, candidate_lists, start, i, matches, breaks)

        return MapMatchResult(matches=matches, unmatched=unmatched, breaks=breaks)

    def _time_gap_exceeded(self, time1: Optional[datetime], time2: Optional[datetime]) -> bool:
        if self.max_time_gap_seconds is None or time1 is None or time2 is None:
            return False
        return (time2 - time1).total_seconds() > self.max_time_gap_seconds

    def _project_on_edge(self, edge: RoadEdge, latitude: float, longitude: float) -> Candidate:
        cos_lat = mod_math.cos(mod_math.radians(latitude))
        best_distance = mod_math.inf
        best_offset = 0.0
        best_latitude = edge.polyline[0][0]
        best_longitude = edge.polyline[0][1]
        for i in range(len(edge.polyline) - 1):
            lat1, lon1 = edge.polyline[i]
            lat2, lon2 = edge.polyline[i + 1]
            # Equirectangular projection to meters around the observation:
            ax = (lon1 - longitude) * mod_geo.ONE_DEGREE * cos_lat
            ay = (lat1 - latitude) * mod_geo.ONE_DEGREE
            bx = (lon2 - longitude) * mod_geo.ONE_DEGREE * cos_lat
            by = (lat2 - latitude) * mod_geo.ONE_DEGREE
            dx = bx - ax
            dy = by - ay
            segment_length_sq = dx * dx + dy * dy
            if segment_length_sq == 0.0:
                t = 0.0
            else:
                t = max(0.0, min(1.0, -(ax * dx + ay * dy) / segment_length_sq))
            px = ax + t * dx
            py = ay + t * dy
            distance = mod_math.hypot(px, py)
            if distance < best_distance:
                best_distance = distance
                best_offset = edge.cumulative[i] + t * mod_math.sqrt(segment_length_sq)
                best_latitude = latitude + py / mod_geo.ONE_DEGREE
                best_longitude = longitude + px / (mod_geo.ONE_DEGREE * cos_lat)
        return Candidate(edge_id=edge.edge_id, offset=best_offset, distance=best_distance,
                         latitude=best_latitude, longitude=best_longitude)

    def _route_distance(self, first: Candidate, second: Candidate) -> float:
        """
        Shortest *legal* network distance in meters between two candidates,
        respecting edge directions and partial edge traversal. ``inf`` when
        there is no legal connection.
        """
        edge1 = self.graph.edges[first.edge_id]
        edge2 = self.graph.edges[second.edge_id]
        if first.edge_id == second.edge_id and second.offset >= first.offset:
            return second.offset - first.offset
        tail = edge1.length - first.offset
        middle = self.graph.shortest_paths_from(edge1.to_node).get(edge2.from_node, mod_math.inf)
        return tail + middle + second.offset

    def _emission(self, candidate: Candidate) -> float:
        return -0.5 * (candidate.distance / self.gps_sigma) ** 2

    def _match_segment(self, latitudes: List[float], longitudes: List[float],
                       candidate_lists: List[Optional[List[Candidate]]],
                       start: int, end: int,
                       matches: List[Optional[PointMatch]],
                       breaks: Dict[int, BreakReason]) -> None:
        columns: List[List[Candidate]] = []
        for i in range(start, end):
            candidates = candidate_lists[i]
            assert candidates is not None
            columns.append(candidates)
        count = len(columns)

        emissions = [[self._emission(candidate) for candidate in column] for column in columns]
        dps: List[List[float]] = []
        parents: List[List[int]] = []
        chain_starts = [0]
        dp = list(emissions[0])
        dps.append(dp)
        parents.append([-1] * len(columns[0]))
        for k in range(1, count):
            great_circle = mod_geo.haversine_distance(
                latitudes[start + k - 1], longitudes[start + k - 1],
                latitudes[start + k], longitudes[start + k])
            new_dp = [-mod_math.inf] * len(columns[k])
            new_parents = [-1] * len(columns[k])
            for j, candidate_b in enumerate(columns[k]):
                best_score = -mod_math.inf
                best_a = -1
                for a, candidate_a in enumerate(columns[k - 1]):
                    if dp[a] == -mod_math.inf:
                        continue
                    route = self._route_distance(candidate_a, candidate_b)
                    if mod_math.isinf(route):
                        continue
                    score = dp[a] - abs(route - great_circle) / self.transition_beta
                    if score > best_score:
                        best_score = score
                        best_a = a
                if best_a >= 0:
                    new_dp[j] = best_score + emissions[k][j]
                    new_parents[j] = best_a
            if all(score == -mod_math.inf for score in new_dp):
                # No legal network connection between the two columns: break
                # the sequence instead of drawing a fake straight line:
                breaks[start + k] = BreakReason.NO_FEASIBLE_CONNECTION
                chain_starts.append(k)
                new_parents = [-1] * len(columns[k])
                new_dp = list(emissions[k])
            dps.append(new_dp)
            parents.append(new_parents)
            dp = new_dp

        chain_ends = chain_starts[1:] + [count]
        for chain_index, chain_start in enumerate(chain_starts):
            chain_end = chain_ends[chain_index]
            best_j = max(range(len(columns[chain_end - 1])),
                         key=lambda j: dps[chain_end - 1][j])
            for k in range(chain_end - 1, chain_start - 1, -1):
                chosen = columns[k][best_j]
                matches[start + k] = PointMatch(
                    latitude=chosen.latitude, longitude=chosen.longitude,
                    edge_id=chosen.edge_id, edge_offset=chosen.offset,
                    distance_to_edge=chosen.distance)
                best_j = parents[k][best_j]
