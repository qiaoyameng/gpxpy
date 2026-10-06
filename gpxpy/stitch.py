"""
Stitch GPX track segments whose order was scrambled by the exporting device.

Given a list of ``GPXTrackSegment`` objects, a maximum join distance and a
time tolerance, :func:`stitch_segments` computes an *optimal* set of
connections between segment endpoints and returns ordered tracks together
with a human-readable explanation for every connection.

Rules implemented here:

* Every input segment is used at most once, and every segment endpoint
  is part of at most one connection (connections form a matching on
  endpoints).
* A segment is only reported as reversed when the time constraints allow
  it. Segments carrying timestamps are never reversed unless
  ``allow_timed_reversal=True``, because reversing them would make the
  recorded timestamps run backwards. Untimed segments may be reversed
  freely.
* Connections never cross clearly regressive time: if the time at the
  outgoing endpoint is later than the time at the incoming endpoint plus
  ``time_tolerance``, the connection is forbidden, no matter how close the
  two endpoints are geographically.
* The chosen connections are globally optimal, not greedy: first the
  number of joins is maximized, then the total join distance is
  minimized. A closest-endpoint-first greedy strategy can miss this
  optimum.
* Connections never form a closed loop. A loop would consume both
  endpoints of every segment in it, leaving no free ends, and the travel
  direction (and hence the time ordering) of the loop would be
  ambiguous. While searching, a union-find structure rejects any
  candidate edge whose two segments are already connected.
* When several solutions are equally good (same number of joins and same
  total distance), the result is marked as ambiguous. The primary
  solution is the lexicographically smallest one (deterministic), and
  every equivalent alternative is listed in canonical sorted order in
  ``StitchResult.alternatives``.

Input segments are never mutated: results reference the input segments by
index plus a ``reversed`` flag, and :meth:`StitchedTrack.materialize`
builds brand new ``GPXTrackSegment`` objects on request.

The solver is an exact branch-and-bound over candidate endpoint pairs, so
it is exponential in the worst case. It is meant for the typical
device-export situation of a handful to a few dozen scrambled segments.
"""

import datetime as mod_datetime
import sys as mod_sys
from typing import FrozenSet, List, Optional, Sequence, Set, Tuple

from . import geo as mod_geo
from . import gpx as mod_gpx

START = 'start'
END = 'end'

#: Two total distances differing by less than this are considered equal
#: (used both for pruning and for detecting equivalent optima).
_EPSILON = 1e-6


class JoinDecision:
    """
    Explains one connection between two segment endpoints.

    ``from_segment`` / ``to_segment`` are indices into the input segment
    list, ``from_side`` / ``to_side`` tell which endpoint of the *original*
    (user supplied) segment is connected (``'start'`` or ``'end'``), and
    ``from_reversed`` / ``to_reversed`` tell whether each segment is used
    in reverse inside the stitched track.
    """

    def __init__(self, from_segment: int, from_side: str, to_segment: int, to_side: str,
                 distance: float,
                 from_time: Optional[mod_datetime.datetime],
                 to_time: Optional[mod_datetime.datetime],
                 from_reversed: bool, to_reversed: bool) -> None:
        self.from_segment = from_segment
        self.from_side = from_side
        self.to_segment = to_segment
        self.to_side = to_side
        self.distance = distance
        self.from_time = from_time
        self.to_time = to_time
        self.from_reversed = from_reversed
        self.to_reversed = to_reversed

    def describe(self) -> str:
        """ Human readable, one line explanation of this connection. """
        parts = [
            f'segment[{self.from_segment}].{self.from_side}'
            f'{" (reversed)" if self.from_reversed else ""}',
            '->',
            f'segment[{self.to_segment}].{self.to_side}'
            f'{" (reversed)" if self.to_reversed else ""}',
            f'gap={self.distance:.1f}m',
        ]
        if self.from_time is not None and self.to_time is not None:
            parts.append(f'time {self.from_time} -> {self.to_time}')
        elif self.from_time is None and self.to_time is None:
            parts.append('no time data')
        else:
            parts.append('partial time data')
        return ' '.join(parts)

    def __repr__(self) -> str:
        return f'JoinDecision({self.describe()})'


class StitchedTrack:
    """
    One ordered chain of input segments.

    ``segment_indices`` are indices into the input list, in travel order.
    ``reversed_flags[i]`` is ``True`` when ``segment_indices[i]`` must be
    walked from its last point to its first point. ``joins`` explains the
    connections between consecutive segments.
    """

    def __init__(self, segment_indices: List[int], reversed_flags: List[bool],
                 joins: List[JoinDecision]) -> None:
        self.segment_indices = segment_indices
        self.reversed_flags = reversed_flags
        self.joins = joins

    def materialize(self, segments: Sequence[mod_gpx.GPXTrackSegment]) -> mod_gpx.GPXTrackSegment:
        """
        Build a new ``GPXTrackSegment`` with the points of this chain in
        travel order. The input segments are never modified; the returned
        segment reuses the existing point objects (they are not copied).
        """
        points: List[mod_gpx.GPXTrackPoint] = []
        for index, reversed_flag in zip(self.segment_indices, self.reversed_flags):
            segment_points = segments[index].points
            if reversed_flag:
                points.extend(reversed(segment_points))
            else:
                points.extend(segment_points)
        return mod_gpx.GPXTrackSegment(points=points)

    def __repr__(self) -> str:
        return f'StitchedTrack({self.segment_indices}, reversed={self.reversed_flags})'


class ExcludedSegment:
    """ An input segment that could not take part in stitching at all. """

    def __init__(self, index: int, reason: str) -> None:
        self.index = index
        self.reason = reason

    def __repr__(self) -> str:
        return f'ExcludedSegment({self.index}, {self.reason!r})'


#: Canonical description of one connection in an alternative solution:
#: (segment_a, side_a, segment_b, side_b, distance) with segment_a < segment_b.
AlternativeJoin = Tuple[int, str, int, str, float]


class StitchResult:
    """
    The outcome of :func:`stitch_segments`.

    Every input segment is accounted for exactly once: either its index
    appears in exactly one of ``tracks`` (a track may be a single,
    unjoined segment), or it is listed in ``excluded`` with a reason.

    ``ambiguous`` is ``True`` when more than one globally optimal solution
    exists. ``tracks`` always describes the deterministic primary solution
    (lexicographically smallest); ``alternatives`` lists the join sets of
    the other equally good solutions in canonical sorted order.
    """

    def __init__(self, tracks: List[StitchedTrack], excluded: List[ExcludedSegment],
                 ambiguous: bool, alternatives: List[List[AlternativeJoin]],
                 total_join_distance: float) -> None:
        self.tracks = tracks
        self.excluded = excluded
        self.ambiguous = ambiguous
        self.alternatives = alternatives
        self.total_join_distance = total_join_distance

    @property
    def joins_no(self) -> int:
        """ Total number of connections in the primary solution. """
        return sum(len(track.joins) for track in self.tracks)

    def describe(self) -> str:
        """ Human readable summary of the whole stitching result. """
        lines = []
        for track_no, track in enumerate(self.tracks):
            lines.append(f'track {track_no}: segments {track.segment_indices} '
                         f'reversed={track.reversed_flags}')
            for join in track.joins:
                lines.append(f'  join: {join.describe()}')
        for excluded in self.excluded:
            lines.append(f'excluded: segment[{excluded.index}]: {excluded.reason}')
        if self.ambiguous:
            lines.append(f'ambiguous: {len(self.alternatives)} equivalent alternative(s)')
            for alternative in self.alternatives:
                lines.append(f'  alternative: {alternative}')
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return (f'StitchResult(tracks={self.tracks}, excluded={self.excluded}, '
                f'ambiguous={self.ambiguous})')


class _SegmentInfo:
    """ Precomputed endpoint data for one input segment. """

    __slots__ = ('start_latitude', 'start_longitude', 'end_latitude', 'end_longitude',
                 'start_time', 'end_time', 'timed')

    def __init__(self, segment: mod_gpx.GPXTrackSegment) -> None:
        first = segment.points[0]
        last = segment.points[-1]
        self.start_latitude = first.latitude
        self.start_longitude = first.longitude
        self.end_latitude = last.latitude
        self.end_longitude = last.longitude
        self.start_time = first.time
        self.end_time = last.time
        self.timed = first.time is not None or last.time is not None

    def time_at(self, side: str) -> Optional[mod_datetime.datetime]:
        return self.start_time if side == START else self.end_time


class _CandidateEdge:
    """ A possible connection between two segment endpoints. """

    __slots__ = ('edge_id', 'segment_a', 'side_a', 'segment_b', 'side_b', 'distance')

    def __init__(self, edge_id: int, segment_a: int, side_a: str,
                 segment_b: int, side_b: str, distance: float) -> None:
        self.edge_id = edge_id
        self.segment_a = segment_a
        self.side_a = side_a
        self.segment_b = segment_b
        self.side_b = side_b
        self.distance = distance

    def side_of(self, segment_index: int) -> str:
        return self.side_a if segment_index == self.segment_a else self.side_b

    def other(self, segment_index: int) -> int:
        return self.segment_b if segment_index == self.segment_a else self.segment_a

    def canonical(self) -> AlternativeJoin:
        return (self.segment_a, self.side_a, self.segment_b, self.side_b, self.distance)


def _time_delta(later: mod_datetime.datetime,
                earlier: mod_datetime.datetime) -> mod_datetime.timedelta:
    """
    Difference between two datetimes. If one is timezone aware and the
    other naive, fall back to comparing wall-clock times.
    """
    try:
        return later - earlier
    except TypeError:
        return later.replace(tzinfo=None) - earlier.replace(tzinfo=None)


def _joint_time_ok(out_time: Optional[mod_datetime.datetime],
                   in_time: Optional[mod_datetime.datetime],
                   tolerance: mod_datetime.timedelta) -> bool:
    """
    A joint is allowed unless time clearly goes backwards: the outgoing
    time must not be later than the incoming time plus the tolerance.
    Missing times impose no constraint.
    """
    if out_time is None or in_time is None:
        return True
    return _time_delta(out_time, in_time) <= tolerance


def _chain_orientations_ok(infos: List[_SegmentInfo], chain: List[int],
                           joints: List[Tuple[str, str]],
                           tolerance: mod_datetime.timedelta,
                           allow_timed_reversal: bool) -> bool:
    """
    Check one travel direction of one chain. ``chain`` lists segment
    indices in travel order and ``joints[k]`` is the (out side of
    chain[k], in side of chain[k+1]) pair. Reversed flags are derived
    from the touched sides.
    """
    for position, segment_index in enumerate(chain):
        info = infos[segment_index]
        if info.timed and not allow_timed_reversal:
            reversed_flag = False
            if position > 0 and joints[position - 1][1] == END:
                reversed_flag = True
            if position < len(joints) and joints[position][0] == START:
                reversed_flag = True
            if reversed_flag:
                return False
    for position, (out_side, in_side) in enumerate(joints):
        out_time = infos[chain[position]].time_at(out_side)
        in_time = infos[chain[position + 1]].time_at(in_side)
        if not _joint_time_ok(out_time, in_time, tolerance):
            return False
    return True


def _chain_feasible(infos: List[_SegmentInfo], chain: List[int],
                    joints: List[Tuple[str, str]],
                    tolerance: mod_datetime.timedelta,
                    allow_timed_reversal: bool) -> bool:
    """
    A chain is feasible when at least one of its two travel directions
    satisfies the reversal and time constraints.
    """
    if _chain_orientations_ok(infos, chain, joints, tolerance, allow_timed_reversal):
        return True
    reversed_chain = list(reversed(chain))
    reversed_joints = [(in_side, out_side) for out_side, in_side in reversed(joints)]
    return _chain_orientations_ok(infos, reversed_chain, reversed_joints,
                                  tolerance, allow_timed_reversal)


class _UnionFind:
    """ Union-find without path compression so unions can be rolled back. """

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> Optional[Tuple[int, int, bool]]:
        """
        Merge the sets of two elements. Returns a rollback token, or
        ``None`` when both are already connected (adding such an edge
        would close a cycle).
        """
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return None
        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        incremented = False
        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1
            incremented = True
        return root_first, root_second, incremented

    def rollback(self, token: Tuple[int, int, bool]) -> None:
        root, child, incremented = token
        self.parent[child] = child
        if incremented:
            self.rank[root] -= 1


def _extract_chain(adjacency: List[List[int]], edges: List[_CandidateEdge],
                   start_segment: int) -> Tuple[List[int], List[Tuple[str, str]], List[int]]:
    """
    Extract the path containing ``start_segment`` from the current
    adjacency. Returns the segment indices in path order, the
    (out side, in side) pair of every joint, and the edge id of every
    joint.
    """
    # Walk to one end of the path first:
    end = start_segment
    previous_edge = -1
    while True:
        next_edges = [edge_id for edge_id in adjacency[end] if edge_id != previous_edge]
        if not next_edges:
            break
        previous_edge = next_edges[0]
        end = edges[previous_edge].other(end)
    # Then walk the whole path from that end:
    chain = [end]
    joints: List[Tuple[str, str]] = []
    joint_edges: List[int] = []
    previous_edge = -1
    node = end
    while True:
        next_edges = [edge_id for edge_id in adjacency[node] if edge_id != previous_edge]
        if not next_edges:
            break
        edge_id = next_edges[0]
        edge = edges[edge_id]
        other = edge.other(node)
        joints.append((edge.side_of(node), edge.side_of(other)))
        joint_edges.append(edge_id)
        chain.append(other)
        node = other
        previous_edge = edge_id
    return chain, joints, joint_edges


class _Solver:
    """
    Exact branch-and-bound over candidate edges. The objective is
    lexicographic: first maximize the number of joins, then minimize the
    total join distance. All optimal solutions are collected so ties can
    be reported as ambiguity.
    """

    def __init__(self, infos: List[_SegmentInfo], edges: List[_CandidateEdge],
                 tolerance: mod_datetime.timedelta,
                 allow_timed_reversal: bool) -> None:
        self.infos = infos
        self.edges = edges
        self.tolerance = tolerance
        self.allow_timed_reversal = allow_timed_reversal
        segment_no = len(infos)
        self.used_endpoints = [False] * (2 * segment_no)
        self.components = _UnionFind(segment_no)
        self.adjacency: List[List[int]] = [[] for _ in range(segment_no)]
        self.chosen: List[int] = []
        self.chosen_distance = 0.0
        self.best_count = 0
        self.best_distance = 0.0
        self.solutions: List[FrozenSet[int]] = [frozenset()]

    @staticmethod
    def _endpoint_id(segment_index: int, side: str) -> int:
        return 2 * segment_index + (0 if side == START else 1)

    def run(self) -> None:
        self._search(0)

    def _record(self) -> None:
        count = len(self.chosen)
        if count > self.best_count \
                or (count == self.best_count
                    and self.chosen_distance < self.best_distance - _EPSILON):
            self.best_count = count
            self.best_distance = self.chosen_distance
            self.solutions = []
        if count == self.best_count \
                and abs(self.chosen_distance - self.best_distance) <= _EPSILON:
            solution = frozenset(self.chosen)
            if solution not in self.solutions:
                self.solutions.append(solution)

    def _search(self, edge_pos: int) -> None:
        # Upper bound on the number of joins still achievable; distances
        # are non-negative, so when the join count can no longer improve,
        # an already worse distance can never catch up either.
        free_endpoints = self.used_endpoints.count(False)
        max_count = len(self.chosen) + free_endpoints // 2
        if max_count < self.best_count:
            return
        if max_count == self.best_count \
                and self.chosen_distance > self.best_distance + _EPSILON:
            return
        if edge_pos == len(self.edges):
            self._record()
            return
        edge = self.edges[edge_pos]
        endpoint_a = self._endpoint_id(edge.segment_a, edge.side_a)
        endpoint_b = self._endpoint_id(edge.segment_b, edge.side_b)
        if not self.used_endpoints[endpoint_a] and not self.used_endpoints[endpoint_b]:
            token = self.components.union(edge.segment_a, edge.segment_b)
            if token is not None:
                self.used_endpoints[endpoint_a] = True
                self.used_endpoints[endpoint_b] = True
                self.adjacency[edge.segment_a].append(edge.edge_id)
                self.adjacency[edge.segment_b].append(edge.edge_id)
                self.chosen.append(edge.edge_id)
                self.chosen_distance += edge.distance
                # Infeasibility is monotone: once a chain violates the
                # time/reversal rules, extending it can never fix that,
                # because existing joints keep their sides. So an
                # infeasible chain prunes the whole branch.
                if self._merged_chain_feasible(edge.segment_a):
                    self._search(edge_pos + 1)
                self.chosen_distance -= edge.distance
                self.chosen.pop()
                self.adjacency[edge.segment_a].pop()
                self.adjacency[edge.segment_b].pop()
                self.used_endpoints[endpoint_a] = False
                self.used_endpoints[endpoint_b] = False
                self.components.rollback(token)
        self._search(edge_pos + 1)

    def _merged_chain_feasible(self, segment_index: int) -> bool:
        chain, joints, _ = _extract_chain(self.adjacency, self.edges, segment_index)
        return _chain_feasible(self.infos, chain, joints,
                               self.tolerance, self.allow_timed_reversal)


def _find_candidates(infos: List[_SegmentInfo], indices: List[int],
                     max_join_distance: float) -> List[_CandidateEdge]:
    """ All endpoint pairs within ``max_join_distance``, sorted by distance. """
    raw: List[Tuple[float, int, str, int, str]] = []
    for position_a, index_a in enumerate(indices):
        for index_b in indices[position_a + 1:]:
            info_a = infos[index_a]
            info_b = infos[index_b]
            for side_a, latitude_a, longitude_a in (
                    (START, info_a.start_latitude, info_a.start_longitude),
                    (END, info_a.end_latitude, info_a.end_longitude)):
                for side_b, latitude_b, longitude_b in (
                        (START, info_b.start_latitude, info_b.start_longitude),
                        (END, info_b.end_latitude, info_b.end_longitude)):
                    distance = mod_geo.haversine_distance(
                            latitude_a, longitude_a, latitude_b, longitude_b)
                    if distance <= max_join_distance:
                        raw.append((distance, index_a, side_a, index_b, side_b))
    raw.sort()
    return [_CandidateEdge(edge_id, segment_a, side_a, segment_b, side_b, distance)
            for edge_id, (distance, segment_a, side_a, segment_b, side_b)
            in enumerate(raw)]


def _build_tracks(infos: List[_SegmentInfo], indices: List[int],
                  edges: List[_CandidateEdge], solution: FrozenSet[int],
                  tolerance: mod_datetime.timedelta,
                  allow_timed_reversal: bool) -> List[StitchedTrack]:
    adjacency: List[List[int]] = [[] for _ in infos]
    for edge_id in solution:
        edge = edges[edge_id]
        adjacency[edge.segment_a].append(edge_id)
        adjacency[edge.segment_b].append(edge_id)
    tracks: List[StitchedTrack] = []
    visited: Set[int] = set()
    for index in indices:
        if index in visited:
            continue
        chain, joints, joint_edges = _extract_chain(adjacency, edges, index)
        visited.update(chain)
        reversed_chain = list(reversed(chain))
        reversed_joints = [(in_side, out_side) for out_side, in_side in reversed(joints)]
        forward_ok = _chain_orientations_ok(infos, chain, joints,
                                            tolerance, allow_timed_reversal)
        backward_ok = _chain_orientations_ok(infos, reversed_chain, reversed_joints,
                                             tolerance, allow_timed_reversal)
        # Both directions may be fine (e.g. untimed chains); pick
        # deterministically so repeated runs report identical tracks.
        if backward_ok and (not forward_ok or tuple(reversed_chain) < tuple(chain)):
            chain = reversed_chain
            joints = reversed_joints
            joint_edges = list(reversed(joint_edges))
        reversed_flags: List[bool] = []
        for position in range(len(chain)):
            reversed_flag = False
            if position > 0 and joints[position - 1][1] == END:
                reversed_flag = True
            if position < len(joints) and joints[position][0] == START:
                reversed_flag = True
            reversed_flags.append(reversed_flag)
        joins: List[JoinDecision] = []
        for position, ((out_side, in_side), edge_id) in enumerate(zip(joints, joint_edges)):
            edge = edges[edge_id]
            joins.append(JoinDecision(
                    from_segment=chain[position], from_side=out_side,
                    to_segment=chain[position + 1], to_side=in_side,
                    distance=edge.distance,
                    from_time=infos[chain[position]].time_at(out_side),
                    to_time=infos[chain[position + 1]].time_at(in_side),
                    from_reversed=reversed_flags[position],
                    to_reversed=reversed_flags[position + 1]))
        tracks.append(StitchedTrack(chain, reversed_flags, joins))
    tracks.sort(key=lambda track: min(track.segment_indices))
    return tracks


def stitch_segments(segments: Sequence[mod_gpx.GPXTrackSegment],
                    max_join_distance: float,
                    time_tolerance: mod_datetime.timedelta = mod_datetime.timedelta(0),
                    allow_timed_reversal: bool = False) -> StitchResult:
    """
    Stitch scrambled track segments into ordered tracks.

    Parameters
    ----------
    segments : list of GPXTrackSegment
        The segments in arbitrary (scrambled) order. Never modified.
    max_join_distance : float
        Maximum distance in meters between two endpoints for a connection
        to be considered at all.
    time_tolerance : datetime.timedelta
        How far the time at a joint may go backwards before the
        connection is forbidden. Zero (the default) means any regression
        vetoes the joint, regardless of distance.
    allow_timed_reversal : bool
        Allow reversing segments that carry timestamps. Off by default,
        because reversing them makes recorded times run backwards.

    Returns
    ----------
    result : StitchResult
        Ordered tracks, per-join explanations, excluded segments and
        ambiguity information. Every input segment appears exactly once,
        either in one of the tracks or in ``excluded``.
    """
    infos: List[_SegmentInfo] = []
    indices: List[int] = []
    excluded: List[ExcludedSegment] = []
    for index, segment in enumerate(segments):
        if not segment.points:
            infos.append(_SegmentInfo(mod_gpx.GPXTrackSegment(
                    points=[mod_gpx.GPXTrackPoint()])))
            excluded.append(ExcludedSegment(index, 'segment has no points'))
        else:
            infos.append(_SegmentInfo(segment))
            indices.append(index)
    edges = _find_candidates(infos, indices, max_join_distance)
    solver = _Solver(infos, edges, time_tolerance, allow_timed_reversal)
    # The search recursion is one level deep per candidate edge:
    needed_limit = 2 * len(edges) + 100
    old_limit = mod_sys.getrecursionlimit()
    if needed_limit > old_limit:
        mod_sys.setrecursionlimit(needed_limit)
    try:
        solver.run()
    finally:
        mod_sys.setrecursionlimit(old_limit)
    # Deterministic primary solution: lexicographically smallest canonical
    # join set among all equally optimal solutions.
    def canonical_key(solution: FrozenSet[int]) -> Tuple[AlternativeJoin, ...]:
        return tuple(sorted(edges[edge_id].canonical() for edge_id in solution))
    ordered = sorted(solver.solutions, key=canonical_key)
    primary = ordered[0]
    alternatives = [list(canonical_key(solution)) for solution in ordered[1:]]
    tracks = _build_tracks(infos, indices, edges, primary,
                           time_tolerance, allow_timed_reversal)
    total_distance = sum(edges[edge_id].distance for edge_id in primary)
    return StitchResult(tracks=tracks, excluded=excluded,
                        ambiguous=len(ordered) > 1,
                        alternatives=alternatives,
                        total_join_distance=total_distance)
