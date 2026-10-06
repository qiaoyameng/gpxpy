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

"""Reassemble independently exported track segments."""

import datetime as mod_datetime

from typing import NamedTuple, Optional, Sequence

from . import gpx as mod_gpx
from . import utils as mod_utils


class SegmentStitchConnection(NamedTuple):
    from_segment_index: int
    to_segment_index: int
    from_reversed: bool
    to_reversed: bool
    distance: float
    time_gap_seconds: Optional[float]


class SegmentStitchTrack(NamedTuple):
    track: mod_gpx.GPXTrack
    segment_indices: tuple[int, ...]
    reversed_segments: tuple[bool, ...]
    connections: tuple[SegmentStitchConnection, ...]
    total_distance: float
    total_time_gap_seconds: Optional[float]


class ExcludedSegment(NamedTuple):
    segment_index: int
    reason: str


class SegmentStitchResult(NamedTuple):
    tracks: tuple[SegmentStitchTrack, ...]
    connections: tuple[SegmentStitchConnection, ...]
    excluded_segments: tuple[ExcludedSegment, ...]
    ambiguous: bool
    optimal_solution_count: int


class _Orientation(NamedTuple):
    segment_index: int
    reversed: bool
    start: mod_gpx.GPXTrackPoint
    end: mod_gpx.GPXTrackPoint
    singleton: bool
    endpoint_mask: int


class _Path(NamedTuple):
    cost: tuple[int, float, float, tuple[tuple[int, ...], ...]]
    nodes: tuple[_Orientation, ...]
    count: int


class _Cover(NamedTuple):
    cost: tuple[int, float, float, tuple[tuple[int, ...], ...]]
    paths: tuple[tuple[_Orientation, ...], ...]
    count: int
    segment_mask: int
    endpoint_mask: int


def _time_tolerance_seconds(
        time_tolerance: float | mod_datetime.timedelta | None) -> Optional[float]:
    if time_tolerance is None:
        return None
    if isinstance(time_tolerance, mod_datetime.timedelta):
        return time_tolerance.total_seconds()
    if not mod_utils.is_numeric(time_tolerance):
        raise TypeError('time_tolerance must be a number of seconds, timedelta, or None')
    return float(time_tolerance)


def _gap_seconds(
        left: mod_gpx.GPXTrackPoint,
        right: mod_gpx.GPXTrackPoint) -> Optional[float]:
    if left.time is None or right.time is None:
        return None
    return mod_utils.total_seconds(right.time - left.time)


def _can_compare_times(left: mod_gpx.GPXTrackPoint, right: mod_gpx.GPXTrackPoint) -> bool:
    if left.time is None or right.time is None:
        return True
    try:
        right.time - left.time
    except TypeError:
        return False
    return True


def _not_before(left_time: Optional[mod_datetime.datetime],
                right_time: Optional[mod_datetime.datetime]) -> bool:
    if left_time is None or right_time is None:
        return True
    try:
        return right_time >= left_time
    except TypeError:
        return False


def _is_valid_orientation(segment: mod_gpx.GPXTrackSegment, reverse: bool) -> bool:
    points = segment.points
    if reverse:
        points = list(reversed(points))

    previous_time: Optional[mod_datetime.datetime] = None
    for point in points:
        if not _not_before(previous_time, point.time):
            return False
        if point.time is not None:
            previous_time = point.time
    return True


def _path_signature(nodes: Sequence[_Orientation]) -> tuple[tuple[int, ...], ...]:
    return tuple((node.segment_index, int(node.reversed)) for node in nodes)


def _connection(
        left: _Orientation,
        right: _Orientation,
        distance: float,
        gap: Optional[float]) -> SegmentStitchConnection:
    return SegmentStitchConnection(
        left.segment_index,
        right.segment_index,
        left.reversed,
        right.reversed,
        distance,
        gap,
    )


def _oriented_points(
        segments: Sequence[mod_gpx.GPXTrackSegment],
        node: _Orientation) -> list[mod_gpx.GPXTrackPoint]:
    points = segments[node.segment_index].points
    if node.reversed:
        return list(reversed(points))
    return list(points)


def stitch_segments(
        segments: Sequence[mod_gpx.GPXTrackSegment],
        max_join_distance: float,
        time_tolerance: float | mod_datetime.timedelta | None = None) -> SegmentStitchResult:
    """
    Find the minimum-cost path cover of *segments*.

    Every valid segment appears in exactly one resulting track. Candidates are
    chosen globally, so a locally nearest endpoint cannot consume an endpoint
    needed by a better complete chain. Each path DP state appends only unused
    segments, and the path-cover DP combines disjoint paths. Consequently, no
    candidate cycle can be selected.

    A reversed orientation is included only when the segment's own timestamps
    do not move backwards. Join timestamps also may not move backwards, and
    forward gaps must not exceed ``time_tolerance``; missing timestamps simply
    do not constrain that edge. The exact optimization is exponential in the
    number of candidate segments, making it suitable for fragment-sized
    exports rather than very large segment collections.
    """

    if not mod_utils.is_numeric(max_join_distance) or max_join_distance < 0:
        raise ValueError('max_join_distance must be a non-negative number')

    tolerance = _time_tolerance_seconds(time_tolerance)
    if tolerance is not None and tolerance < 0:
        raise ValueError('time_tolerance must be non-negative')

    orientations_by_segment: list[list[_Orientation]] = []
    excluded: list[ExcludedSegment] = []
    segment_count = len(segments)
    endpoint_count = 0

    for segment_index, segment in enumerate(segments):
        points = segment.points
        if not points:
            excluded.append(ExcludedSegment(segment_index, 'segment has no points'))
            orientations_by_segment.append([])
            continue

        options: list[_Orientation] = []
        singleton = len(points) == 1
        start = points[0]
        end = points[-1]
        start_port_id = endpoint_count
        end_port_id = endpoint_count + 1
        endpoint_count += 2
        if singleton:
            endpoint_mask = 1 << start_port_id
        else:
            endpoint_mask = (1 << start_port_id) | (1 << end_port_id)

        if _is_valid_orientation(segment, False):
            options.append(_Orientation(segment_index, False, start, end, singleton, endpoint_mask))
        if start is not end and _is_valid_orientation(segment, True):
            options.append(_Orientation(segment_index, True, end, start, singleton, endpoint_mask))

        if not options:
            excluded.append(ExcludedSegment(
                segment_index,
                'no orientation has chronologically non-decreasing timestamps'))

        orientations_by_segment.append(options)

    candidate_distance: dict[tuple[int, int], float] = {}
    candidate_gap: dict[tuple[int, int], Optional[float]] = {}

    for left_options in orientations_by_segment:
        for left in left_options:
            for right_options in orientations_by_segment:
                for right in right_options:
                    if left.segment_index == right.segment_index:
                        continue
                    if left.end is right.start:
                        distance = 0.0
                    else:
                        calculated_distance = left.end.distance_2d(right.start)
                        if calculated_distance is None:
                            continue
                        distance = calculated_distance
                    if distance > max_join_distance:
                        continue

                    if not _can_compare_times(left.end, right.start):
                        continue
                    gap = _gap_seconds(left.end, right.start)
                    if gap is not None and gap < 0:
                        continue
                    if gap is not None and tolerance is not None and gap > tolerance:
                        continue

                    key = (id(left), id(right))
                    if key not in candidate_distance or distance < candidate_distance[key]:
                        candidate_distance[key] = distance
                        candidate_gap[key] = gap

    paths: dict[tuple[int, int, int], _Path] = {}
    all_orientations = [
        orientation
        for options in orientations_by_segment
        for orientation in options
    ]

    for orientation in all_orientations:
        signature = ((orientation.segment_index, int(orientation.reversed)),)
        paths[(1 << orientation.segment_index, orientation.endpoint_mask, id(orientation))] = _Path(
            (0, 0.0, 0.0, signature),
            (orientation,),
            1,
        )

    for mask in range(1, 1 << segment_count):
        for state_key, state in list(paths.items()):
            used_mask, endpoint_mask, last_orientation_id = state_key
            if used_mask != mask:
                continue

            last = next(
                orientation
                for orientation in all_orientations
                if id(orientation) == last_orientation_id
            )

            for following in all_orientations:
                next_bit = 1 << following.segment_index
                if mask & next_bit or endpoint_mask & following.endpoint_mask:
                    continue
                if len(state.nodes) > 1 and last.singleton:
                    continue
                if following.singleton:
                    continue

                edge = (id(last), id(following))
                if edge not in candidate_distance:
                    continue

                distance = candidate_distance[edge]
                gap = candidate_gap[edge]
                gap_value = gap if gap is not None else 0.0
                next_nodes = state.nodes + (following,)
                next_cost = (
                    0,
                    state.cost[1] + distance,
                    state.cost[2] + gap_value,
                    tuple(
                        (node.segment_index, int(node.reversed))
                        for node in next_nodes
                    ),
                )
                next_key = (
                    mask | next_bit,
                    endpoint_mask | following.endpoint_mask,
                    id(following),
                )
                next_path = _Path(next_cost, next_nodes, state.count)
                current_path = paths.get(next_key)
                if current_path is None or next_cost < current_path.cost:
                    paths[next_key] = next_path

    valid_masks = [
        (1 << segment_index)
        for segment_index, options in enumerate(orientations_by_segment)
        if options
    ]

    best_paths_by_key: dict[tuple[int, int], _Path] = {}
    paths_by_segment_mask: dict[int, list[tuple[int, _Path]]] = {}
    for mask, path in paths.items():
        used_mask, _endpoint_mask, _last_orientation = mask
        path_key = (used_mask, _endpoint_mask)
        current = best_paths_by_key.get(path_key)
        path_objective = (path.cost[1], path.cost[2])
        if current is None:
            best_paths_by_key[path_key] = path
            continue

        current_objective = (current.cost[1], current.cost[2])
        if path_objective < current_objective or (
                path_objective == current_objective and path.cost[3] < current.cost[3]):
            best_paths_by_key[path_key] = path._replace(count=current.count + path.count)
        elif path_objective == current_objective:
            best_paths_by_key[path_key] = current._replace(
                count=current.count + path.count)

    for (used_mask, endpoint_mask), path in best_paths_by_key.items():
        paths_by_segment_mask.setdefault(used_mask, []).append((endpoint_mask, path))

    full_mask = 0
    for bit in valid_masks:
        full_mask |= bit

    covers: dict[tuple[int, int], _Cover] = {
        (0, 0): _Cover((0, 0.0, 0.0, ()), (), 1, 0, 0)
    }

    for segment_mask in range(full_mask + 1):
        if segment_mask & ~full_mask:
            continue

        for endpoint_mask in range(1 << endpoint_count):
            cover = covers.get((segment_mask, endpoint_mask))
            if cover is None:
                continue

            remaining = full_mask ^ segment_mask
            if not remaining:
                continue

            first_bit = remaining & -remaining
            remaining_without_first = remaining ^ first_bit
            path_segment_mask = remaining_without_first
            while True:
                full_path_mask = first_bit | path_segment_mask
                for path_endpoint_mask, path in paths_by_segment_mask.get(full_path_mask, []):
                    if endpoint_mask & path_endpoint_mask:
                        continue

                    next_mask = segment_mask | full_path_mask
                    path_signatures = tuple(sorted(
                        cover.cost[3] + (_path_signature(path.nodes),)))
                    next_cost = (
                        cover.cost[0] + 1,
                        cover.cost[1] + path.cost[1],
                        cover.cost[2] + path.cost[2],
                        path_signatures,
                    )
                    next_count = cover.count * path.count
                    next_key = (next_mask, endpoint_mask | path_endpoint_mask)
                    next_cover = _Cover(
                        next_cost,
                        cover.paths + (path.nodes,),
                        next_count,
                        next_mask,
                        endpoint_mask | path_endpoint_mask,
                    )
                    current_cover = covers.get(next_key)
                    if current_cover is None or next_cost < current_cover.cost:
                        covers[next_key] = next_cover
                    else:
                        next_objective = (next_cost[0], next_cost[1], next_cost[2])
                        current_objective = (
                            current_cover.cost[0],
                            current_cover.cost[1],
                            current_cover.cost[2],
                        )
                        if next_objective == current_objective:
                            total_count = current_cover.count + next_count
                            if next_cost[3] < current_cover.cost[3]:
                                covers[next_key] = next_cover._replace(count=total_count)
                            else:
                                covers[next_key] = current_cover._replace(count=total_count)

                if path_segment_mask == 0:
                    break
                path_segment_mask = (path_segment_mask - 1) & remaining_without_first

    best_cover = min(
        cover
        for key, cover in covers.items()
        if key[0] == full_mask
    )
    ordered_paths = sorted(best_cover.paths, key=lambda path: _path_signature(path)[0])

    stitched_tracks: list[SegmentStitchTrack] = []
    all_connections: list[SegmentStitchConnection] = []

    for nodes in ordered_paths:
        track = mod_gpx.GPXTrack()
        ordered_segments: list[int] = []
        reversed_flags: list[bool] = []
        connections: list[SegmentStitchConnection] = []
        total_distance = 0.0
        known_gap_seconds = 0.0
        has_known_gap = False

        for node in nodes:
            track.segments.append(mod_gpx.GPXTrackSegment(
                _oriented_points(segments, node)))
            ordered_segments.append(node.segment_index)
            reversed_flags.append(node.reversed)

        for left, right in zip(nodes, nodes[1:]):
            edge = (id(left), id(right))
            distance = candidate_distance[edge]
            gap = candidate_gap[edge]
            connection = _connection(left, right, distance, gap)
            connections.append(connection)
            all_connections.append(connection)
            total_distance += distance
            if gap is not None:
                has_known_gap = True
                known_gap_seconds += gap

        stitched_tracks.append(SegmentStitchTrack(
            track,
            tuple(ordered_segments),
            tuple(reversed_flags),
            tuple(connections),
            total_distance,
            known_gap_seconds if has_known_gap else None,
        ))

    all_connections.sort(
        key=lambda connection: (
            connection.from_segment_index,
            connection.to_segment_index,
            connection.from_reversed,
            connection.to_reversed,
        ))
    excluded.sort(key=lambda item: item.segment_index)

    return SegmentStitchResult(
        tuple(stitched_tracks),
        tuple(all_connections),
        tuple(excluded),
        best_cover.count > 1,
        best_cover.count,
    )
