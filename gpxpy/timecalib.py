# Copyright 2011 Tomo Krajina
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Estimate the linear clock transform (offset + drift) between two GPX
recordings of the same activity and produce a time-corrected copy of the
track to be calibrated.

The corrected time of a target point is::

    corrected_time = time + offset + drift * (time - t_ref)

where ``offset`` is in seconds, ``drift`` is dimensionless (seconds of clock
error per second of elapsed time) and ``t_ref`` is a fixed reference epoch
(by default the middle of the target time range, which keeps the two
parameters mostly independent).

Candidates are searched inside the caller-supplied offset/drift ranges and
scored by the *median* spatial distance between the time-corrected target
points and the reference track, interpolated linearly along its segments.
The median makes the score robust to a small number of GPS outliers.
Positions are never interpolated across segment boundaries or across
temporal gaps inside a segment.

When the data cannot identify a unique transform (stationary tracks, too
short overlap, or several near-equivalent minima) the result is flagged as
uncertain instead of reporting a deceptively precise number.

Only point times are modified in the corrected copy; coordinates,
elevations and extensions are left untouched.
"""

import bisect as mod_bisect
import datetime as mod_datetime

from typing import List, Optional, Tuple

from . import geo as mod_geo
from . import gpx as mod_gpx

# Segment of timestamped coordinates: (epoch_seconds, latitudes, longitudes)
_Segment = Tuple[List[float], List[float], List[float]]


def _to_epoch(time: mod_datetime.datetime) -> float:
    if time.tzinfo is None:
        time = time.replace(tzinfo=mod_datetime.timezone.utc)
    return time.timestamp()


def _extract_segments(gpx: mod_gpx.GPX) -> List[_Segment]:
    segments: List[_Segment] = []
    for track in gpx.tracks:
        for segment in track.segments:
            times: List[float] = []
            latitudes: List[float] = []
            longitudes: List[float] = []
            for point in segment.points:
                if point.time is None:
                    continue
                times.append(_to_epoch(point.time))
                latitudes.append(point.latitude)
                longitudes.append(point.longitude)
            if len(times) >= 2:
                segments.append((times, latitudes, longitudes))
    return segments


def _route_length(segments: List[_Segment]) -> float:
    length = 0.0
    for times, latitudes, longitudes in segments:
        for i in range(1, len(times)):
            length += mod_geo.haversine_distance(
                latitudes[i - 1], longitudes[i - 1],
                latitudes[i], longitudes[i])
    return length


def _gap_thresholds(segments: List[_Segment], max_gap_seconds: Optional[float]) -> List[float]:
    thresholds = []
    for times, _, _ in segments:
        if max_gap_seconds is not None:
            thresholds.append(max_gap_seconds)
            continue
        deltas = sorted(times[i] - times[i - 1] for i in range(1, len(times)))
        median_delta = deltas[len(deltas) // 2] if deltas else 1.0
        thresholds.append(max(10.0 * median_delta, 1.0))
    return thresholds


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


class CalibrationResult:
    """
    Outcome of :func:`calibrate_time`.

    ``offset``/``drift`` are the estimated transform parameters (seconds and
    seconds-per-second) or ``None`` when no estimate could be made at all.
    ``corrected`` is a copy of the target GPX with only point times shifted,
    or ``None`` when no estimate exists. ``uncertain`` is True whenever the
    data does not identify a unique reliable transform; the reasons are
    listed in ``uncertainty_reasons``.
    """

    def __init__(self, t_ref: float) -> None:
        self.offset: Optional[float] = None
        self.drift: Optional[float] = None
        self.t_ref: float = t_ref
        self.corrected: Optional[mod_gpx.GPX] = None
        self.uncertain: bool = False
        self.uncertainty_reasons: List[str] = []
        self.median_error: Optional[float] = None
        self.matched_points: int = 0
        self.overlap_seconds: float = 0.0

    def __repr__(self) -> str:
        return (f'CalibrationResult(offset={self.offset!r}, drift={self.drift!r}, '
                f'uncertain={self.uncertain!r}, reasons={self.uncertainty_reasons!r}, '
                f'median_error={self.median_error!r}, matched_points={self.matched_points!r})')


class _CostEvaluator:
    def __init__(self, reference: mod_gpx.GPX, target: mod_gpx.GPX,
                 t_ref: float, max_gap_seconds: Optional[float]) -> None:
        self.ref_segments = _extract_segments(reference)
        self.ref_gap_thresholds = _gap_thresholds(self.ref_segments, max_gap_seconds)
        self.target_segments = _extract_segments(target)
        self.t_ref = t_ref
        self.last_matched = 0
        self.last_overlap_seconds = 0.0

    def cost(self, offset: float, drift: float) -> float:
        distances: List[float] = []
        overlap_start: Optional[float] = None
        overlap_end: Optional[float] = None
        for times, latitudes, longitudes in self.target_segments:
            for i, time in enumerate(times):
                corrected = time + offset + drift * (time - self.t_ref)
                position = self._reference_position_at(corrected)
                if position is None:
                    continue
                latitude, longitude = position
                distances.append(mod_geo.haversine_distance(
                    latitudes[i], longitudes[i], latitude, longitude))
                if overlap_start is None or corrected < overlap_start:
                    overlap_start = corrected
                if overlap_end is None or corrected > overlap_end:
                    overlap_end = corrected
        self.last_matched = len(distances)
        self.last_overlap_seconds = (overlap_end - overlap_start) if overlap_start is not None and overlap_end is not None else 0.0
        if not distances:
            return float('inf')
        return _median(distances)

    def _reference_position_at(self, epoch: float) -> Optional[Tuple[float, float]]:
        for (times, latitudes, longitudes), gap_threshold in zip(self.ref_segments, self.ref_gap_thresholds):
            if epoch < times[0] or epoch > times[-1]:
                continue
            index = mod_bisect.bisect_left(times, epoch)
            if index < len(times) and times[index] == epoch:
                return latitudes[index], longitudes[index]
            if index == 0 or index >= len(times):
                continue
            # Never interpolate across an interruption of the recording:
            if times[index] - times[index - 1] > gap_threshold:
                continue
            ratio = (epoch - times[index - 1]) / (times[index] - times[index - 1])
            return (latitudes[index - 1] + ratio * (latitudes[index] - latitudes[index - 1]),
                    longitudes[index - 1] + ratio * (longitudes[index] - longitudes[index - 1]))
        return None


def apply_correction(gpx: mod_gpx.GPX, offset: float, drift: float, t_ref: float) -> mod_gpx.GPX:
    """
    Return a copy of ``gpx`` where every point time is replaced by
    ``time + offset + drift * (time - t_ref)``. Nothing else is modified:
    coordinates, elevations and extensions are preserved as-is.
    """
    corrected = gpx.clone()
    for point in corrected.walk(only_points=True):
        if point.time is None:
            continue
        shift = offset + drift * (_to_epoch(point.time) - t_ref)
        point.time = point.time + mod_datetime.timedelta(seconds=shift)
    return corrected


def calibrate_time(reference: mod_gpx.GPX, target: mod_gpx.GPX,
                   max_offset: float = 120.0, max_drift: float = 1e-3,
                   t_ref: Optional[float] = None,
                   offset_step: float = 2.0,
                   drift_steps: int = 31,
                   min_overlap_seconds: float = 120.0,
                   min_overlap_points: int = 20,
                   min_route_length: float = 50.0,
                   max_gap_seconds: Optional[float] = None,
                   ambiguity_tolerance: float = 0.5) -> CalibrationResult:
    """
    Estimate the linear clock transform mapping ``target`` times onto
    ``reference`` times and return a :class:`CalibrationResult`.

    ``max_offset`` (seconds) and ``max_drift`` (seconds per second) define
    the symmetric search ranges ``[-max_offset, max_offset]`` and
    ``[-max_drift, max_drift]``. Candidates are scored by the median spatial
    distance between the corrected target points and the reference track,
    which tolerates a small number of localisation outliers. Interpolation
    of the reference track never crosses segment boundaries or gaps longer
    than ``max_gap_seconds`` (auto-detected from the sampling interval when
    ``None``).

    The result is flagged ``uncertain`` when the tracks are effectively
    stationary, the temporal overlap is too short, or several distinct
    candidates are near-equivalent.
    """
    target_segments = _extract_segments(target)
    all_times = [time for times, _, _ in target_segments for time in times]
    if t_ref is None:
        t_ref = (min(all_times) + max(all_times)) / 2.0 if all_times else 0.0
    result = CalibrationResult(t_ref)
    evaluator = _CostEvaluator(reference, target, t_ref, max_gap_seconds)

    if _route_length(evaluator.ref_segments) < min_route_length:
        result.uncertainty_reasons.append('reference track is effectively stationary')
    if _route_length(evaluator.target_segments) < min_route_length:
        result.uncertainty_reasons.append('target track is effectively stationary')

    # Coarse grid search over the caller-supplied ranges:
    drift_step = 2.0 * max_drift / (drift_steps - 1) if drift_steps > 1 else 0.0
    offsets: List[float] = []
    current = -max_offset
    while current <= max_offset + 1e-9:
        offsets.append(current)
        current += offset_step
    drifts = [-max_drift + i * drift_step for i in range(drift_steps)] if drift_steps > 1 else [0.0]

    grid: List[List[float]] = []
    best_cost = float('inf')
    best_indices = (0, 0)
    for i, offset in enumerate(offsets):
        row = []
        for j, drift in enumerate(drifts):
            cost = evaluator.cost(offset, drift)
            row.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_indices = (i, j)
        grid.append(row)

    if best_cost == float('inf'):
        result.uncertainty_reasons.append('no temporal overlap between the tracks')
        result.uncertain = True
        return result

    # Ambiguity: is there a distinct local minimum almost as good as the
    # best one? Neighbouring cells along a valley floor are the same
    # solution, only a separate local minimum counts as a competitor.
    local_minima: List[float] = []
    for i in range(len(offsets)):
        for j in range(len(drifts)):
            cost = grid[i][j]
            is_local_minimum = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < len(offsets) and 0 <= nj < len(drifts) and grid[ni][nj] < cost:
                        is_local_minimum = False
            if is_local_minimum:
                local_minima.append(cost)
    local_minima.sort()
    ambiguous = len(local_minima) > 1 and \
        local_minima[1] <= local_minima[0] + max(0.05 * local_minima[0], ambiguity_tolerance)

    # Local refinement (coordinate descent with shrinking steps):
    best_offset = offsets[best_indices[0]]
    best_drift = drifts[best_indices[1]]
    step_offset = offset_step
    step_drift = drift_step if drift_step > 0 else 1e-6
    for _ in range(200):
        improved = False
        for delta_offset, delta_drift in ((step_offset, 0.0), (-step_offset, 0.0),
                                          (0.0, step_drift), (0.0, -step_drift),
                                          (step_offset, step_drift), (step_offset, -step_drift),
                                          (-step_offset, step_drift), (-step_offset, -step_drift)):
            candidate_offset = best_offset + delta_offset
            candidate_drift = best_drift + delta_drift
            if abs(candidate_offset) > max_offset or abs(candidate_drift) > max_drift:
                continue
            cost = evaluator.cost(candidate_offset, candidate_drift)
            if cost < best_cost:
                best_cost = cost
                best_offset = candidate_offset
                best_drift = candidate_drift
                improved = True
        if not improved:
            step_offset /= 2.0
            step_drift /= 2.0
            if step_offset < 0.05 and step_drift < 5e-7:
                break

    evaluator.cost(best_offset, best_drift)
    result.offset = best_offset
    result.drift = best_drift
    result.median_error = best_cost
    result.matched_points = evaluator.last_matched
    result.overlap_seconds = evaluator.last_overlap_seconds
    result.corrected = apply_correction(target, best_offset, best_drift, t_ref)

    if evaluator.last_matched < min_overlap_points or evaluator.last_overlap_seconds < min_overlap_seconds:
        result.uncertainty_reasons.append('temporal overlap between the tracks is too short')
    if ambiguous:
        result.uncertainty_reasons.append('multiple near-equivalent offset/drift solutions')
    if result.uncertainty_reasons:
        result.uncertain = True
    return result
