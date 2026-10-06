"""
Time calibration between two GPX recordings of the same activity.

When the same activity is recorded with two devices, one clock may be off
by a constant offset and may additionally drift (run slightly too fast or
too slow). This module estimates a linear time transform

    corrected_time = recorded_time + offset + drift * (recorded_time - time_origin)

mapping the timestamps of the ``target`` recording onto the timeline of the
``reference`` recording, and returns a corrected copy of the target GPX in
which only timestamps are changed (locations and extensions are preserved).
``offset`` is the clock error (in seconds) at ``time_origin`` (the first
timed point of the target track) and ``drift`` is the dimensionless
relative clock rate error.

The transform is found by a grid search (plus local refinement) over
caller-supplied offset/drift ranges. Candidates are scored by the spatial
error between the two tracks on their temporal overlap, using a trimmed
mean so that a few localisation outliers do not dominate. Positions are
interpolated along the reference track, but never across recording
interruptions (segment boundaries or long time gaps).

Calibration is fundamentally impossible or ill-conditioned in some
situations (a track that never moves, a very short temporal overlap, or
several near-equivalent solutions, e.g. repeated identical laps). In those
cases the result is flagged as uncertain instead of reporting a
deceptively precise number.
"""

import bisect as mod_bisect
import copy as mod_copy
import datetime as mod_datetime
import logging as mod_logging
import math as mod_math

from typing import *

from . import gpx as mod_gpx

log = mod_logging.getLogger(__name__)

# Mean earth radius, used for the local planar approximation when scoring:
_EARTH_RADIUS = 6371000.0


class CalibrationResult:
    """
    Outcome of `calibrate()`.

    `offset` is in seconds and `drift` is dimensionless, such that
    ``corrected_time = recorded_time + offset + drift * (recorded_time -
    time_origin)`` where `time_origin` is the first timed point of the
    target track.

    If `uncertain` is True the estimate is unreliable (or no estimate could
    be made at all); `reason` explains why and `corrected` is None. If
    `uncertain` is False, `corrected` holds a copy of the target GPX with
    the estimated transform applied to all timestamps.
    """

    def __init__(self, offset: Optional[float]=None, drift: Optional[float]=None,
                 corrected: Optional[mod_gpx.GPX]=None, uncertain: bool=True,
                 reason: str='', error: Optional[float]=None, pairs: int=0,
                 time_origin: Optional[mod_datetime.datetime]=None) -> None:
        self.offset = offset
        self.drift = drift
        self.corrected = corrected
        self.uncertain = uncertain
        self.reason = reason
        self.time_origin = time_origin
        # Robust (trimmed) mean spatial error of the best fit, in meters:
        self.error = error
        # Number of point pairs used to score the best fit:
        self.pairs = pairs

    def __repr__(self) -> str:
        if self.uncertain:
            return f'CalibrationResult(uncertain, reason={self.reason!r})'
        return (f'CalibrationResult(offset={self.offset:.3f}s, drift={self.drift:.6g}, '
                f'error={self.error:.2f}m, pairs={self.pairs})')


class _Run:
    """ A maximal continuous run of timed points (no interpolation across runs). """

    __slots__ = ('times', 'latitudes', 'longitudes')

    def __init__(self) -> None:
        self.times: List[float] = []
        self.latitudes: List[float] = []
        self.longitudes: List[float] = []

    @property
    def start(self) -> float:
        return self.times[0]

    @property
    def end(self) -> float:
        return self.times[-1]


def _epoch_for(time: mod_datetime.datetime) -> mod_datetime.datetime:
    return mod_datetime.datetime(1970, 1, 1, tzinfo=time.tzinfo)


def _timestamp(time: mod_datetime.datetime) -> float:
    return (time - _epoch_for(time)).total_seconds()


def _extract_runs(gpx: mod_gpx.GPX, gap_threshold: float) -> List[_Run]:
    """
    Timed track points grouped into continuous runs. A new run starts at
    every segment boundary and whenever the gap between two consecutive
    timestamps exceeds `gap_threshold` seconds.
    """
    runs = []
    for track in gpx.tracks:
        for segment in track.segments:
            run: Optional[_Run] = None
            previous_time: Optional[float] = None
            for point in segment.points:
                if point.time is None:
                    continue
                time = _timestamp(point.time)
                if run is None or (previous_time is not None and time - previous_time > gap_threshold):
                    run = _Run()
                    runs.append(run)
                run.times.append(time)
                run.latitudes.append(point.latitude)
                run.longitudes.append(point.longitude)
                previous_time = time
    return [run for run in runs if len(run.times) >= 2]


def _distance(latitude_1: float, longitude_1: float, latitude_2: float, longitude_2: float) -> float:
    """ Fast local planar approximation, accurate for short distances. """
    d_lat = mod_math.radians(latitude_2 - latitude_1)
    d_lon = mod_math.radians(longitude_2 - longitude_1)
    mean_lat = mod_math.radians((latitude_1 + latitude_2) / 2)
    x = d_lon * mod_math.cos(mean_lat)
    return _EARTH_RADIUS * mod_math.sqrt(x * x + d_lat * d_lat)


def _path_length(runs: List[_Run]) -> float:
    total = 0.0
    for run in runs:
        for i in range(1, len(run.times)):
            total += _distance(run.latitudes[i - 1], run.longitudes[i - 1],
                               run.latitudes[i], run.longitudes[i])
    return total


def _interpolate(runs: List[_Run], time: float, hint: int) -> Tuple[Optional[Tuple[float, float]], int]:
    """
    Interpolated (latitude, longitude) on the reference runs at `time`, or
    None if `time` falls outside every run. Never interpolates across run
    boundaries. `hint` speeds up lookups for monotonically increasing times.
    """
    indices = list(range(len(runs)))
    if 0 <= hint < len(runs):
        indices.remove(hint)
        indices.insert(0, hint)
    for index in indices:
        run = runs[index]
        if run.start <= time <= run.end:
            i = mod_bisect.bisect_right(run.times, time)
            if i == 0:
                return (run.latitudes[0], run.longitudes[0]), index
            if i >= len(run.times):
                return (run.latitudes[-1], run.longitudes[-1]), index
            t_0, t_1 = run.times[i - 1], run.times[i]
            ratio = (time - t_0) / (t_1 - t_0) if t_1 > t_0 else 0.0
            latitude = run.latitudes[i - 1] + ratio * (run.latitudes[i] - run.latitudes[i - 1])
            longitude = run.longitudes[i - 1] + ratio * (run.longitudes[i] - run.longitudes[i - 1])
            return (latitude, longitude), index
    return None, hint


def _score(points: List[Tuple[float, float, float]], runs: List[_Run],
           offset: float, drift: float, time_origin: float, outlier_ratio: float,
           min_pairs: int) -> Tuple[float, int]:
    """
    Trimmed mean spatial error (meters) of `points` (recorded time, lat,
    lon) against the reference `runs` after applying the candidate
    transform. The largest `outlier_ratio` fraction of errors is discarded,
    so a few localisation outliers do not affect the score.
    """
    errors = []
    hint = 0
    for time, latitude, longitude in points:
        corrected = time + offset + drift * (time - time_origin)
        position, hint = _interpolate(runs, corrected, hint)
        if position is None:
            continue
        errors.append(_distance(latitude, longitude, position[0], position[1]))
    if len(errors) < min_pairs:
        return mod_math.inf, len(errors)
    errors.sort()
    keep = max(1, int(len(errors) * (1 - outlier_ratio)))
    return sum(errors[:keep]) / keep, len(errors)


def _grid(start: float, end: float, steps: int) -> List[float]:
    if steps <= 0 or end <= start:
        return [(start + end) / 2]
    return [start + (end - start) * i / steps for i in range(steps + 1)]


def _first_time(gpx: mod_gpx.GPX) -> Optional[mod_datetime.datetime]:
    result = None
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.time is not None and (result is None or point.time < result):
                    result = point.time
    return result


def _refine(points: List[Tuple[float, float, float]], runs: List[_Run],
            seed_offset: float, seed_drift: float,
            offset_step: float, drift_step: float,
            offset_range: Tuple[float, float], drift_range: Tuple[float, float],
            time_origin: float, outlier_ratio: float, min_pairs: int,
            evaluated: List[Tuple[float, float, float, int]]) -> Tuple[float, float, float, int]:
    """
    Pattern-search refinement of a seed candidate, clamped to the search
    ranges. Unlike a fixed local grid this can walk along the offset/drift
    trade-off valley towards the optimum. All evaluations are appended to
    `evaluated`; returns the best (score, offset, drift, pairs) found.
    """
    best_score, best_pairs = _score(points, runs, seed_offset, seed_drift, time_origin,
                                    outlier_ratio, min_pairs)
    best_offset, best_drift = seed_offset, seed_drift
    evaluated.append((best_score, best_offset, best_drift, best_pairs))
    for _ in range(60):
        improved = False
        for d_offset, d_drift in ((1, 0), (-1, 0), (0, 1), (0, -1),
                                  (1, 1), (1, -1), (-1, 1), (-1, -1)):
            offset = min(max(best_offset + d_offset * offset_step, offset_range[0]), offset_range[1])
            drift = min(max(best_drift + d_drift * drift_step, drift_range[0]), drift_range[1])
            if offset == best_offset and drift == best_drift:
                continue
            score, pairs = _score(points, runs, offset, drift, time_origin,
                                  outlier_ratio, min_pairs)
            evaluated.append((score, offset, drift, pairs))
            if score < best_score:
                best_score, best_offset, best_drift, best_pairs = score, offset, drift, pairs
                improved = True
        if not improved:
            offset_step /= 5
            drift_step /= 5
            if offset_step < 0.01 and drift_step < 1e-6:
                break
    return best_score, best_offset, best_drift, best_pairs


def _apply_transform(target: mod_gpx.GPX, offset: float, drift: float, time_origin: float) -> mod_gpx.GPX:
    """
    Deep copy of `target` with the linear time transform applied to every
    timestamp. Locations, elevations and extensions are left untouched.
    """
    corrected = mod_copy.deepcopy(target)

    def fix_point(point: Any) -> None:
        if point.time is None:
            return
        epoch = _epoch_for(point.time)
        seconds = _timestamp(point.time) + offset + drift * (_timestamp(point.time) - time_origin)
        point.time = epoch + mod_datetime.timedelta(seconds=seconds)

    for waypoint in corrected.waypoints:
        fix_point(waypoint)
    for route in corrected.routes:
        for route_point in route.points:
            fix_point(route_point)
    for track in corrected.tracks:
        for segment in track.segments:
            for track_point in segment.points:
                fix_point(track_point)
    return corrected


def calibrate(reference: mod_gpx.GPX, target: mod_gpx.GPX,
              offset_range: Tuple[float, float]=(-180.0, 180.0),
              drift_range: Tuple[float, float]=(-0.05, 0.05),
              outlier_ratio: float=0.1,
              gap_threshold: float=30.0,
              min_overlap: float=120.0,
              min_movement: float=50.0,
              min_pairs: int=20,
              score_tolerance: float=2.0,
              score_tolerance_relative: float=0.25,
              ambiguity_tolerance: float=3.0,
              offset_steps: int=50,
              drift_steps: int=30,
              max_eval_points: int=250) -> CalibrationResult:
    """
    Estimate the linear time transform mapping `target` timestamps onto the
    `reference` timeline, and return a `CalibrationResult`.

    offset_range / drift_range: search bounds for the offset (seconds) and
    drift (dimensionless) of the transform
    ``corrected_time = recorded_time + offset + drift * (recorded_time -
    time_origin)`` with ``time_origin`` the first timed target point.
    outlier_ratio: fraction of worst point pairs ignored when scoring.
    gap_threshold: time gaps (seconds) larger than this are recording
    interruptions; no interpolation across them.
    min_overlap: required temporal overlap (seconds) after correction.
    min_movement: required total path length (meters) of each track.
    score_tolerance / score_tolerance_relative: candidates scoring within
    max(score_tolerance, score_tolerance_relative * best_score) meters of
    the best score are considered near-equivalent.
    ambiguity_tolerance: the result is reported as uncertain if a distinct
    near-equivalent solution exists whose implied correction differs by
    more than this (seconds), or if perturbing the best correction by this
    amount stays within the score tolerance.
    """
    def uncertain(reason: str) -> CalibrationResult:
        log.debug('Calibration uncertain: %s', reason)
        return CalibrationResult(uncertain=True, reason=reason)

    reference_runs = _extract_runs(reference, gap_threshold)
    target_runs = _extract_runs(target, gap_threshold)
    if not reference_runs or not target_runs:
        return uncertain('missing timed track points')

    if _path_length(reference_runs) < min_movement:
        return uncertain('reference track is (almost) stationary, time cannot be calibrated')
    if _path_length(target_runs) < min_movement:
        return uncertain('target track is (almost) stationary, time cannot be calibrated')

    # Evaluation points from the target track, subsampled deterministically:
    all_points = [(t, lat, lon)
                  for run in target_runs
                  for t, lat, lon in zip(run.times, run.latitudes, run.longitudes)]
    stride = max(1, len(all_points) // max_eval_points)
    points = all_points[::stride]
    if len(points) < min_pairs:
        return uncertain(f'too few timed points to compare ({len(points)})')

    target_start = min(run.start for run in target_runs)
    target_end = max(run.end for run in target_runs)
    reference_start = min(run.start for run in reference_runs)
    reference_end = max(run.end for run in reference_runs)
    time_origin = target_start

    # Coarse grid search over the caller-supplied ranges:
    evaluated: List[Tuple[float, float, float, int]] = []  # (score, offset, drift, pairs)
    for offset in _grid(offset_range[0], offset_range[1], offset_steps):
        for drift in _grid(drift_range[0], drift_range[1], drift_steps):
            score, pairs = _score(points, reference_runs, offset, drift, time_origin,
                                  outlier_ratio, min_pairs)
            evaluated.append((score, offset, drift, pairs))

    if not evaluated:
        return uncertain('empty search range')

    if not mod_math.isfinite(min(evaluated)[0]):
        return uncertain('no temporal overlap between the two tracks')

    offset_step = (offset_range[1] - offset_range[0]) / max(1, offset_steps)
    drift_step = (drift_range[1] - drift_range[0]) / max(1, drift_steps)

    # Refine every local minimum of the coarse grid, not only the best one:
    # a discrete near-equivalent optimum (e.g. shifted by one lap of a
    # repeated loop) may be narrower than the coarse grid step and would
    # otherwise go unnoticed:
    coarse_scores = {(round(offset, 9), round(drift, 9)): score
                     for score, offset, drift, _pairs in evaluated}
    seeds = []
    for score, offset, drift, _pairs in evaluated:
        neighbours = [coarse_scores.get((round(offset + do * offset_step, 9),
                                         round(drift + dd * drift_step, 9)), mod_math.inf)
                      for do in (-1, 0, 1) for dd in (-1, 0, 1) if do or dd]
        if all(score <= neighbour for neighbour in neighbours):
            seeds.append((score, offset, drift))
    seeds.sort()

    best_score, best_offset, best_drift, best_pairs = min(evaluated)
    for _seed_score, seed_offset, seed_drift in seeds[:5]:
        candidate = _refine(points, reference_runs, seed_offset, seed_drift,
                            offset_step, drift_step, offset_range, drift_range,
                            time_origin, outlier_ratio, min_pairs, evaluated)
        if candidate[0] < best_score:
            best_score, best_offset, best_drift, best_pairs = candidate

    # A best solution on the edge of the search range may be pushed there by
    # the caller's bounds rather than by the data:
    if (abs(best_offset - offset_range[0]) < 1e-9 or
            abs(best_offset - offset_range[1]) < 1e-9 or
            abs(best_drift - drift_range[0]) < 1e-9 or
            abs(best_drift - drift_range[1]) < 1e-9):
        return uncertain('best solution lies on the search range boundary; widen the ranges')

    # Temporal overlap of the corrected target with the reference:
    corrected_start = target_start + best_offset
    corrected_end = target_end + best_offset + best_drift * (target_end - time_origin)
    overlap = min(corrected_end, reference_end) - max(corrected_start, reference_start)
    if overlap < min_overlap:
        return uncertain(f'temporal overlap too short ({overlap:.0f}s < {min_overlap:.0f}s)')

    tolerance = max(score_tolerance, score_tolerance_relative * best_score)
    target_mid = (target_start + target_end) / 2
    half_span = max(target_mid - target_start, 1.0)

    # Identifiability probes: if the best correction can be perturbed by a
    # full `ambiguity_tolerance` -- shifted in offset, or pivoted via drift
    # around the overlap midpoint -- without leaving the near-equivalent
    # score band, then the data (e.g. an almost stationary track) does not
    # determine the transform precisely enough:
    probes = [(best_offset + ambiguity_tolerance, best_drift),
              (best_offset - ambiguity_tolerance, best_drift),
              (best_offset - ambiguity_tolerance,
               best_drift + ambiguity_tolerance / half_span),
              (best_offset + ambiguity_tolerance,
               best_drift - ambiguity_tolerance / half_span)]
    for offset, drift in probes:
        if not (offset_range[0] <= offset <= offset_range[1]):
            continue
        if not (drift_range[0] <= drift <= drift_range[1]):
            continue
        score, pairs = _score(points, reference_runs, offset, drift, time_origin,
                              outlier_ratio, min_pairs)
        evaluated.append((score, offset, drift, pairs))
        if score <= best_score + tolerance:
            return uncertain(f'transform not identifiable within {ambiguity_tolerance:.1f}s '
                             f'(perturbed solutions score {score:.1f}m vs best {best_score:.1f}m)')

    # Discrete alternatives: a near-equivalent candidate whose implied
    # correction at the overlap midpoint differs by more than
    # `ambiguity_tolerance` from the best one is a distinct solution (e.g.
    # shifted by one lap of a repeated loop) -- no single precise answer
    # exists:
    best_shift_mid = best_offset + best_drift * (target_mid - time_origin)
    for score, offset, drift, _pairs in evaluated:
        if score > best_score + tolerance:
            continue
        shift_mid = offset + drift * (target_mid - time_origin)
        if abs(shift_mid - best_shift_mid) > ambiguity_tolerance:
            return uncertain(f'multiple near-equivalent solutions (corrections '
                             f'{best_shift_mid:.1f}s and {shift_mid:.1f}s both fit)')

    corrected = _apply_transform(target, best_offset, best_drift, time_origin)
    return CalibrationResult(offset=best_offset, drift=best_drift, corrected=corrected,
                             uncertain=False, reason='', error=best_score, pairs=best_pairs,
                             time_origin=_first_time(target))
