"""
Clip a GPX track to a polygon (with a hole).

The clipping uses a local planar projection.  You must explicitly choose the
base point; the flat approximation is accurate only in a limited area and is
not a globally exact computation.
"""

import gpxpy.gpx
import gpxpy.clip

# Explicitly choose the projection base point (near/in the area of interest):
projection = gpxpy.clip.LocalProjection(base_latitude=0.5, base_longitude=0.5,
                                        max_extent=400_000)

# A simple exterior ring with one hole (a pond tracks must not cross).
# Coordinates are (latitude, longitude); rings may be open or closed.
park = gpxpy.clip.Polygon(
    exterior=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    holes=[[(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)]])

# A track that enters the park, crosses the pond (a hole), and leaves again:
track = gpxpy.gpx.GPXTrack(name='walk')
track.segments.append(gpxpy.gpx.GPXTrackSegment([
    gpxpy.gpx.GPXTrackPoint(0.5, -1.0),
    gpxpy.gpx.GPXTrackPoint(0.5, 2.0),
]))

# Returns a *new* track; the original is left untouched.  Crossing the hole
# splits the single input segment into two output segments:
clipped = track.clip(park, projection)
for segment in clipped.segments:
    print([(p.latitude, p.longitude) for p in segment.points])
# [(0.5, 0.0), (0.5, 0.4)]
# [(0.5, 0.6), (0.5, 1.0)]
