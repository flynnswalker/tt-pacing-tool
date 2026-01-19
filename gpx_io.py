"""
GPX I/O Module - Parse GPX files, compute distance/bearing, smooth elevation, resample.
"""

import math
from dataclasses import dataclass
from typing import Optional
import numpy as np
import gpxpy
import gpxpy.gpx
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


@dataclass
class RawTrackPoint:
    """Raw point from GPX file."""
    lat: float
    lon: float
    elevation: float


@dataclass
class CoursePoint:
    """Processed course point at fixed distance resolution."""
    distance_m: float      # cumulative distance from start
    elevation_m: float     # smoothed elevation
    grade_pct: float       # local grade percentage
    lat: float
    lon: float
    bearing_deg: float     # heading direction (0-360, 0=North)


@dataclass
class Course:
    """Complete processed course."""
    points: list[CoursePoint]
    total_distance_m: float
    total_elevation_gain_m: float
    total_elevation_loss_m: float
    start_lat: float
    start_lon: float
    
    @property
    def n_points(self) -> int:
        return len(self.points)
    
    def get_distances(self) -> np.ndarray:
        return np.array([p.distance_m for p in self.points])
    
    def get_elevations(self) -> np.ndarray:
        return np.array([p.elevation_m for p in self.points])
    
    def get_grades(self) -> np.ndarray:
        return np.array([p.grade_pct for p in self.points])
    
    def get_bearings(self) -> np.ndarray:
        return np.array([p.bearing_deg for p in self.points])
    
    def get_lats(self) -> np.ndarray:
        return np.array([p.lat for p in self.points])
    
    def get_lons(self) -> np.ndarray:
        return np.array([p.lon for p in self.points])


# Earth radius in meters
EARTH_RADIUS_M = 6371000


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    
    Args:
        lat1, lon1: First point coordinates in degrees
        lat2, lon2: Second point coordinates in degrees
        
    Returns:
        Distance in meters
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return EARTH_RADIUS_M * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the initial bearing from point 1 to point 2.
    
    Args:
        lat1, lon1: Start point coordinates in degrees
        lat2, lon2: End point coordinates in degrees
        
    Returns:
        Bearing in degrees (0-360, 0=North, 90=East)
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    
    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
    
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def parse_gpx(file_path: str) -> list[RawTrackPoint]:
    """
    Parse a GPX file and extract track points.
    
    Args:
        file_path: Path to GPX file
        
    Returns:
        List of RawTrackPoint objects
    """
    with open(file_path, 'r') as f:
        gpx = gpxpy.parse(f)
    
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                elevation = point.elevation if point.elevation is not None else 0.0
                points.append(RawTrackPoint(
                    lat=point.latitude,
                    lon=point.longitude,
                    elevation=elevation
                ))
    
    return points


def parse_gpx_from_string(gpx_string: str) -> list[RawTrackPoint]:
    """
    Parse GPX content from a string.
    
    Args:
        gpx_string: GPX file content as string
        
    Returns:
        List of RawTrackPoint objects
    """
    gpx = gpxpy.parse(gpx_string)
    
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                elevation = point.elevation if point.elevation is not None else 0.0
                points.append(RawTrackPoint(
                    lat=point.latitude,
                    lon=point.longitude,
                    elevation=elevation
                ))
    
    return points


def compute_cumulative_distances(points: list[RawTrackPoint]) -> np.ndarray:
    """
    Compute cumulative distances along the track.
    
    Args:
        points: List of raw track points
        
    Returns:
        Array of cumulative distances in meters
    """
    distances = [0.0]
    for i in range(1, len(points)):
        dist = haversine_distance(
            points[i-1].lat, points[i-1].lon,
            points[i].lat, points[i].lon
        )
        distances.append(distances[-1] + dist)
    return np.array(distances)


def compute_bearings(points: list[RawTrackPoint]) -> np.ndarray:
    """
    Compute bearing at each point (direction of travel).
    
    Args:
        points: List of raw track points
        
    Returns:
        Array of bearings in degrees
    """
    bearings = []
    for i in range(len(points) - 1):
        bearing = calculate_bearing(
            points[i].lat, points[i].lon,
            points[i+1].lat, points[i+1].lon
        )
        bearings.append(bearing)
    
    # Last point uses same bearing as previous
    if bearings:
        bearings.append(bearings[-1])
    else:
        bearings.append(0.0)
    
    return np.array(bearings)


def smooth_elevation(
    elevations: np.ndarray,
    method: str = 'savgol',
    window: int = 5,
    polyorder: int = 2
) -> np.ndarray:
    """
    Smooth elevation data to reduce GPS noise.
    
    Args:
        elevations: Raw elevation array
        method: 'savgol' for Savitzky-Golay filter or 'rolling' for rolling mean
        window: Window size (must be odd for savgol)
        polyorder: Polynomial order for savgol filter
        
    Returns:
        Smoothed elevation array
    """
    if len(elevations) < window:
        return elevations.copy()
    
    if method == 'savgol':
        # Ensure window is odd
        if window % 2 == 0:
            window += 1
        # Ensure window doesn't exceed data length
        window = min(window, len(elevations))
        if window % 2 == 0:
            window -= 1
        if window < 3:
            return elevations.copy()
        # Ensure polyorder is less than window
        polyorder = min(polyorder, window - 1)
        return savgol_filter(elevations, window, polyorder)
    
    elif method == 'rolling':
        # Simple rolling mean
        kernel = np.ones(window) / window
        # Pad to handle edges
        padded = np.pad(elevations, (window//2, window//2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed[:len(elevations)]
    
    else:
        return elevations.copy()


def compute_grades(elevations: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """
    Compute grade (slope) at each point.
    
    Args:
        elevations: Array of elevations in meters
        distances: Array of cumulative distances in meters
        
    Returns:
        Array of grades as percentages (rise/run * 100)
    """
    grades = np.zeros(len(elevations))
    
    for i in range(1, len(elevations)):
        delta_dist = distances[i] - distances[i-1]
        delta_elev = elevations[i] - elevations[i-1]
        
        if delta_dist > 0:
            grades[i] = (delta_elev / delta_dist) * 100
        else:
            grades[i] = 0.0
    
    # First point uses same grade as second
    if len(grades) > 1:
        grades[0] = grades[1]
    
    return grades


def resample_track(
    points: list[RawTrackPoint],
    step_m: float = 50.0,
    smoothing_method: str = 'savgol',
    smoothing_window: int = 5,
    elevation_source: str = 'gpx'  # Hook for future DEM support
) -> Course:
    """
    Resample track to fixed distance intervals with smoothed elevation.
    
    Args:
        points: List of raw track points
        step_m: Distance step in meters (default 50m)
        smoothing_method: Method for elevation smoothing
        smoothing_window: Window size for smoothing
        elevation_source: 'gpx' or 'dem' (dem not yet implemented)
        
    Returns:
        Course object with resampled points
    """
    if len(points) < 2:
        raise ValueError("Need at least 2 points to create a course")
    
    # Compute cumulative distances and bearings for raw points
    raw_distances = compute_cumulative_distances(points)
    raw_bearings = compute_bearings(points)
    raw_elevations = np.array([p.elevation for p in points])
    raw_lats = np.array([p.lat for p in points])
    raw_lons = np.array([p.lon for p in points])
    
    total_distance = raw_distances[-1]
    
    # Create interpolators
    elev_interp = interp1d(raw_distances, raw_elevations, kind='linear', fill_value='extrapolate')
    lat_interp = interp1d(raw_distances, raw_lats, kind='linear', fill_value='extrapolate')
    lon_interp = interp1d(raw_distances, raw_lons, kind='linear', fill_value='extrapolate')
    bearing_interp = interp1d(raw_distances, raw_bearings, kind='nearest', fill_value='extrapolate')
    
    # Create new distance array at fixed intervals
    n_points = max(2, int(np.ceil(total_distance / step_m)) + 1)
    new_distances = np.linspace(0, total_distance, n_points)
    
    # Interpolate values
    new_elevations = elev_interp(new_distances)
    new_lats = lat_interp(new_distances)
    new_lons = lon_interp(new_distances)
    new_bearings = bearing_interp(new_distances)
    
    # Smooth elevation
    smoothed_elevations = smooth_elevation(
        new_elevations, 
        method=smoothing_method,
        window=smoothing_window
    )
    
    # Compute grades from smoothed elevation
    grades = compute_grades(smoothed_elevations, new_distances)
    
    # Compute elevation gain/loss
    elev_diffs = np.diff(smoothed_elevations)
    total_gain = np.sum(elev_diffs[elev_diffs > 0])
    total_loss = np.abs(np.sum(elev_diffs[elev_diffs < 0]))
    
    # Create CoursePoint objects
    course_points = []
    for i in range(len(new_distances)):
        course_points.append(CoursePoint(
            distance_m=new_distances[i],
            elevation_m=smoothed_elevations[i],
            grade_pct=grades[i],
            lat=new_lats[i],
            lon=new_lons[i],
            bearing_deg=new_bearings[i]
        ))
    
    return Course(
        points=course_points,
        total_distance_m=total_distance,
        total_elevation_gain_m=total_gain,
        total_elevation_loss_m=total_loss,
        start_lat=points[0].lat,
        start_lon=points[0].lon
    )


def load_course(
    file_path: str,
    step_m: float = 50.0,
    smoothing_method: str = 'savgol',
    smoothing_window: int = 5
) -> Course:
    """
    Convenience function to load and process a GPX file.
    
    Args:
        file_path: Path to GPX file
        step_m: Distance step in meters
        smoothing_method: Method for elevation smoothing
        smoothing_window: Window size for smoothing
        
    Returns:
        Processed Course object
    """
    raw_points = parse_gpx(file_path)
    return resample_track(
        raw_points,
        step_m=step_m,
        smoothing_method=smoothing_method,
        smoothing_window=smoothing_window
    )


def load_course_from_string(
    gpx_string: str,
    step_m: float = 50.0,
    smoothing_method: str = 'savgol',
    smoothing_window: int = 5
) -> Course:
    """
    Load and process GPX content from a string.
    
    Args:
        gpx_string: GPX file content as string
        step_m: Distance step in meters
        smoothing_method: Method for elevation smoothing
        smoothing_window: Window size for smoothing
        
    Returns:
        Processed Course object
    """
    raw_points = parse_gpx_from_string(gpx_string)
    return resample_track(
        raw_points,
        step_m=step_m,
        smoothing_method=smoothing_method,
        smoothing_window=smoothing_window
    )
