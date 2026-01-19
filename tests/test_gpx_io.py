"""
Tests for gpx_io module.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpx_io import (
    haversine_distance, calculate_bearing, parse_gpx_from_string,
    compute_cumulative_distances, smooth_elevation, compute_grades,
    resample_track, load_course_from_string, RawTrackPoint
)


# Sample GPX content for testing
SAMPLE_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1">
  <trk>
    <name>Test Course</name>
    <trkseg>
      <trkpt lat="45.0000" lon="-122.0000"><ele>100</ele></trkpt>
      <trkpt lat="45.0009" lon="-122.0000"><ele>110</ele></trkpt>
      <trkpt lat="45.0018" lon="-122.0000"><ele>120</ele></trkpt>
      <trkpt lat="45.0027" lon="-122.0000"><ele>130</ele></trkpt>
      <trkpt lat="45.0036" lon="-122.0000"><ele>140</ele></trkpt>
      <trkpt lat="45.0045" lon="-122.0000"><ele>150</ele></trkpt>
    </trkseg>
  </trk>
</gpx>
"""


class TestHaversine:
    """Tests for haversine distance calculation."""
    
    def test_same_point_zero_distance(self):
        """Same point should have zero distance."""
        dist = haversine_distance(45.0, -122.0, 45.0, -122.0)
        assert dist == 0.0
    
    def test_known_distance(self):
        """Test against known distance."""
        # ~1 degree latitude is roughly 111 km
        dist = haversine_distance(45.0, -122.0, 46.0, -122.0)
        assert 110000 < dist < 112000  # Should be ~111 km
    
    def test_symmetry(self):
        """Distance should be symmetric."""
        d1 = haversine_distance(45.0, -122.0, 45.5, -121.5)
        d2 = haversine_distance(45.5, -121.5, 45.0, -122.0)
        assert abs(d1 - d2) < 0.01


class TestBearing:
    """Tests for bearing calculation."""
    
    def test_north(self):
        """Going north should be ~0 degrees."""
        bearing = calculate_bearing(45.0, -122.0, 46.0, -122.0)
        assert abs(bearing - 0) < 1 or abs(bearing - 360) < 1
    
    def test_east(self):
        """Going east should be ~90 degrees."""
        bearing = calculate_bearing(45.0, -122.0, 45.0, -121.0)
        assert abs(bearing - 90) < 1
    
    def test_south(self):
        """Going south should be ~180 degrees."""
        bearing = calculate_bearing(46.0, -122.0, 45.0, -122.0)
        assert abs(bearing - 180) < 1


class TestGpxParsing:
    """Tests for GPX parsing."""
    
    def test_parse_sample_gpx(self):
        """Should parse sample GPX correctly."""
        points = parse_gpx_from_string(SAMPLE_GPX)
        assert len(points) == 6
        assert points[0].lat == 45.0
        assert points[0].elevation == 100
        assert points[-1].elevation == 150
    
    def test_compute_distances(self):
        """Cumulative distances should be monotonically increasing."""
        points = parse_gpx_from_string(SAMPLE_GPX)
        distances = compute_cumulative_distances(points)
        
        assert distances[0] == 0
        assert all(distances[i] <= distances[i+1] for i in range(len(distances)-1))


class TestElevationSmoothing:
    """Tests for elevation smoothing."""
    
    def test_smoothing_reduces_noise(self):
        """Smoothing should reduce noise/variability."""
        # Create noisy elevation data
        noisy = np.array([100, 105, 98, 112, 95, 108, 102, 115, 99, 110])
        
        smoothed = smooth_elevation(noisy, method='savgol', window=5)
        
        # Smoothed should have lower variance
        assert np.std(smoothed) < np.std(noisy)
    
    def test_smoothing_preserves_length(self):
        """Smoothed array should have same length."""
        original = np.array([100, 110, 120, 130, 140])
        smoothed = smooth_elevation(original, window=3)
        assert len(smoothed) == len(original)
    
    def test_rolling_method(self):
        """Rolling mean smoothing should work."""
        original = np.array([100, 110, 120, 130, 140])
        smoothed = smooth_elevation(original, method='rolling', window=3)
        assert len(smoothed) == len(original)


class TestGradeComputation:
    """Tests for grade computation."""
    
    def test_flat_grade(self):
        """Flat terrain should have zero grade."""
        elevations = np.array([100, 100, 100, 100])
        distances = np.array([0, 100, 200, 300])
        grades = compute_grades(elevations, distances)
        
        assert all(abs(g) < 0.01 for g in grades)
    
    def test_uphill_grade(self):
        """Uphill should have positive grade."""
        elevations = np.array([100, 110, 120, 130])
        distances = np.array([0, 100, 200, 300])
        grades = compute_grades(elevations, distances)
        
        # 10m rise over 100m run = 10% grade
        assert all(g > 0 for g in grades[1:])
        assert abs(grades[1] - 10.0) < 0.1
    
    def test_downhill_grade(self):
        """Downhill should have negative grade."""
        elevations = np.array([130, 120, 110, 100])
        distances = np.array([0, 100, 200, 300])
        grades = compute_grades(elevations, distances)
        
        assert all(g < 0 for g in grades[1:])


class TestResampling:
    """Tests for track resampling."""
    
    def test_resample_produces_correct_step(self):
        """Resampling should produce approximately correct distance steps."""
        points = parse_gpx_from_string(SAMPLE_GPX)
        course = resample_track(points, step_m=50.0)
        
        # Check distances are roughly 50m apart
        distances = course.get_distances()
        diffs = np.diff(distances)
        
        # Most should be close to 50m (except possibly last segment)
        assert np.mean(diffs[:-1]) == pytest.approx(50, rel=0.1)
    
    def test_resample_preserves_total_distance(self):
        """Total distance should be preserved after resampling."""
        points = parse_gpx_from_string(SAMPLE_GPX)
        
        # Get original total distance
        orig_distances = compute_cumulative_distances(points)
        orig_total = orig_distances[-1]
        
        # Resample
        course = resample_track(points, step_m=50.0)
        
        # Should be approximately equal
        assert course.total_distance_m == pytest.approx(orig_total, rel=0.01)
    
    def test_course_has_required_attributes(self):
        """Course object should have all required attributes."""
        course = load_course_from_string(SAMPLE_GPX, step_m=50.0)
        
        assert hasattr(course, 'points')
        assert hasattr(course, 'total_distance_m')
        assert hasattr(course, 'total_elevation_gain_m')
        assert course.n_points > 0
        
        # Check point attributes
        point = course.points[0]
        assert hasattr(point, 'distance_m')
        assert hasattr(point, 'elevation_m')
        assert hasattr(point, 'grade_pct')
        assert hasattr(point, 'bearing_deg')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
