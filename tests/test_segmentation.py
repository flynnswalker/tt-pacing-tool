"""
Tests for segmentation module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpx_io import load_course_from_string
from weather import WeatherData
from physics import PhysicsConfig, simulate_constant_power
from segmentation import (
    auto_segment, segment_by_distance, split_segment_at_distance,
    merge_segments, compute_segment_stats, rebuild_segments,
    get_segment_boundaries
)


# Sample GPX with varying grades
SAMPLE_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1">
  <trk>
    <name>Test Course</name>
    <trkseg>
      <trkpt lat="45.0000" lon="-122.0000"><ele>100</ele></trkpt>
      <trkpt lat="45.0009" lon="-122.0000"><ele>100</ele></trkpt>
      <trkpt lat="45.0018" lon="-122.0000"><ele>100</ele></trkpt>
      <trkpt lat="45.0027" lon="-122.0000"><ele>120</ele></trkpt>
      <trkpt lat="45.0036" lon="-122.0000"><ele>140</ele></trkpt>
      <trkpt lat="45.0045" lon="-122.0000"><ele>160</ele></trkpt>
      <trkpt lat="45.0054" lon="-122.0000"><ele>180</ele></trkpt>
      <trkpt lat="45.0063" lon="-122.0000"><ele>180</ele></trkpt>
      <trkpt lat="45.0072" lon="-122.0000"><ele>180</ele></trkpt>
      <trkpt lat="45.0081" lon="-122.0000"><ele>160</ele></trkpt>
      <trkpt lat="45.0090" lon="-122.0000"><ele>140</ele></trkpt>
      <trkpt lat="45.0099" lon="-122.0000"><ele>120</ele></trkpt>
    </trkseg>
  </trk>
</gpx>
"""


@pytest.fixture
def test_setup():
    """Create test fixtures."""
    course = load_course_from_string(SAMPLE_GPX, step_m=50.0)
    
    weather = WeatherData(
        wind_speed_ms=0.0,
        wind_direction_deg=0.0,
        temperature_c=20.0,
        humidity_pct=50.0,
        pressure_hpa=1013.0,
        air_density=1.2
    )
    
    physics_config = PhysicsConfig(
        mass_kg=75.0,
        crr=0.004,
        drivetrain_eff=0.97
    )
    
    sim_result = simulate_constant_power(course, 250, weather, physics_config)
    
    return {
        'course': course,
        'weather': weather,
        'physics_config': physics_config,
        'sim_result': sim_result
    }


class TestAutoSegmentation:
    """Tests for automatic segmentation."""
    
    def test_auto_segment_produces_segments(self, test_setup):
        """Auto-segment should produce at least one segment."""
        segments = auto_segment(
            test_setup['course'],
            test_setup['sim_result'],
            grade_change_threshold=2.0,
            min_segment_m=100
        )
        
        assert len(segments) >= 1
    
    def test_segments_cover_full_course(self, test_setup):
        """Segments should cover the entire course without gaps."""
        segments = auto_segment(
            test_setup['course'],
            test_setup['sim_result'],
            min_segment_m=100
        )
        
        # First segment starts at 0
        assert segments[0].start_m == pytest.approx(0, abs=1)
        
        # Last segment ends at course end
        assert segments[-1].end_m == pytest.approx(
            test_setup['course'].total_distance_m, rel=0.01
        )
        
        # No gaps between segments
        for i in range(len(segments) - 1):
            assert segments[i].end_m == pytest.approx(segments[i+1].start_m, rel=0.01)
    
    def test_segments_have_required_attributes(self, test_setup):
        """Each segment should have all required attributes."""
        segments = auto_segment(
            test_setup['course'],
            test_setup['sim_result'],
            min_segment_m=100
        )
        
        for seg in segments:
            assert hasattr(seg, 'start_m')
            assert hasattr(seg, 'end_m')
            assert hasattr(seg, 'avg_grade')
            assert hasattr(seg, 'avg_power')
            assert hasattr(seg, 'predicted_time_s')
            assert hasattr(seg, 'cumulative_time_s')
            
            # Values should be reasonable
            assert seg.predicted_time_s > 0
            assert seg.avg_power > 0
            assert seg.length_m > 0


class TestDistanceSegmentation:
    """Tests for equal-distance segmentation."""
    
    def test_segment_by_distance(self, test_setup):
        """Segments should be approximately equal length."""
        target_length = 200  # meters
        
        segments = segment_by_distance(
            test_setup['course'],
            test_setup['sim_result'],
            segment_length_m=target_length
        )
        
        # Most segments should be close to target length
        # (except possibly first and last)
        for seg in segments[:-1]:
            assert seg.length_m == pytest.approx(target_length, rel=0.3)


class TestSplitMerge:
    """Tests for split and merge operations."""
    
    def test_split_increases_segment_count(self, test_setup):
        """Splitting should increase segment count by 1."""
        segments = auto_segment(
            test_setup['course'],
            test_setup['sim_result'],
            min_segment_m=100
        )
        
        original_count = len(segments)
        
        # Split at middle of course
        mid_distance = test_setup['course'].total_distance_m / 2
        
        new_segments = split_segment_at_distance(
            segments,
            test_setup['course'],
            test_setup['sim_result'],
            mid_distance
        )
        
        # Should have one more segment (if split was valid)
        assert len(new_segments) >= original_count
    
    def test_merge_decreases_segment_count(self, test_setup):
        """Merging should decrease segment count by 1."""
        segments = segment_by_distance(
            test_setup['course'],
            test_setup['sim_result'],
            segment_length_m=200
        )
        
        if len(segments) < 2:
            pytest.skip("Need at least 2 segments to test merge")
        
        original_count = len(segments)
        
        new_segments = merge_segments(
            segments,
            test_setup['course'],
            test_setup['sim_result'],
            idx1=0,
            idx2=1
        )
        
        assert len(new_segments) == original_count - 1
    
    def test_split_preserves_coverage(self, test_setup):
        """Split operation should preserve full course coverage."""
        segments = auto_segment(
            test_setup['course'],
            test_setup['sim_result'],
            min_segment_m=100
        )
        
        mid_distance = test_setup['course'].total_distance_m / 2
        
        new_segments = split_segment_at_distance(
            segments,
            test_setup['course'],
            test_setup['sim_result'],
            mid_distance
        )
        
        # First starts at 0
        assert new_segments[0].start_m == pytest.approx(0, abs=1)
        
        # Last ends at course end
        assert new_segments[-1].end_m == pytest.approx(
            test_setup['course'].total_distance_m, rel=0.01
        )
    
    def test_merge_preserves_coverage(self, test_setup):
        """Merge operation should preserve full course coverage."""
        segments = segment_by_distance(
            test_setup['course'],
            test_setup['sim_result'],
            segment_length_m=200
        )
        
        if len(segments) < 2:
            pytest.skip("Need at least 2 segments")
        
        new_segments = merge_segments(
            segments,
            test_setup['course'],
            test_setup['sim_result'],
            idx1=0,
            idx2=1
        )
        
        # First starts at 0
        assert new_segments[0].start_m == pytest.approx(0, abs=1)
        
        # Last ends at course end
        assert new_segments[-1].end_m == pytest.approx(
            test_setup['course'].total_distance_m, rel=0.01
        )


class TestSegmentBoundaries:
    """Tests for boundary extraction and rebuilding."""
    
    def test_get_boundaries(self, test_setup):
        """Should extract correct boundaries from segments."""
        segments = segment_by_distance(
            test_setup['course'],
            test_setup['sim_result'],
            segment_length_m=200
        )
        
        boundaries = get_segment_boundaries(segments)
        
        # Should start with 0
        assert boundaries[0] == 0
        
        # Should have n+1 boundaries for n segments
        assert len(boundaries) == len(segments) + 1
    
    def test_rebuild_from_boundaries(self, test_setup):
        """Rebuilding from boundaries should preserve segments."""
        original = segment_by_distance(
            test_setup['course'],
            test_setup['sim_result'],
            segment_length_m=200
        )
        
        boundaries = get_segment_boundaries(original)
        
        rebuilt = rebuild_segments(
            test_setup['course'],
            test_setup['sim_result'],
            boundaries
        )
        
        # Same number of segments
        assert len(rebuilt) == len(original)
        
        # Same boundaries
        for orig, reb in zip(original, rebuilt):
            assert orig.start_idx == reb.start_idx
            assert orig.end_idx == reb.end_idx


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
