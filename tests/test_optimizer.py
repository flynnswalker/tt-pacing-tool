"""
Tests for optimizer module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpx_io import load_course_from_string
from weather import WeatherData
from physics import PhysicsConfig, simulate_constant_power
from pdc import fit_pdc, default_anchors_from_ftp
from optimizer import (
    optimize_pacing, quick_optimize, OptimizationConfig,
    create_bounds, estimate_initial_duration
)


# Sample GPX for testing
SAMPLE_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1">
  <trk>
    <name>Test Course</name>
    <trkseg>
      <trkpt lat="45.0000" lon="-122.0000"><ele>100</ele></trkpt>
      <trkpt lat="45.0009" lon="-122.0000"><ele>105</ele></trkpt>
      <trkpt lat="45.0018" lon="-122.0000"><ele>115</ele></trkpt>
      <trkpt lat="45.0027" lon="-122.0000"><ele>130</ele></trkpt>
      <trkpt lat="45.0036" lon="-122.0000"><ele>150</ele></trkpt>
      <trkpt lat="45.0045" lon="-122.0000"><ele>160</ele></trkpt>
      <trkpt lat="45.0054" lon="-122.0000"><ele>165</ele></trkpt>
      <trkpt lat="45.0063" lon="-122.0000"><ele>160</ele></trkpt>
      <trkpt lat="45.0072" lon="-122.0000"><ele>150</ele></trkpt>
      <trkpt lat="45.0081" lon="-122.0000"><ele>140</ele></trkpt>
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
    
    anchors = default_anchors_from_ftp(250)
    pdc = fit_pdc(anchors)
    
    return {
        'course': course,
        'weather': weather,
        'physics_config': physics_config,
        'pdc': pdc
    }


class TestOptimization:
    """Tests for pacing optimization."""
    
    def test_optimizer_produces_result(self, test_setup):
        """Optimizer should produce a valid result."""
        config = OptimizationConfig(
            regularization=0.1,
            max_iterations=50,
            verbose=False
        )
        
        result = optimize_pacing(
            test_setup['course'],
            test_setup['weather'],
            test_setup['physics_config'],
            test_setup['pdc'],
            config
        )
        
        assert result is not None
        assert result.power_plan is not None
        assert len(result.power_plan) == len(test_setup['course'].points)
        assert result.total_time_s > 0
    
    def test_optimized_faster_than_baseline(self, test_setup):
        """Optimized plan should be faster than constant power baseline."""
        config = OptimizationConfig(
            regularization=0.1,
            max_iterations=100,
            verbose=False
        )
        
        result = optimize_pacing(
            test_setup['course'],
            test_setup['weather'],
            test_setup['physics_config'],
            test_setup['pdc'],
            config
        )
        
        # Optimized should be faster (or at least not slower)
        # Allow small tolerance for numerical issues
        assert result.total_time_s <= result.baseline_time_s * 1.01
    
    def test_power_within_bounds(self, test_setup):
        """Optimized power should be within configured bounds."""
        config = OptimizationConfig(
            regularization=0.1,
            max_iterations=50
        )
        
        result = optimize_pacing(
            test_setup['course'],
            test_setup['weather'],
            test_setup['physics_config'],
            test_setup['pdc'],
            config
        )
        
        p_min = test_setup['physics_config'].p_min
        
        # All powers should be >= p_min
        assert all(p >= p_min for p in result.power_plan)
        
        # All powers should be reasonable (< 2x FTP)
        ftp = test_setup['pdc'].ftp_estimate()
        assert all(p < ftp * 2 for p in result.power_plan)
    
    def test_quick_optimize_works(self, test_setup):
        """Quick optimize should work with minimal config."""
        result = quick_optimize(
            test_setup['course'],
            test_setup['weather'],
            test_setup['physics_config'],
            test_setup['pdc']
        )
        
        assert result is not None
        assert result.total_time_s > 0


class TestDurationEstimation:
    """Tests for duration estimation."""
    
    def test_duration_estimate_reasonable(self, test_setup):
        """Duration estimate should be reasonable for course."""
        ftp = test_setup['pdc'].ftp_estimate()
        
        duration = estimate_initial_duration(
            test_setup['course'],
            ftp,
            test_setup['weather'],
            test_setup['physics_config']
        )
        
        # Should be positive
        assert duration > 0
        
        # For ~1km course at ~30km/h, should be ~2 minutes
        # Allow wide tolerance since course has climbing
        assert 30 < duration < 600


class TestBounds:
    """Tests for power bounds creation."""
    
    def test_bounds_have_correct_length(self, test_setup):
        """Bounds should have correct number of segments."""
        n_segments = len(test_setup['course'].points)
        
        bounds = create_bounds(
            n_segments,
            test_setup['pdc'],
            test_setup['physics_config'],
            expected_duration_s=300
        )
        
        assert len(bounds.lb) == n_segments
        assert len(bounds.ub) == n_segments
    
    def test_bounds_are_valid(self, test_setup):
        """Lower bounds should be less than upper bounds."""
        bounds = create_bounds(
            100,
            test_setup['pdc'],
            test_setup['physics_config'],
            expected_duration_s=300
        )
        
        assert all(bounds.lb[i] < bounds.ub[i] for i in range(100))


class TestOptimizationConfig:
    """Tests for optimization configuration."""
    
    def test_default_config(self):
        """Default config should have reasonable values."""
        config = OptimizationConfig()
        
        assert config.regularization > 0
        assert config.max_iterations > 0
        assert 0 < config.initial_power_frac <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
