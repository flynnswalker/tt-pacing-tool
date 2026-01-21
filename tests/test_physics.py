"""
Tests for physics module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics import (
    get_cda, compute_gravity_force, compute_rolling_resistance,
    compute_aero_drag, compute_required_power, compute_speed_from_power,
    simulate_segment, PhysicsConfig, AeroConfig
)


class TestCdaSwitching:
    """Tests for CdA switching logic."""
    
    def test_flat_terrain_uses_aero_cda(self):
        """Flat terrain at normal speed should use aero CdA (TT bike)."""
        config = AeroConfig(bike_type='tt', cda_aero=0.22, cda_non_aero=0.28,
                           grade_threshold=6.0, speed_threshold=6.0)
        
        cda = get_cda(grade_pct=2.0, speed_ms=10.0, config=config)
        assert cda == 0.22
    
    def test_steep_grade_uses_non_aero_cda(self):
        """Steep grade should use non-aero CdA (TT bike)."""
        config = AeroConfig(bike_type='tt', cda_aero=0.22, cda_non_aero=0.28,
                           grade_threshold=6.0, speed_threshold=6.0)
        
        cda = get_cda(grade_pct=8.0, speed_ms=10.0, config=config)
        assert cda == 0.28
    
    def test_low_speed_uses_non_aero_cda(self):
        """Low speed should use non-aero CdA (TT bike)."""
        config = AeroConfig(bike_type='tt', cda_aero=0.22, cda_non_aero=0.28,
                           grade_threshold=6.0, speed_threshold=6.0)
        
        cda = get_cda(grade_pct=2.0, speed_ms=4.0, config=config)
        assert cda == 0.28
    
    def test_threshold_boundary(self):
        """Test behavior at exact threshold values."""
        config = AeroConfig(bike_type='tt', cda_aero=0.22, cda_non_aero=0.28,
                           grade_threshold=6.0, speed_threshold=6.0)
        
        # At exact threshold, should still use aero
        cda = get_cda(grade_pct=6.0, speed_ms=6.0, config=config)
        assert cda == 0.22  # <= threshold uses aero
    
    def test_road_bike_single_cda(self):
        """Road bike should use single CdA regardless of conditions."""
        config = AeroConfig(bike_type='road', cda_road=0.32)
        
        # Flat terrain
        cda1 = get_cda(grade_pct=2.0, speed_ms=10.0, config=config)
        assert cda1 == 0.32
        
        # Steep grade
        cda2 = get_cda(grade_pct=12.0, speed_ms=10.0, config=config)
        assert cda2 == 0.32
        
        # Low speed
        cda3 = get_cda(grade_pct=2.0, speed_ms=2.0, config=config)
        assert cda3 == 0.32


class TestForces:
    """Tests for force calculations."""
    
    def test_gravity_uphill_positive(self):
        """Gravity force should be positive (resistance) going uphill."""
        force = compute_gravity_force(mass_kg=75.0, grade_pct=5.0)
        assert force > 0
    
    def test_gravity_downhill_negative(self):
        """Gravity force should be negative (assistance) going downhill."""
        force = compute_gravity_force(mass_kg=75.0, grade_pct=-5.0)
        assert force < 0
    
    def test_gravity_flat_zero(self):
        """Gravity force should be ~zero on flat."""
        force = compute_gravity_force(mass_kg=75.0, grade_pct=0.0)
        assert abs(force) < 0.01
    
    def test_rolling_resistance_always_positive(self):
        """Rolling resistance should always be positive (resistance)."""
        force_flat = compute_rolling_resistance(75.0, 0.0, 0.004)
        force_up = compute_rolling_resistance(75.0, 5.0, 0.004)
        force_down = compute_rolling_resistance(75.0, -5.0, 0.004)
        
        assert force_flat > 0
        assert force_up > 0
        assert force_down > 0
    
    def test_aero_drag_increases_with_speed(self):
        """Aero drag should increase with speed."""
        drag_slow = compute_aero_drag(5.0, 0.0, 1.2, 0.25)
        drag_fast = compute_aero_drag(10.0, 0.0, 1.2, 0.25)
        
        assert drag_fast > drag_slow
    
    def test_headwind_increases_drag(self):
        """Headwind should increase aerodynamic drag."""
        drag_no_wind = compute_aero_drag(10.0, 0.0, 1.2, 0.25)
        drag_headwind = compute_aero_drag(10.0, 5.0, 1.2, 0.25)
        
        assert drag_headwind > drag_no_wind
    
    def test_tailwind_decreases_drag(self):
        """Tailwind should decrease aerodynamic drag."""
        drag_no_wind = compute_aero_drag(10.0, 0.0, 1.2, 0.25)
        drag_tailwind = compute_aero_drag(10.0, -5.0, 1.2, 0.25)
        
        assert drag_tailwind < drag_no_wind


class TestPowerSpeed:
    """Tests for power-speed relationships."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PhysicsConfig(
            mass_kg=75.0,
            crr=0.004,
            drivetrain_eff=0.97
        )
    
    def test_higher_power_higher_speed(self):
        """Higher power should result in higher speed."""
        speed_low = compute_speed_from_power(
            200, 0.0, 0.0, 1.2, self.config
        )
        speed_high = compute_speed_from_power(
            300, 0.0, 0.0, 1.2, self.config
        )
        
        assert speed_high > speed_low
    
    def test_uphill_slower_than_flat(self):
        """Same power should be slower uphill than on flat."""
        speed_flat = compute_speed_from_power(
            250, 0.0, 0.0, 1.2, self.config
        )
        speed_uphill = compute_speed_from_power(
            250, 5.0, 0.0, 1.2, self.config
        )
        
        assert speed_uphill < speed_flat
    
    def test_headwind_slower(self):
        """Headwind should reduce speed at same power."""
        speed_calm = compute_speed_from_power(
            250, 0.0, 0.0, 1.2, self.config
        )
        speed_headwind = compute_speed_from_power(
            250, 0.0, 5.0, 1.2, self.config
        )
        
        assert speed_headwind < speed_calm
    
    def test_power_speed_roundtrip(self):
        """compute_required_power should be inverse of compute_speed_from_power."""
        original_power = 250
        
        speed = compute_speed_from_power(
            original_power, 0.0, 0.0, 1.2, self.config
        )
        
        calculated_power = compute_required_power(
            speed, 0.0, 0.0, 1.2, self.config
        )
        
        # Should be close (not exact due to numerical methods)
        assert calculated_power == pytest.approx(original_power, rel=0.02)


class TestSimulation:
    """Tests for segment simulation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PhysicsConfig(
            mass_kg=75.0,
            crr=0.004,
            drivetrain_eff=0.97,
            p_min=50.0
        )
    
    def test_simulation_returns_positive_time(self):
        """Simulation should return positive time for any segment."""
        speed, time = simulate_segment(
            power_w=250,
            distance_m=100,
            grade_pct=0.0,
            headwind_ms=0.0,
            air_density=1.2,
            config=self.config
        )
        
        assert time > 0
        assert speed > 0
    
    def test_simulation_time_proportional_to_distance(self):
        """Doubling distance should roughly double time."""
        _, time_100m = simulate_segment(
            250, 100, 0.0, 0.0, 1.2, self.config
        )
        _, time_200m = simulate_segment(
            250, 200, 0.0, 0.0, 1.2, self.config
        )
        
        # Should be roughly proportional
        assert time_200m == pytest.approx(time_100m * 2, rel=0.05)
    
    def test_minimum_power_enforced(self):
        """Very low power should use p_min."""
        speed, _ = simulate_segment(
            power_w=10,  # Below p_min
            distance_m=100,
            grade_pct=-10.0,  # Steep descent
            headwind_ms=0.0,
            air_density=1.2,
            config=self.config
        )
        
        # Should still have positive speed (using p_min)
        assert speed > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
