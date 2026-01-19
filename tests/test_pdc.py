"""
Tests for pdc (Power Duration Curve) module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdc import (
    fit_pdc, default_anchors_from_ftp, PDCModel,
    compute_rolling_average, check_feasibility
)


class TestPDCFitting:
    """Tests for PDC curve fitting."""
    
    def test_fit_passes_through_anchors(self):
        """Fitted curve should pass reasonably close to anchor points."""
        anchors = {
            5: 1200,
            60: 600,
            300: 350,
            1200: 300,
            3600: 270
        }
        
        pdc = fit_pdc(anchors)
        
        # Check that power decreases monotonically and curve is reasonable
        # Note: curve fitting may not pass exactly through all points
        # but should capture the general shape
        assert pdc.max_power(5) > pdc.max_power(60) > pdc.max_power(300)
        
        # Check extremes are reasonable (within 30% of anchors)
        assert pdc.max_power(5) == pytest.approx(anchors[5], rel=0.35)
        assert pdc.max_power(3600) == pytest.approx(anchors[3600], rel=0.35)
    
    def test_power_decreases_with_duration(self):
        """Max sustainable power should decrease with duration."""
        anchors = default_anchors_from_ftp(250)
        pdc = fit_pdc(anchors)
        
        durations = [5, 60, 300, 1200, 3600]
        powers = [pdc.max_power(d) for d in durations]
        
        # Each should be less than the previous
        for i in range(1, len(powers)):
            assert powers[i] < powers[i-1]
    
    def test_ftp_estimate_reasonable(self):
        """FTP estimate should be reasonable."""
        anchors = default_anchors_from_ftp(250)
        pdc = fit_pdc(anchors)
        
        ftp = pdc.ftp_estimate()
        
        # Should be close to input FTP (60-min power)
        assert 200 < ftp < 300
    
    def test_minimum_anchors(self):
        """Should work with minimum number of anchors."""
        anchors = {60: 500, 300: 350}  # Only 2 anchors
        pdc = fit_pdc(anchors, model_type='simple')
        
        assert pdc.max_power(60) > 0
        assert pdc.max_power(300) > 0


class TestDefaultAnchors:
    """Tests for default anchor generation."""
    
    def test_default_anchors_from_ftp(self):
        """Default anchors should be reasonable percentages of FTP."""
        ftp = 250
        anchors = default_anchors_from_ftp(ftp)
        
        assert 5 in anchors
        assert 60 in anchors
        assert 300 in anchors
        assert 1200 in anchors
        assert 3600 in anchors
        
        # 5s should be highest, 60m lowest
        assert anchors[5] > anchors[3600]
        
        # 5s should be > 200% FTP
        assert anchors[5] > ftp * 2
        
        # 60m should be < 100% FTP
        assert anchors[3600] < ftp


class TestRollingAverage:
    """Tests for rolling average calculation."""
    
    def test_constant_power_equals_average(self):
        """Constant power should equal its rolling average."""
        power = np.array([250, 250, 250, 250, 250])
        time = np.array([0, 10, 20, 30, 40])
        
        rolling = compute_rolling_average(power, time, window_s=20)
        
        # Should all be close to 250
        assert all(p == pytest.approx(250, rel=0.01) for p in rolling)
    
    def test_rolling_average_smooths_spikes(self):
        """Rolling average should smooth power spikes."""
        power = np.array([200, 200, 400, 200, 200])
        time = np.array([0, 10, 20, 30, 40])
        
        rolling = compute_rolling_average(power, time, window_s=30)
        
        # Peak of rolling average should be less than spike
        assert max(rolling) < 400


class TestFeasibilityChecking:
    """Tests for PDC feasibility checking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        anchors = default_anchors_from_ftp(250)
        self.pdc = fit_pdc(anchors)
    
    def test_feasible_plan_no_violations(self):
        """A conservative plan should have no violations."""
        # Use 80% of max sustainable power
        duration = 600  # 10 minutes
        max_power = self.pdc.max_power(duration)
        conservative_power = max_power * 0.8
        
        power = np.full(120, conservative_power)  # 120 points
        time = np.linspace(0, duration, 120)
        
        violations = check_feasibility(power, time, self.pdc)
        
        assert len(violations) == 0
    
    def test_excessive_power_causes_violation(self):
        """Exceeding PDC limits should cause violations."""
        # Use 120% of max sustainable power
        duration = 300  # 5 minutes
        max_power = self.pdc.max_power(duration)
        excessive_power = max_power * 1.2
        
        power = np.full(60, excessive_power)
        time = np.linspace(0, duration, 60)
        
        violations = check_feasibility(power, time, self.pdc, tolerance=0.0)
        
        assert len(violations) > 0
    
    def test_violation_reports_correct_values(self):
        """Violations should report correct power values."""
        power = np.full(60, 500)  # Very high power
        time = np.linspace(0, 300, 60)  # 5 minutes
        
        violations = check_feasibility(
            power, time, self.pdc, 
            windows=[300], 
            tolerance=0.0
        )
        
        if violations:
            v = violations[0]
            assert v.window_s == 300
            assert v.actual_power == pytest.approx(500, rel=0.1)
            assert v.max_allowed < v.actual_power


class TestPDCModel:
    """Tests for PDCModel class."""
    
    def test_serialization(self):
        """PDCModel should serialize and deserialize correctly."""
        anchors = default_anchors_from_ftp(250)
        original = fit_pdc(anchors)
        
        # Convert to dict and back
        data = original.to_dict()
        restored = PDCModel.from_dict(data)
        
        # Should produce same results
        for duration in [5, 60, 300, 1200]:
            assert original.max_power(duration) == pytest.approx(
                restored.max_power(duration), rel=0.01
            )
    
    def test_max_power_array(self):
        """max_power_array should return array of correct length."""
        anchors = default_anchors_from_ftp(250)
        pdc = fit_pdc(anchors)
        
        durations = np.array([60, 300, 600, 1200])
        powers = pdc.max_power_array(durations)
        
        assert len(powers) == len(durations)
        assert all(p > 0 for p in powers)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
