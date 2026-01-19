"""
Power Duration Curve Module - Anchor ingestion, curve fitting, feasibility checking.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
import csv
import io


@dataclass
class Violation:
    """Represents a PDC violation."""
    window_s: int           # Duration window that was exceeded
    actual_power: float     # Actual rolling average power
    max_allowed: float      # Maximum allowed per PDC
    excess_pct: float       # Percentage over limit
    
    @property
    def excess_w(self) -> float:
        return self.actual_power - self.max_allowed


@dataclass
class PDCModel:
    """Fitted Power Duration Curve model."""
    cp: float              # Critical Power (W)
    w_prime: float         # W' (anaerobic work capacity, J)
    p_max: float           # Maximum instantaneous power (W)
    tau: float             # Time constant for decay (s)
    anchors: Dict[int, float]  # Original anchor points
    
    def max_power(self, duration_s: float) -> float:
        """
        Get maximum sustainable power for a given duration.
        
        Uses extended CP model: P(t) = CP + W'/t + (Pmax - CP) * exp(-t/tau)
        
        Args:
            duration_s: Duration in seconds
            
        Returns:
            Maximum power in Watts
        """
        if duration_s <= 0:
            return self.p_max
        
        # Extended model components
        cp_component = self.cp
        wprime_component = self.w_prime / duration_s
        decay_component = (self.p_max - self.cp - self.w_prime) * math.exp(-duration_s / self.tau)
        
        return cp_component + wprime_component + max(0, decay_component)
    
    def max_power_array(self, durations_s: np.ndarray) -> np.ndarray:
        """Vectorized max power calculation."""
        return np.array([self.max_power(d) for d in durations_s])
    
    def ftp_estimate(self) -> float:
        """Estimate FTP (typically 95% of 20-min power or ~100% of 60-min)."""
        return self.max_power(3600)  # 60-minute power
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'cp': self.cp,
            'w_prime': self.w_prime,
            'p_max': self.p_max,
            'tau': self.tau,
            'anchors': self.anchors
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PDCModel':
        """Create from dictionary."""
        return cls(
            cp=data['cp'],
            w_prime=data['w_prime'],
            p_max=data['p_max'],
            tau=data['tau'],
            anchors=data['anchors']
        )


def extended_cp_model(t: np.ndarray, cp: float, w_prime: float, p_max: float, tau: float) -> np.ndarray:
    """
    Extended Critical Power model function.
    
    P(t) = CP + W'/t + (Pmax - CP - W') * exp(-t/tau)
    
    This model combines:
    - Classic CP model for longer durations (CP + W'/t)
    - Exponential decay for short durations to handle peak power
    """
    # Avoid division by zero
    t = np.maximum(t, 0.1)
    
    base = cp + w_prime / t
    decay = (p_max - cp - w_prime / 5) * np.exp(-t / tau)  # Normalize decay at 5s
    
    return base + np.maximum(0, decay)


def simple_cp_model(t: np.ndarray, cp: float, w_prime: float) -> np.ndarray:
    """
    Simple 2-parameter Critical Power model.
    
    P(t) = CP + W'/t
    """
    t = np.maximum(t, 0.1)
    return cp + w_prime / t


def fit_pdc(anchors: Dict[int, float], model_type: str = 'extended') -> PDCModel:
    """
    Fit a power-duration curve to anchor points.
    
    Args:
        anchors: Dictionary mapping duration_seconds -> power_watts
        model_type: 'extended' (4-param) or 'simple' (2-param)
        
    Returns:
        Fitted PDCModel
    """
    if len(anchors) < 2:
        raise ValueError("Need at least 2 anchor points to fit PDC")
    
    # Convert to arrays
    durations = np.array(list(anchors.keys()), dtype=float)
    powers = np.array(list(anchors.values()), dtype=float)
    
    # Sort by duration
    sort_idx = np.argsort(durations)
    durations = durations[sort_idx]
    powers = powers[sort_idx]
    
    # Initial estimates
    # CP: approximately the longest duration power
    cp_init = min(powers)
    
    # W': estimate from short duration excess over CP
    w_prime_init = (powers[0] - cp_init) * durations[0]
    
    # Pmax: highest power (usually shortest duration)
    p_max_init = max(powers) * 1.2
    
    # Tau: time constant (typically 10-30s)
    tau_init = 15.0
    
    if model_type == 'simple' or len(anchors) < 4:
        # Fit simple 2-parameter model
        try:
            popt, _ = curve_fit(
                simple_cp_model,
                durations,
                powers,
                p0=[cp_init, w_prime_init],
                bounds=([50, 1000], [500, 50000]),
                maxfev=5000
            )
            cp, w_prime = popt
            
            # Estimate p_max and tau from data
            if 5 in anchors:
                p_max = anchors[5] * 1.1
            else:
                p_max = powers[0] * 1.2
            tau = 20.0
            
        except Exception:
            # Fallback to manual calculation
            cp = min(powers)
            w_prime = (max(powers) - cp) * min(durations)
            p_max = max(powers) * 1.2
            tau = 20.0
    else:
        # Fit extended 4-parameter model
        try:
            popt, _ = curve_fit(
                extended_cp_model,
                durations,
                powers,
                p0=[cp_init, w_prime_init, p_max_init, tau_init],
                bounds=([50, 1000, 200, 5], [500, 50000, 2500, 60]),
                maxfev=5000
            )
            cp, w_prime, p_max, tau = popt
            
        except Exception:
            # Fallback to simple model
            try:
                popt, _ = curve_fit(
                    simple_cp_model,
                    durations,
                    powers,
                    p0=[cp_init, w_prime_init],
                    bounds=([50, 1000], [500, 50000]),
                    maxfev=5000
                )
                cp, w_prime = popt
                p_max = max(powers) * 1.2
                tau = 20.0
            except Exception:
                # Final fallback
                cp = min(powers)
                w_prime = (max(powers) - cp) * min(durations)
                p_max = max(powers) * 1.2
                tau = 20.0
    
    return PDCModel(
        cp=cp,
        w_prime=w_prime,
        p_max=p_max,
        tau=tau,
        anchors=dict(anchors)
    )


def default_anchors_from_ftp(ftp: float) -> Dict[int, float]:
    """
    Generate typical anchor points based on FTP.
    
    Uses typical ratios for recreational/amateur cyclists.
    
    Args:
        ftp: Functional Threshold Power in Watts
        
    Returns:
        Dictionary of duration -> power anchors
    """
    return {
        5: ftp * 2.5,      # 5-second max (~250% FTP)
        60: ftp * 1.5,     # 1-minute max (~150% FTP)
        300: ftp * 1.15,   # 5-minute max (~115% FTP)
        1200: ftp * 1.02,  # 20-minute max (~102% FTP)
        3600: ftp * 0.95,  # 60-minute (~95% FTP)
    }


def parse_anchors_csv(csv_content: str) -> Dict[int, float]:
    """
    Parse anchor points from CSV content.
    
    Expected format:
    duration_s,power_w
    5,1200
    60,600
    ...
    
    Args:
        csv_content: CSV file content as string
        
    Returns:
        Dictionary of duration -> power
    """
    anchors = {}
    
    reader = csv.DictReader(io.StringIO(csv_content))
    
    for row in reader:
        duration = int(row.get('duration_s', row.get('duration', 0)))
        power = float(row.get('power_w', row.get('power', 0)))
        
        if duration > 0 and power > 0:
            anchors[duration] = power
    
    return anchors


def compute_rolling_average(
    power_array: np.ndarray,
    time_array: np.ndarray,
    window_s: float
) -> np.ndarray:
    """
    Compute rolling average power over a time window.
    
    Args:
        power_array: Array of power values
        time_array: Array of cumulative times (corresponding to power values)
        window_s: Window size in seconds
        
    Returns:
        Array of rolling average powers
    """
    n = len(power_array)
    rolling_avg = np.zeros(n)
    
    for i in range(n):
        current_time = time_array[i]
        start_time = current_time - window_s
        
        # Find points within window
        mask = (time_array >= start_time) & (time_array <= current_time)
        
        if np.any(mask):
            # Weight by time spent at each power level
            indices = np.where(mask)[0]
            
            if len(indices) > 1:
                # Time-weighted average
                times_in_window = time_array[indices]
                powers_in_window = power_array[indices]
                
                # Compute time deltas
                time_deltas = np.diff(times_in_window)
                
                if len(time_deltas) > 0 and np.sum(time_deltas) > 0:
                    # Use midpoint powers
                    mid_powers = (powers_in_window[:-1] + powers_in_window[1:]) / 2
                    rolling_avg[i] = np.sum(mid_powers * time_deltas) / np.sum(time_deltas)
                else:
                    rolling_avg[i] = np.mean(powers_in_window)
            else:
                rolling_avg[i] = power_array[indices[0]]
        else:
            rolling_avg[i] = power_array[i]
    
    return rolling_avg


def check_feasibility(
    power_array: np.ndarray,
    time_array: np.ndarray,
    pdc: PDCModel,
    windows: Optional[List[int]] = None,
    tolerance: float = 0.02
) -> List[Violation]:
    """
    Check if a power plan violates PDC limits.
    
    Args:
        power_array: Array of power values
        time_array: Array of cumulative times
        pdc: PDC model
        windows: List of duration windows to check (default: [5, 60, 300, 1200])
        tolerance: Allowed percentage over limit (default 2%)
        
    Returns:
        List of Violation objects for any exceeded limits
    """
    if windows is None:
        windows = [5, 60, 300, 1200]
    
    violations = []
    
    for window_s in windows:
        # Skip if ride is shorter than window
        if time_array[-1] < window_s:
            continue
        
        rolling_avg = compute_rolling_average(power_array, time_array, window_s)
        max_rolling = np.max(rolling_avg)
        max_allowed = pdc.max_power(window_s)
        
        threshold = max_allowed * (1 + tolerance)
        
        if max_rolling > threshold:
            excess_pct = (max_rolling / max_allowed - 1) * 100
            violations.append(Violation(
                window_s=window_s,
                actual_power=max_rolling,
                max_allowed=max_allowed,
                excess_pct=excess_pct
            ))
    
    return violations


def compute_pdc_penalty(
    power_array: np.ndarray,
    time_array: np.ndarray,
    pdc: PDCModel,
    windows: Optional[List[int]] = None,
    penalty_weight: float = 1000.0
) -> float:
    """
    Compute soft penalty for PDC violations.
    
    Args:
        power_array: Array of power values
        time_array: Array of cumulative times
        pdc: PDC model
        windows: Duration windows to check
        penalty_weight: Multiplier for penalty
        
    Returns:
        Total penalty value
    """
    violations = check_feasibility(power_array, time_array, pdc, windows, tolerance=0.0)
    
    total_penalty = 0.0
    for v in violations:
        # Quadratic penalty on excess
        excess = max(0, v.actual_power - v.max_allowed)
        total_penalty += penalty_weight * (excess / v.max_allowed) ** 2
    
    return total_penalty


def format_duration(seconds: int) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        if secs == 0:
            return f"{mins}m"
        return f"{mins}m{secs}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        if mins == 0:
            return f"{hours}h"
        return f"{hours}h{mins}m"


def format_violations(violations: List[Violation]) -> str:
    """Format violations for display."""
    if not violations:
        return "No PDC violations detected."
    
    lines = ["PDC Violations Detected:"]
    for v in violations:
        duration_str = format_duration(v.window_s)
        lines.append(
            f"  - {duration_str} window: {v.actual_power:.0f}W actual vs "
            f"{v.max_allowed:.0f}W max (+{v.excess_pct:.1f}%)"
        )
    
    return "\n".join(lines)
