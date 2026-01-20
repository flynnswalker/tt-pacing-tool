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
    """Represents a PDC violation based on Normalized Power."""
    window_s: int           # Duration window that was exceeded
    actual_power: float     # Actual NP over this window
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
        
        Uses interpolation between anchor points for accuracy, with 
        model-based extrapolation only outside anchor range.
        
        Args:
            duration_s: Duration in seconds
            
        Returns:
            Maximum power in Watts
        """
        if duration_s <= 0:
            return self.p_max
        
        # If we have anchor points, interpolate between them for accuracy
        if self.anchors and len(self.anchors) >= 2:
            anchor_durations = sorted(self.anchors.keys())
            anchor_powers = [self.anchors[d] for d in anchor_durations]
            
            # If within anchor range, interpolate
            if anchor_durations[0] <= duration_s <= anchor_durations[-1]:
                # Find surrounding anchors
                for i in range(len(anchor_durations) - 1):
                    if anchor_durations[i] <= duration_s <= anchor_durations[i + 1]:
                        d1, d2 = anchor_durations[i], anchor_durations[i + 1]
                        p1, p2 = anchor_powers[i], anchor_powers[i + 1]
                        
                        # Log-linear interpolation (power vs log(duration))
                        log_d1, log_d2, log_d = math.log(d1), math.log(d2), math.log(duration_s)
                        t = (log_d - log_d1) / (log_d2 - log_d1)
                        return p1 + t * (p2 - p1)
            
            # Below shortest anchor - use model
            if duration_s < anchor_durations[0]:
                # Use model but cap at shortest anchor extrapolated
                model_power = self._model_power(duration_s)
                return min(model_power, self.p_max)
            
            # Above longest anchor - extrapolate conservatively
            if duration_s > anchor_durations[-1]:
                # Use the last anchor as base and decay slowly
                last_power = anchor_powers[-1]
                last_dur = anchor_durations[-1]
                # Assume ~5% drop per doubling of time
                ratio = duration_s / last_dur
                decay = 0.95 ** (math.log2(ratio))
                return last_power * decay
        
        # Fallback to model
        return self._model_power(duration_s)
    
    def _model_power(self, duration_s: float) -> float:
        """Calculate power using the extended CP model formula."""
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
    
    The model prioritizes matching longer duration anchors since those are 
    most critical for pacing validation.
    
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
    
    # For accurate pacing, we MUST respect the longer duration anchors
    # Use a weighted fit that prioritizes longer durations
    # Weight by log of duration so 3600s is weighted 3x more than 60s
    weights = np.log10(durations + 1)
    weights = weights / weights.sum() * len(weights)  # Normalize
    
    # Initial estimates
    # CP: use the longest duration power as a more accurate starting point
    cp_init = powers[-1]  # Longest duration power
    
    # W': estimate from short duration excess over CP
    w_prime_init = (powers[0] - cp_init) * durations[0]
    
    # Pmax: highest power (usually shortest duration)
    p_max_init = max(powers) * 1.1
    
    # Tau: time constant (typically 10-30s)
    tau_init = 15.0
    
    if model_type == 'simple' or len(anchors) < 4:
        # Fit simple 2-parameter model with weights
        try:
            popt, _ = curve_fit(
                simple_cp_model,
                durations,
                powers,
                p0=[cp_init, w_prime_init],
                bounds=([50, 1000], [500, 50000]),
                sigma=1/weights,  # Higher weight = lower sigma
                maxfev=5000
            )
            cp, w_prime = popt
            
            # Estimate p_max and tau from data
            if 5 in anchors:
                p_max = anchors[5] * 1.05
            else:
                p_max = powers[0] * 1.1
            tau = 20.0
            
        except Exception:
            # Fallback to manual calculation
            cp = powers[-1]  # Use longest duration
            w_prime = (powers[0] - cp) * durations[0]
            p_max = max(powers) * 1.1
            tau = 20.0
    else:
        # Fit extended 4-parameter model with weights
        try:
            popt, _ = curve_fit(
                extended_cp_model,
                durations,
                powers,
                p0=[cp_init, w_prime_init, p_max_init, tau_init],
                bounds=([50, 1000, 200, 5], [500, 50000, 2500, 60]),
                sigma=1/weights,  # Higher weight = lower sigma
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
                    sigma=1/weights,
                    maxfev=5000
                )
                cp, w_prime = popt
                p_max = max(powers) * 1.1
                tau = 20.0
            except Exception:
                # Final fallback - use anchor values directly
                cp = powers[-1]  # Use longest duration
                w_prime = (powers[0] - cp) * durations[0]
                p_max = max(powers) * 1.1
                tau = 20.0
    
    # Validate: max_power at any anchor duration should not exceed anchor by >10%
    # If it does, adjust CP downward
    model = PDCModel(cp=cp, w_prime=w_prime, p_max=p_max, tau=tau, anchors=dict(anchors))
    
    # Check longest anchor specifically
    longest_dur = max(anchors.keys())
    longest_power = anchors[longest_dur]
    model_power = model.max_power(longest_dur)
    
    if model_power > longest_power * 1.05:
        # Model overestimates long duration - reduce CP
        adjustment = (longest_power * 1.02) / model_power
        cp = cp * adjustment
        model = PDCModel(cp=cp, w_prime=w_prime, p_max=p_max, tau=tau, anchors=dict(anchors))
    
    return model


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


def compute_normalized_power(
    power_array: np.ndarray,
    time_array: np.ndarray,
    window_s: float = 30.0
) -> float:
    """
    Compute Normalized Power (NP) for a power array.
    
    NP = 4th root of mean of (30-second rolling average)^4
    
    Args:
        power_array: Array of power values
        time_array: Array of cumulative times
        window_s: Rolling average window (default 30s per NP standard)
        
    Returns:
        Normalized Power in Watts
    """
    if len(power_array) < 2:
        return np.mean(power_array) if len(power_array) > 0 else 0
    
    # Compute 30-second rolling average
    rolling_avg = compute_rolling_average(power_array, time_array, window_s)
    
    # NP = 4th root of mean of 4th powers
    return (np.mean(rolling_avg ** 4)) ** 0.25


def compute_rolling_np(
    power_array: np.ndarray,
    time_array: np.ndarray,
    window_s: float
) -> float:
    """
    Compute the maximum NP over any window of the given duration.
    
    This is what we check against PDC limits - the highest NP achieved
    over any window of that duration.
    
    Args:
        power_array: Array of power values
        time_array: Array of cumulative times
        window_s: Duration window to check
        
    Returns:
        Maximum NP over any window of that duration
    """
    if len(power_array) < 2 or time_array[-1] < window_s:
        return compute_normalized_power(power_array, time_array)
    
    max_np = 0.0
    n = len(power_array)
    
    # Slide window and compute NP for each position
    # Use coarser step for efficiency (check every ~5% of window)
    step = max(1, n // 50)
    
    for i in range(0, n, step):
        current_time = time_array[i]
        end_time = current_time + window_s
        
        if end_time > time_array[-1]:
            break
        
        # Find points within this window
        mask = (time_array >= current_time) & (time_array <= end_time)
        
        if np.sum(mask) > 1:
            window_powers = power_array[mask]
            window_times = time_array[mask] - current_time  # Normalize to start at 0
            
            # Compute NP for this window
            np_val = compute_normalized_power(window_powers, window_times)
            max_np = max(max_np, np_val)
    
    return max_np


def get_check_windows(race_duration_s: float) -> List[int]:
    """
    Get appropriate PDC check windows based on race duration.
    
    Args:
        race_duration_s: Expected race duration in seconds
        
    Returns:
        List of duration windows to check
    """
    # Base windows for short efforts
    windows = [5, 60, 300]
    
    # Add longer windows based on race duration
    if race_duration_s > 600:    # >10 min
        windows.append(600)
    if race_duration_s > 1200:   # >20 min
        windows.append(1200)
    if race_duration_s > 1800:   # >30 min
        windows.append(1800)
    if race_duration_s > 2700:   # >45 min
        windows.append(2700)
    if race_duration_s > 3600:   # >60 min
        windows.append(3600)
    if race_duration_s > 5400:   # >90 min
        windows.append(5400)
    
    # Always check a window at ~90% of race duration
    long_window = int(race_duration_s * 0.9)
    if long_window > 300 and long_window not in windows:
        windows.append(long_window)
    
    return sorted(windows)


def check_feasibility(
    power_array: np.ndarray,
    time_array: np.ndarray,
    pdc: PDCModel,
    windows: Optional[List[int]] = None,
    tolerance: float = 0.02
) -> List[Violation]:
    """
    Check if a power plan violates PDC limits using Normalized Power.
    
    This checks that NP over any window doesn't exceed PDC max for that duration.
    NP accounts for the physiological cost of variable power output.
    
    Args:
        power_array: Array of power values
        time_array: Array of cumulative times
        pdc: PDC model
        windows: List of duration windows to check (auto-selected if None)
        tolerance: Allowed percentage over limit (default 2%)
        
    Returns:
        List of Violation objects for any exceeded limits
    """
    race_duration = time_array[-1] if len(time_array) > 0 else 3600
    
    if windows is None:
        windows = get_check_windows(race_duration)
    
    violations = []
    
    for window_s in windows:
        # Skip if ride is shorter than window
        if time_array[-1] < window_s:
            continue
        
        # Check NP over this window duration (not just average)
        max_np = compute_rolling_np(power_array, time_array, window_s)
        max_allowed = pdc.max_power(window_s)
        
        threshold = max_allowed * (1 + tolerance)
        
        if max_np > threshold:
            excess_pct = (max_np / max_allowed - 1) * 100
            violations.append(Violation(
                window_s=window_s,
                actual_power=max_np,  # This is now NP, not average
                max_allowed=max_allowed,
                excess_pct=excess_pct
            ))
    
    return violations


def compute_pdc_penalty(
    power_array: np.ndarray,
    time_array: np.ndarray,
    pdc: PDCModel,
    windows: Optional[List[int]] = None,
    penalty_weight: float = 10000.0
) -> float:
    """
    Compute soft penalty for PDC violations.
    
    Uses a steep penalty function that makes violations very expensive.
    
    Args:
        power_array: Array of power values
        time_array: Array of cumulative times
        pdc: PDC model
        windows: Duration windows to check (auto-selected if None)
        penalty_weight: Multiplier for penalty (default 10000)
        
    Returns:
        Total penalty value
    """
    violations = check_feasibility(power_array, time_array, pdc, windows, tolerance=0.0)
    
    total_penalty = 0.0
    for v in violations:
        # Strong penalty: linear component + quadratic for larger violations
        excess_pct = v.excess_pct / 100.0  # Convert to fraction
        
        # Linear penalty for any violation (1% over = penalty_weight penalty)
        linear_penalty = penalty_weight * excess_pct
        
        # Quadratic penalty grows with longer windows (violations at race duration are worse)
        window_factor = max(1.0, v.window_s / 300.0)  # Scale by duration
        quadratic_penalty = penalty_weight * (excess_pct ** 2) * window_factor * 10
        
        total_penalty += linear_penalty + quadratic_penalty
    
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
    
    lines = ["PDC Violations (Normalized Power exceeds limits):"]
    for v in violations:
        duration_str = format_duration(v.window_s)
        lines.append(
            f"  - {duration_str} window: {v.actual_power:.0f}W NP vs "
            f"{v.max_allowed:.0f}W max (+{v.excess_pct:.1f}%)"
        )
    
    return "\n".join(lines)
