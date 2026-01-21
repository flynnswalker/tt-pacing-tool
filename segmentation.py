"""
Segmentation Module - Auto-segmentation, manual split/merge operations.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy.signal import savgol_filter

from gpx_io import Course
from physics import SimulationResult


@dataclass
class Segment:
    """A course segment with aggregated data."""
    index: int              # Segment index (0-based)
    start_m: float          # Start distance in meters
    end_m: float            # End distance in meters
    start_idx: int          # Start index in course points
    end_idx: int            # End index in course points (exclusive)
    avg_grade: float        # Average grade in percent
    min_grade: float        # Minimum grade
    max_grade: float        # Maximum grade
    elevation_gain: float   # Elevation gain in meters
    elevation_loss: float   # Elevation loss in meters
    avg_power: float        # Average target power (W)
    power_min: float        # Minimum power in segment
    power_max: float        # Maximum power in segment
    predicted_time_s: float # Time for this segment
    cumulative_time_s: float  # Cumulative time at end of segment
    avg_speed_ms: float     # Average speed in m/s
    
    @property
    def length_m(self) -> float:
        return self.end_m - self.start_m
    
    @property
    def length_km(self) -> float:
        return self.length_m / 1000
    
    @property
    def power_range(self) -> Tuple[float, float]:
        return (self.power_min, self.power_max)
    
    @property
    def avg_speed_kmh(self) -> float:
        return self.avg_speed_ms * 3.6
    
    def format_time(self, seconds: float) -> str:
        """Format time in MM:SS or H:MM:SS."""
        if seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}:{secs:02d}"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}:{mins:02d}:{secs:02d}"
    
    @property
    def segment_time_str(self) -> str:
        return self.format_time(self.predicted_time_s)
    
    @property
    def cumulative_time_str(self) -> str:
        return self.format_time(self.cumulative_time_s)


def compute_segment_stats(
    course: Course,
    sim_result: SimulationResult,
    start_idx: int,
    end_idx: int,
    segment_index: int,
    cumulative_time_offset: float = 0.0
) -> Segment:
    """
    Compute statistics for a course segment.
    
    Args:
        course: Full course
        sim_result: Simulation result with power/speed/time
        start_idx: Start index in course points
        end_idx: End index (exclusive)
        segment_index: Index of this segment
        cumulative_time_offset: Time at start of segment
        
    Returns:
        Segment with computed stats
    """
    # Get arrays for this segment
    points = course.points[start_idx:end_idx]
    sim_points = sim_result.points[start_idx:end_idx]
    
    if len(points) == 0:
        raise ValueError(f"Empty segment: start={start_idx}, end={end_idx}")
    
    # Distance - use start_idx for start_m and end_idx-1 for end_m to ensure continuity
    start_m = course.points[start_idx].distance_m
    # For end_m, if this isn't the last segment, use the start of the next segment
    if end_idx < len(course.points):
        end_m = course.points[end_idx].distance_m
    else:
        end_m = course.points[-1].distance_m
    
    # Grades
    grades = np.array([p.grade_pct for p in points])
    avg_grade = np.mean(grades)
    min_grade = np.min(grades)
    max_grade = np.max(grades)
    
    # Elevation
    elevations = np.array([p.elevation_m for p in points])
    elev_diffs = np.diff(elevations)
    elevation_gain = np.sum(elev_diffs[elev_diffs > 0])
    elevation_loss = np.abs(np.sum(elev_diffs[elev_diffs < 0]))
    
    # Power
    powers = np.array([p.power_w for p in sim_points])
    avg_power = np.mean(powers)
    power_min = np.min(powers)
    power_max = np.max(powers)
    
    # Time
    segment_times = np.array([p.segment_time_s for p in sim_points])
    predicted_time = np.sum(segment_times)
    cumulative_time = cumulative_time_offset + predicted_time
    
    # Speed
    speeds = np.array([p.speed_ms for p in sim_points])
    avg_speed = np.mean(speeds)
    
    return Segment(
        index=segment_index,
        start_m=start_m,
        end_m=end_m,
        start_idx=start_idx,
        end_idx=end_idx,
        avg_grade=avg_grade,
        min_grade=min_grade,
        max_grade=max_grade,
        elevation_gain=elevation_gain,
        elevation_loss=elevation_loss,
        avg_power=avg_power,
        power_min=power_min,
        power_max=power_max,
        predicted_time_s=predicted_time,
        cumulative_time_s=cumulative_time,
        avg_speed_ms=avg_speed
    )


def find_gradient_kinks(
    course: Course,
    smoothing_window: int = 11,
    derivative_threshold: float = 0.5
) -> List[int]:
    """
    Find points where gradient changes direction (inflection points).
    Uses the second derivative of elevation (first derivative of grade).
    
    Args:
        course: Course with elevation data
        smoothing_window: Window size for Savitzky-Golay smoothing (must be odd)
        derivative_threshold: Threshold for significant gradient change
        
    Returns:
        List of point indices where gradient inflection occurs
    """
    if len(course.points) < smoothing_window:
        return []
    
    # Extract grades
    grades = np.array([p.grade_pct for p in course.points])
    
    # Smooth grades to reduce GPS noise
    # savgol_filter requires odd window size
    if smoothing_window % 2 == 0:
        smoothing_window += 1
    smoothing_window = min(smoothing_window, len(grades) - 1)
    if smoothing_window < 3:
        smoothing_window = 3
    
    smoothed = savgol_filter(grades, smoothing_window, polyorder=2)
    
    # First derivative of grade = rate of change of gradient
    # This tells us where the slope is getting steeper or shallower
    grade_derivative = np.gradient(smoothed)
    
    # Find inflection points: where the derivative changes sign (zero crossings)
    # Also catch points where derivative magnitude exceeds threshold (sharp changes)
    kinks = []
    
    for i in range(1, len(grade_derivative) - 1):
        # Zero crossing: sign change in derivative
        if grade_derivative[i-1] * grade_derivative[i+1] < 0:
            kinks.append(i)
        # Large magnitude change: significant gradient shift
        elif abs(grade_derivative[i]) > derivative_threshold:
            # Only add if not too close to last kink
            if not kinks or i - kinks[-1] > 5:
                kinks.append(i)
    
    return sorted(set(kinks))


def segments_from_kinks(
    course: Course,
    kinks: List[int],
    min_segment_m: float = 200.0
) -> List[int]:
    """
    Convert kink indices to segment boundaries, respecting minimum length.
    
    Args:
        course: Course data
        kinks: List of kink point indices
        min_segment_m: Minimum segment length in meters
        
    Returns:
        List of boundary indices (including 0 and n_points)
    """
    n_points = len(course.points)
    boundaries = [0]
    
    for kink_idx in kinks:
        if kink_idx >= n_points:
            continue
            
        kink_dist = course.points[kink_idx].distance_m
        last_dist = course.points[boundaries[-1]].distance_m
        
        # Only add if min distance from last boundary
        if kink_dist - last_dist >= min_segment_m:
            boundaries.append(kink_idx)
    
    # Make sure final boundary is included
    if boundaries[-1] != n_points:
        # Check if last segment would be too short
        last_dist = course.points[boundaries[-1]].distance_m
        end_dist = course.points[-1].distance_m
        if end_dist - last_dist < min_segment_m and len(boundaries) > 1:
            # Remove last boundary so it merges with final segment
            boundaries.pop()
        boundaries.append(n_points)
    
    return boundaries


def merge_similar_segments(
    boundaries: List[int],
    powers: np.ndarray,
    threshold: float = 0.03
) -> List[int]:
    """
    Iteratively merge adjacent segments with power difference <= threshold.
    
    Merges the MOST similar pair first (smallest power difference),
    but only if that difference is <= threshold.
    
    Args:
        boundaries: List of segment boundary indices
        powers: Array of power values for each point
        threshold: Maximum relative power difference (0.03 = 3%) for merge
        
    Returns:
        Updated list of boundaries after merging
    """
    boundaries = boundaries.copy()  # Don't modify original
    
    changed = True
    while changed:
        changed = False
        best_merge = None
        best_diff = float('inf')
        
        # Find the pair with smallest power difference that's also <= threshold
        for i in range(len(boundaries) - 2):
            start1, end1 = boundaries[i], boundaries[i + 1]
            start2, end2 = boundaries[i + 1], boundaries[i + 2]
            
            avg1 = np.mean(powers[start1:end1]) if end1 > start1 else 0
            avg2 = np.mean(powers[start2:end2]) if end2 > start2 else 0
            avg_both = (avg1 + avg2) / 2
            
            if avg_both > 0:
                diff_pct = abs(avg1 - avg2) / avg_both
            else:
                diff_pct = 0
            
            # Only merge if within threshold AND it's the smallest difference
            if diff_pct <= threshold and diff_pct < best_diff:
                best_diff = diff_pct
                best_merge = i + 1
        
        if best_merge is not None:
            boundaries.pop(best_merge)
            changed = True
    
    return boundaries


def merge_short_duration_segments(
    boundaries: List[int],
    sim_result: SimulationResult,
    min_duration_s: float = 30.0
) -> List[int]:
    """
    Merge segments that are shorter than the minimum duration.
    
    Short segments are merged with the adjacent segment that has the most
    similar average power.
    
    Args:
        boundaries: List of segment boundary indices
        sim_result: Simulation result with time data
        min_duration_s: Minimum segment duration in seconds
        
    Returns:
        Updated list of boundaries after merging short segments
    """
    if len(boundaries) <= 2:
        return boundaries
    
    boundaries = boundaries.copy()
    times = np.array([p.time_s for p in sim_result.points])
    powers = np.array([p.power_w for p in sim_result.points])
    
    changed = True
    while changed and len(boundaries) > 2:
        changed = False
        
        # Find shortest segment that's below minimum duration
        shortest_idx = None
        shortest_duration = float('inf')
        
        for i in range(len(boundaries) - 1):
            start_idx, end_idx = boundaries[i], boundaries[i + 1]
            if end_idx <= start_idx:
                continue
            
            # Calculate segment duration
            start_time = times[start_idx]
            end_time = times[min(end_idx, len(times) - 1)]
            duration = end_time - start_time
            
            if duration < min_duration_s and duration < shortest_duration:
                shortest_duration = duration
                shortest_idx = i
        
        if shortest_idx is not None:
            # Merge this segment with the more similar neighbor
            start_idx = boundaries[shortest_idx]
            end_idx = boundaries[shortest_idx + 1]
            avg_power = np.mean(powers[start_idx:end_idx]) if end_idx > start_idx else 0
            
            # Check neighbors
            merge_with_prev = False
            merge_with_next = False
            
            if shortest_idx > 0:
                # Previous segment exists
                prev_start = boundaries[shortest_idx - 1]
                prev_end = boundaries[shortest_idx]
                prev_power = np.mean(powers[prev_start:prev_end]) if prev_end > prev_start else 0
                prev_diff = abs(avg_power - prev_power)
                merge_with_prev = True
            else:
                prev_diff = float('inf')
            
            if shortest_idx + 2 < len(boundaries):
                # Next segment exists
                next_start = boundaries[shortest_idx + 1]
                next_end = boundaries[shortest_idx + 2]
                next_power = np.mean(powers[next_start:next_end]) if next_end > next_start else 0
                next_diff = abs(avg_power - next_power)
                merge_with_next = True
            else:
                next_diff = float('inf')
            
            # Merge with the more similar neighbor
            if merge_with_prev and (not merge_with_next or prev_diff <= next_diff):
                # Remove boundary between this segment and previous
                boundaries.pop(shortest_idx)
            elif merge_with_next:
                # Remove boundary between this segment and next
                boundaries.pop(shortest_idx + 1)
            else:
                # Can't merge (shouldn't happen)
                break
            
            changed = True
    
    return boundaries


def auto_segment(
    course: Course,
    sim_result: SimulationResult,
    grade_change_threshold: float = 2.0,  # Kept for API compatibility
    min_segment_m: float = 200.0,
    max_segment_m: float = 5000.0,  # Kept for API compatibility but not used
    target_segments: int = 6,
    min_segment_duration_s: float = 30.0
) -> List[Segment]:
    """
    Automatically segment a course using gradient inflection points.
    
    Strategy:
    1. Find gradient "kinks" (inflection points) using second derivative of elevation
    2. Create initial segments from kinks, respecting min_segment_m
    3. Iteratively merge adjacent segments with similar target power (<= 4% difference)
    4. Merge any segments shorter than min_segment_duration_s
    
    Args:
        course: Course to segment
        sim_result: Simulation result with power/speed/time
        grade_change_threshold: Not used (kept for API compatibility)
        min_segment_m: Minimum segment length in meters
        max_segment_m: Not used (kept for API compatibility)
        target_segments: Not used directly (merging is based on power similarity)
        min_segment_duration_s: Minimum segment duration in seconds (default 30s)
        
    Returns:
        List of Segment objects
    """
    n_points = len(course.points)
    
    if n_points < 2:
        return []
    
    powers = np.array([p.power_w for p in sim_result.points])
    
    # Step 1: Find gradient inflection points (kinks)
    kinks = find_gradient_kinks(course, smoothing_window=11, derivative_threshold=0.5)
    
    # Step 2: Create initial segment boundaries from kinks
    if kinks:
        boundaries = segments_from_kinks(course, kinks, min_segment_m)
    else:
        # Fallback: if no kinks found, create regular segments
        boundaries = [0]
        initial_len = max(min_segment_m, course.total_distance_m / 10)
        for i in range(1, n_points):
            dist_since_last = course.points[i].distance_m - course.points[boundaries[-1]].distance_m
            if dist_since_last >= initial_len:
                boundaries.append(i)
        boundaries.append(n_points)
    
    # Step 3: Iteratively merge segments with similar power (<= 4% difference)
    boundaries = merge_similar_segments(boundaries, powers, threshold=0.04)
    
    # Step 4: Merge segments shorter than minimum duration
    if min_segment_duration_s > 0:
        boundaries = merge_short_duration_segments(boundaries, sim_result, min_segment_duration_s)
    
    # Step 5: Create final segment objects
    segments = []
    cumulative_time = 0.0
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        
        if end_idx > start_idx:
            segment = compute_segment_stats(
                course, sim_result, start_idx, end_idx, len(segments), cumulative_time
            )
            segments.append(segment)
            cumulative_time = segment.cumulative_time_s
    
    return segments


def segment_by_distance(
    course: Course,
    sim_result: SimulationResult,
    segment_length_m: float = 1000.0
) -> List[Segment]:
    """
    Segment course into equal-distance chunks.
    
    Args:
        course: Course to segment
        sim_result: Simulation result
        segment_length_m: Target segment length in meters
        
    Returns:
        List of Segment objects
    """
    total_distance = course.total_distance_m
    n_segments = max(1, int(np.ceil(total_distance / segment_length_m)))
    
    boundaries = [0]
    
    for i in range(1, n_segments):
        target_distance = i * segment_length_m
        
        # Find closest point to target distance
        for j, point in enumerate(course.points):
            if point.distance_m >= target_distance:
                boundaries.append(j)
                break
    
    boundaries.append(len(course.points))
    
    # Create segments
    segments = []
    cumulative_time = 0.0
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        
        if end_idx > start_idx:
            segment = compute_segment_stats(
                course, sim_result, start_idx, end_idx, i, cumulative_time
            )
            segments.append(segment)
            cumulative_time = segment.cumulative_time_s
    
    return segments


def split_segment_at_distance(
    segments: List[Segment],
    course: Course,
    sim_result: SimulationResult,
    distance_m: float
) -> List[Segment]:
    """
    Split segments at a specified distance.
    
    Args:
        segments: Current segment list
        course: Course data
        sim_result: Simulation result
        distance_m: Distance at which to split
        
    Returns:
        Updated segment list
    """
    # Find which segment contains this distance
    target_segment_idx = None
    
    for i, seg in enumerate(segments):
        if seg.start_m <= distance_m < seg.end_m:
            target_segment_idx = i
            break
    
    if target_segment_idx is None:
        return segments  # Distance not within any segment
    
    target_seg = segments[target_segment_idx]
    
    # Find the course point index closest to split distance
    split_idx = None
    for i in range(target_seg.start_idx, target_seg.end_idx):
        if course.points[i].distance_m >= distance_m:
            split_idx = i
            break
    
    if split_idx is None or split_idx <= target_seg.start_idx or split_idx >= target_seg.end_idx - 1:
        return segments  # Can't split here
    
    # Rebuild segments with the split
    new_segments = []
    cumulative_time = 0.0
    new_idx = 0
    
    for i, seg in enumerate(segments):
        if i < target_segment_idx:
            # Keep as-is but renumber
            new_seg = compute_segment_stats(
                course, sim_result, seg.start_idx, seg.end_idx, new_idx, cumulative_time
            )
            new_segments.append(new_seg)
            cumulative_time = new_seg.cumulative_time_s
            new_idx += 1
            
        elif i == target_segment_idx:
            # Split this segment
            seg1 = compute_segment_stats(
                course, sim_result, seg.start_idx, split_idx, new_idx, cumulative_time
            )
            new_segments.append(seg1)
            cumulative_time = seg1.cumulative_time_s
            new_idx += 1
            
            seg2 = compute_segment_stats(
                course, sim_result, split_idx, seg.end_idx, new_idx, cumulative_time
            )
            new_segments.append(seg2)
            cumulative_time = seg2.cumulative_time_s
            new_idx += 1
            
        else:
            # Keep as-is but renumber
            new_seg = compute_segment_stats(
                course, sim_result, seg.start_idx, seg.end_idx, new_idx, cumulative_time
            )
            new_segments.append(new_seg)
            cumulative_time = new_seg.cumulative_time_s
            new_idx += 1
    
    return new_segments


def merge_segments(
    segments: List[Segment],
    course: Course,
    sim_result: SimulationResult,
    idx1: int,
    idx2: int
) -> List[Segment]:
    """
    Merge two adjacent segments.
    
    Args:
        segments: Current segment list
        course: Course data
        sim_result: Simulation result
        idx1: First segment index
        idx2: Second segment index (must be idx1 + 1)
        
    Returns:
        Updated segment list
    """
    if idx2 != idx1 + 1:
        raise ValueError("Can only merge adjacent segments")
    
    if idx1 < 0 or idx2 >= len(segments):
        raise ValueError("Segment indices out of range")
    
    seg1 = segments[idx1]
    seg2 = segments[idx2]
    
    # Rebuild segments with the merge
    new_segments = []
    cumulative_time = 0.0
    new_idx = 0
    
    for i, seg in enumerate(segments):
        if i < idx1:
            # Keep as-is but renumber
            new_seg = compute_segment_stats(
                course, sim_result, seg.start_idx, seg.end_idx, new_idx, cumulative_time
            )
            new_segments.append(new_seg)
            cumulative_time = new_seg.cumulative_time_s
            new_idx += 1
            
        elif i == idx1:
            # Merge idx1 and idx2
            merged = compute_segment_stats(
                course, sim_result, seg1.start_idx, seg2.end_idx, new_idx, cumulative_time
            )
            new_segments.append(merged)
            cumulative_time = merged.cumulative_time_s
            new_idx += 1
            
        elif i == idx2:
            # Skip (already merged)
            pass
            
        else:
            # Keep as-is but renumber
            new_seg = compute_segment_stats(
                course, sim_result, seg.start_idx, seg.end_idx, new_idx, cumulative_time
            )
            new_segments.append(new_seg)
            cumulative_time = new_seg.cumulative_time_s
            new_idx += 1
    
    return new_segments


def rebuild_segments(
    course: Course,
    sim_result: SimulationResult,
    boundaries: List[int]
) -> List[Segment]:
    """
    Rebuild segment list from boundary indices.
    
    Args:
        course: Course data
        sim_result: Simulation result
        boundaries: List of point indices marking segment boundaries
        
    Returns:
        List of Segment objects
    """
    if len(boundaries) < 2:
        return []
    
    # Ensure boundaries are sorted and include start/end
    boundaries = sorted(set(boundaries))
    
    if boundaries[0] != 0:
        boundaries = [0] + boundaries
    
    if boundaries[-1] != len(course.points):
        boundaries.append(len(course.points))
    
    segments = []
    cumulative_time = 0.0
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        
        if end_idx > start_idx:
            segment = compute_segment_stats(
                course, sim_result, start_idx, end_idx, i, cumulative_time
            )
            segments.append(segment)
            cumulative_time = segment.cumulative_time_s
    
    return segments


def get_segment_boundaries(segments: List[Segment]) -> List[int]:
    """
    Extract boundary indices from segment list.
    
    Args:
        segments: List of segments
        
    Returns:
        List of boundary point indices
    """
    if not segments:
        return [0]
    
    boundaries = [segments[0].start_idx]
    for seg in segments:
        boundaries.append(seg.end_idx)
    
    return boundaries


def format_segments_table(segments: List[Segment]) -> str:
    """
    Format segments as a text table.
    
    Args:
        segments: List of segments
        
    Returns:
        Formatted table string
    """
    if not segments:
        return "No segments"
    
    lines = []
    header = (
        f"{'#':>3} | {'Start':>7} | {'End':>7} | {'Length':>6} | "
        f"{'Grade':>6} | {'Power':>6} | {'Range':>11} | "
        f"{'Time':>7} | {'Cumul':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    
    for seg in segments:
        range_str = f"{seg.power_min:.0f}-{seg.power_max:.0f}"
        line = (
            f"{seg.index+1:>3} | "
            f"{seg.start_m/1000:>6.2f}k | "
            f"{seg.end_m/1000:>6.2f}k | "
            f"{seg.length_m:>5.0f}m | "
            f"{seg.avg_grade:>5.1f}% | "
            f"{seg.avg_power:>5.0f}W | "
            f"{range_str:>11} | "
            f"{seg.segment_time_str:>7} | "
            f"{seg.cumulative_time_str:>7}"
        )
        lines.append(line)
    
    return "\n".join(lines)
