"""
Segmentation Module - Auto-segmentation, manual split/merge operations.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

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


def auto_segment(
    course: Course,
    sim_result: SimulationResult,
    grade_change_threshold: float = 2.0,
    min_segment_m: float = 500.0,
    max_segment_m: float = 5000.0
) -> List[Segment]:
    """
    Automatically segment a course based on grade changes.
    
    Args:
        course: Course to segment
        sim_result: Simulation result with power/speed/time
        grade_change_threshold: Grade change to trigger new segment (%)
        min_segment_m: Minimum segment length in meters
        max_segment_m: Maximum segment length in meters
        
    Returns:
        List of Segment objects
    """
    n_points = len(course.points)
    
    if n_points < 2:
        return []
    
    # Find segment boundaries
    boundaries = [0]  # Start with first point
    
    current_start = 0
    running_grade_sum = 0.0
    running_count = 0
    
    for i in range(1, n_points):
        point = course.points[i]
        current_distance = point.distance_m - course.points[current_start].distance_m
        
        # Update running average
        running_grade_sum += point.grade_pct
        running_count += 1
        running_avg_grade = running_grade_sum / running_count
        
        # Check for segment boundary
        grade_diff = abs(point.grade_pct - running_avg_grade)
        
        should_split = False
        
        # Split on significant grade change (if minimum length reached)
        if grade_diff > grade_change_threshold and current_distance >= min_segment_m:
            should_split = True
        
        # Split if maximum length reached
        if current_distance >= max_segment_m:
            should_split = True
        
        if should_split:
            boundaries.append(i)
            current_start = i
            running_grade_sum = point.grade_pct
            running_count = 1
    
    # Add final boundary
    if boundaries[-1] != n_points:
        boundaries.append(n_points)
    
    # Merge very short segments at the end
    while len(boundaries) > 2:
        last_length = course.points[boundaries[-1] - 1].distance_m - course.points[boundaries[-2]].distance_m
        if last_length < min_segment_m * 0.5:
            boundaries.pop(-2)  # Remove second-to-last boundary
        else:
            break
    
    # Create segments
    segments = []
    cumulative_time = 0.0
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        
        segment = compute_segment_stats(
            course, sim_result, start_idx, end_idx, i, cumulative_time
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
