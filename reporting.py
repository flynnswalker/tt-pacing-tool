"""
Reporting Module - Plotly charts, sector tables, HR/feel heuristics, export.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gpx_io import Course
from physics import SimulationResult
from segmentation import Segment


@dataclass
class HRGuidance:
    """Heart rate guidance for a segment."""
    segment_index: int
    segment_name: str
    hr_low: int
    hr_high: int
    intensity_zone: str
    notes: str


@dataclass
class FeelAdjustment:
    """Legs-feel based adjustment guidance."""
    checkpoint_segment: int
    checkpoint_name: str
    if_good: str
    if_ok: str
    if_bad: str


@dataclass
class RaceGuidance:
    """Complete race guidance package."""
    hr_guidance: List[HRGuidance]
    feel_adjustments: List[FeelAdjustment]
    key_segments: List[str]
    pacing_summary: str


def create_power_plot(
    course: Course,
    sim_result: SimulationResult,
    segments: Optional[List[Segment]] = None,
    title: str = "Power vs Distance"
) -> go.Figure:
    """
    Create power vs distance plot with grade shading.
    
    Args:
        course: Course data
        sim_result: Simulation result
        segments: Optional segment boundaries to show
        title: Plot title
        
    Returns:
        Plotly Figure
    """
    distances_km = course.get_distances() / 1000
    grades = course.get_grades()
    powers = sim_result.get_powers()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05
    )
    
    # Main power plot
    fig.add_trace(
        go.Scatter(
            x=distances_km,
            y=powers,
            mode='lines',
            name='Target Power',
            line=dict(color='#2196F3', width=2)
        ),
        row=1, col=1
    )
    
    # Add segment boundaries if provided
    if segments:
        for seg in segments:
            fig.add_vline(
                x=seg.start_m / 1000,
                line_dash="dash",
                line_color="gray",
                opacity=0.5,
                row=1, col=1
            )
    
    # Grade profile in bottom subplot
    # Color by grade: green (downhill) to red (uphill)
    colors = ['#4CAF50' if g < 0 else '#F44336' if g > 5 else '#FFC107' for g in grades]
    
    fig.add_trace(
        go.Bar(
            x=distances_km,
            y=grades,
            name='Grade',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    fig.update_xaxes(title_text="Distance (km)", row=2, col=1)
    fig.update_yaxes(title_text="Power (W)", row=1, col=1)
    fig.update_yaxes(title_text="Grade (%)", row=2, col=1)
    
    return fig


def create_speed_plot(
    course: Course,
    sim_result: SimulationResult,
    title: str = "Speed vs Distance"
) -> go.Figure:
    """
    Create speed vs distance plot.
    
    Args:
        course: Course data
        sim_result: Simulation result
        title: Plot title
        
    Returns:
        Plotly Figure
    """
    distances_km = course.get_distances() / 1000
    speeds_kmh = sim_result.get_speeds() * 3.6
    elevations = course.get_elevations()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05
    )
    
    # Speed plot
    fig.add_trace(
        go.Scatter(
            x=distances_km,
            y=speeds_kmh,
            mode='lines',
            name='Speed',
            line=dict(color='#4CAF50', width=2)
        ),
        row=1, col=1
    )
    
    # Elevation profile
    fig.add_trace(
        go.Scatter(
            x=distances_km,
            y=elevations,
            mode='lines',
            name='Elevation',
            fill='tozeroy',
            line=dict(color='#795548'),
            fillcolor='rgba(121, 85, 72, 0.3)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=title,
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    fig.update_xaxes(title_text="Distance (km)", row=2, col=1)
    fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
    fig.update_yaxes(title_text="Elevation (m)", row=2, col=1)
    
    return fig


def create_elevation_profile(
    course: Course,
    sim_result: SimulationResult,
    segments: Optional[List[Segment]] = None,
    title: str = "Elevation Profile"
) -> go.Figure:
    """
    Create elevation profile with time annotations.
    
    Args:
        course: Course data
        sim_result: Simulation result
        segments: Optional segments for annotations
        title: Plot title
        
    Returns:
        Plotly Figure
    """
    distances_km = course.get_distances() / 1000
    elevations = course.get_elevations()
    times = sim_result.get_times()
    
    fig = go.Figure()
    
    # Elevation profile
    fig.add_trace(
        go.Scatter(
            x=distances_km,
            y=elevations,
            mode='lines',
            name='Elevation',
            fill='tozeroy',
            line=dict(color='#795548', width=2),
            fillcolor='rgba(121, 85, 72, 0.3)'
        )
    )
    
    # Add segment annotations with cumulative time
    if segments:
        for seg in segments:
            # Add marker at segment start
            idx = seg.start_idx
            fig.add_annotation(
                x=course.points[idx].distance_m / 1000,
                y=course.points[idx].elevation_m,
                text=f"Seg {seg.index+1}<br>{seg.cumulative_time_str}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                font=dict(size=10)
            )
    
    fig.update_layout(
        title=title,
        xaxis_title="Distance (km)",
        yaxis_title="Elevation (m)",
        height=400
    )
    
    return fig


def create_segment_overlay_chart(
    course: Course,
    sim_result: SimulationResult,
    segments: List[Segment],
    title: str = "Pacing Plan Overview"
) -> go.Figure:
    """
    Create elevation profile with segment power targets overlaid as rectangles.
    
    Shows:
    - Elevation profile as background (filled area)
    - Grade coloring on the elevation
    - Segment power targets as semi-transparent rectangles
    
    Args:
        course: Course data
        sim_result: Simulation result
        segments: Segment list with power targets
        title: Plot title
        
    Returns:
        Plotly Figure with dual y-axes (elevation left, power right)
    """
    distances_km = course.get_distances() / 1000
    elevations = course.get_elevations()
    grades = course.get_grades()
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Color elevation by grade
    # Create segments of elevation colored by grade
    n_points = len(distances_km)
    
    # Add elevation profile as filled area
    fig.add_trace(
        go.Scatter(
            x=distances_km,
            y=elevations,
            mode='lines',
            name='Elevation',
            line=dict(color='#795548', width=1),
            fill='tozeroy',
            fillcolor='rgba(121, 85, 72, 0.2)',
            hovertemplate='%{x:.2f} km<br>%{y:.0f} m<extra>Elevation</extra>'
        ),
        secondary_y=False
    )
    
    # Add grade coloring as scatter points on the elevation line
    # Color: green (descent) -> yellow (flat) -> red (steep climb)
    grade_colors = []
    for g in grades:
        if g < -2:
            grade_colors.append('#4CAF50')  # Green - descent
        elif g < 2:
            grade_colors.append('#FFC107')  # Yellow - flat
        elif g < 6:
            grade_colors.append('#FF9800')  # Orange - moderate
        else:
            grade_colors.append('#F44336')  # Red - steep
    
    fig.add_trace(
        go.Scatter(
            x=distances_km,
            y=elevations,
            mode='markers',
            name='Grade',
            marker=dict(
                color=grade_colors,
                size=3,
                opacity=0.7
            ),
            hovertemplate='%{x:.2f} km<br>Grade: %{text}%<extra></extra>',
            text=[f'{g:.1f}' for g in grades],
            showlegend=False
        ),
        secondary_y=False
    )
    
    # Add segment power rectangles on secondary y-axis
    if segments:
        # Get power range for scaling
        all_powers = [seg.avg_power for seg in segments]
        min_power = min(all_powers) * 0.9
        max_power = max(all_powers) * 1.1
        
        # Color palette for segments (alternating for clarity)
        colors = [
            'rgba(33, 150, 243, 0.4)',   # Blue
            'rgba(76, 175, 80, 0.4)',    # Green  
            'rgba(255, 152, 0, 0.4)',    # Orange
            'rgba(156, 39, 176, 0.4)',   # Purple
            'rgba(0, 188, 212, 0.4)',    # Cyan
            'rgba(255, 87, 34, 0.4)',    # Deep Orange
            'rgba(103, 58, 183, 0.4)',   # Deep Purple
            'rgba(0, 150, 136, 0.4)',    # Teal
        ]
        
        for i, seg in enumerate(segments):
            color = colors[i % len(colors)]
            border_color = color.replace('0.4)', '0.8)')
            
            start_km = seg.start_m / 1000
            end_km = seg.end_m / 1000
            
            # Add rectangle for this segment
            fig.add_shape(
                type="rect",
                x0=start_km,
                x1=end_km,
                y0=min_power,
                y1=seg.avg_power,
                fillcolor=color,
                line=dict(color=border_color, width=2),
                layer="above",
                yref="y2"
            )
            
            # Add power label at top of rectangle
            fig.add_annotation(
                x=(start_km + end_km) / 2,
                y=seg.avg_power,
                yref="y2",
                text=f"<b>{seg.avg_power:.0f}W</b>",
                showarrow=False,
                font=dict(size=11, color='white'),
                bgcolor=border_color,
                borderpad=3
            )
            
            # Add segment number at bottom
            fig.add_annotation(
                x=(start_km + end_km) / 2,
                y=min_power,
                yref="y2",
                text=f"Seg {seg.index + 1}",
                showarrow=False,
                font=dict(size=9),
                yshift=10
            )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode='x unified'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Distance (km)")
    fig.update_yaxes(title_text="Elevation (m)", secondary_y=False)
    fig.update_yaxes(
        title_text="Target Power (W)", 
        secondary_y=True,
        range=[min_power * 0.95, max_power] if segments else None
    )
    
    return fig


def create_splits_chart(
    segments: List[Segment],
    title: str = "Segment Times"
) -> go.Figure:
    """
    Create bar chart of segment times.
    
    Args:
        segments: List of segments
        title: Plot title
        
    Returns:
        Plotly Figure
    """
    segment_names = [f"Seg {s.index+1}" for s in segments]
    times_min = [s.predicted_time_s / 60 for s in segments]
    avg_powers = [s.avg_power for s in segments]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Time bars
    fig.add_trace(
        go.Bar(
            x=segment_names,
            y=times_min,
            name='Time (min)',
            marker_color='#2196F3'
        ),
        secondary_y=False
    )
    
    # Power line
    fig.add_trace(
        go.Scatter(
            x=segment_names,
            y=avg_powers,
            mode='lines+markers',
            name='Avg Power (W)',
            line=dict(color='#F44336', width=2),
            marker=dict(size=8)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title=title,
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    fig.update_yaxes(title_text="Time (min)", secondary_y=False)
    fig.update_yaxes(title_text="Power (W)", secondary_y=True)
    
    return fig


def create_comparison_plot(
    course: Course,
    optimized_sim: SimulationResult,
    baseline_sim: SimulationResult,
    title: str = "Optimized vs Constant Power"
) -> go.Figure:
    """
    Create comparison plot of optimized vs baseline pacing.
    
    Args:
        course: Course data
        optimized_sim: Optimized simulation result
        baseline_sim: Constant power baseline simulation
        title: Plot title
        
    Returns:
        Plotly Figure
    """
    distances_km = course.get_distances() / 1000
    
    opt_powers = optimized_sim.get_powers()
    opt_times = optimized_sim.get_times()
    
    base_powers = baseline_sim.get_powers()
    base_times = baseline_sim.get_times()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.1,
        subplot_titles=("Power Comparison", "Time Difference")
    )
    
    # Power comparison
    fig.add_trace(
        go.Scatter(
            x=distances_km,
            y=opt_powers,
            mode='lines',
            name='Optimized',
            line=dict(color='#2196F3', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=distances_km,
            y=base_powers,
            mode='lines',
            name='Constant',
            line=dict(color='#9E9E9E', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # Time difference (cumulative)
    time_diff = base_times - opt_times  # Positive = optimized is faster
    
    fig.add_trace(
        go.Scatter(
            x=distances_km,
            y=time_diff,
            mode='lines',
            name='Time Saved',
            fill='tozeroy',
            line=dict(color='#4CAF50', width=2),
            fillcolor='rgba(76, 175, 80, 0.3)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=title,
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    fig.update_xaxes(title_text="Distance (km)", row=2, col=1)
    fig.update_yaxes(title_text="Power (W)", row=1, col=1)
    fig.update_yaxes(title_text="Time Saved (s)", row=2, col=1)
    
    return fig


def segments_to_dataframe(segments: List[Segment]) -> pd.DataFrame:
    """
    Convert segments to pandas DataFrame.
    
    Args:
        segments: List of segments
        
    Returns:
        DataFrame with segment data
    """
    data = []
    for seg in segments:
        data.append({
            'Segment': seg.index + 1,
            'Start (km)': f"{seg.start_m/1000:.2f}",
            'End (km)': f"{seg.end_m/1000:.2f}",
            'Length (m)': int(seg.length_m),
            'Avg Grade (%)': f"{seg.avg_grade:.1f}",
            'Target Power (W)': int(seg.avg_power),
            'Power Range': f"{int(seg.power_min)}-{int(seg.power_max)}",
            'Seg Time': seg.segment_time_str,
            'Cumul Time': seg.cumulative_time_str,
            'Avg Speed (km/h)': f"{seg.avg_speed_kmh:.1f}"
        })
    
    return pd.DataFrame(data)


def generate_hr_guidance(
    segments: List[Segment],
    hr_max: int,
    ftp: float
) -> List[HRGuidance]:
    """
    Generate HR guidance for each segment.
    
    Maps power zones to expected HR bands.
    
    Args:
        segments: List of segments
        hr_max: Maximum heart rate
        ftp: Functional threshold power
        
    Returns:
        List of HRGuidance objects
    """
    guidance = []
    
    for seg in segments:
        intensity = seg.avg_power / ftp
        
        if intensity < 0.75:
            zone = "Z2 (Endurance)"
            hr_low = int(0.69 * hr_max)
            hr_high = int(0.83 * hr_max)
            notes = "Easy, conversational pace"
        elif intensity < 0.90:
            zone = "Z3 (Tempo)"
            hr_low = int(0.83 * hr_max)
            hr_high = int(0.89 * hr_max)
            notes = "Comfortably hard"
        elif intensity < 1.05:
            zone = "Z4 (Threshold)"
            hr_low = int(0.89 * hr_max)
            hr_high = int(0.95 * hr_max)
            notes = "Hard, sustainable effort"
        elif intensity < 1.20:
            zone = "Z5 (VO2max)"
            hr_low = int(0.95 * hr_max)
            hr_high = int(1.00 * hr_max)
            notes = "Very hard, at limit"
        else:
            zone = "Z6 (Anaerobic)"
            hr_low = int(0.98 * hr_max)
            hr_high = hr_max
            notes = "Max effort, unsustainable"
        
        guidance.append(HRGuidance(
            segment_index=seg.index,
            segment_name=f"Segment {seg.index + 1}",
            hr_low=hr_low,
            hr_high=hr_high,
            intensity_zone=zone,
            notes=notes
        ))
    
    return guidance


def generate_feel_adjustments(
    segments: List[Segment],
    ftp: float
) -> List[FeelAdjustment]:
    """
    Generate legs-feel based adjustment guidance.
    
    Creates decision rules for early segments that affect later pacing.
    
    Args:
        segments: List of segments
        ftp: Functional threshold power
        
    Returns:
        List of FeelAdjustment objects
    """
    adjustments = []
    n_segments = len(segments)
    
    # Generate checkpoints at ~25% and ~50% of course
    checkpoints = []
    total_distance = segments[-1].end_m if segments else 0
    
    for i, seg in enumerate(segments):
        progress = seg.end_m / total_distance if total_distance > 0 else 0
        if 0.2 <= progress <= 0.35 and len(checkpoints) == 0:
            checkpoints.append(i)
        elif 0.45 <= progress <= 0.6 and len(checkpoints) == 1:
            checkpoints.append(i)
    
    for cp_idx in checkpoints:
        seg = segments[cp_idx]
        remaining = n_segments - cp_idx - 1
        
        if remaining > 0:
            # Calculate adjustment amounts based on FTP
            good_adj = int(ftp * 0.02)  # +2% FTP
            bad_adj = int(ftp * 0.03)   # -3% FTP
            
            adjustments.append(FeelAdjustment(
                checkpoint_segment=cp_idx + 1,
                checkpoint_name=f"End of Segment {cp_idx + 1}",
                if_good=f"Add +{good_adj}W for remaining {remaining} segments",
                if_ok="Hold current targets",
                if_bad=f"Reduce by {bad_adj}W for remaining {remaining} segments"
            ))
    
    return adjustments


def generate_race_guidance(
    segments: List[Segment],
    hr_max: int,
    ftp: float
) -> RaceGuidance:
    """
    Generate complete race guidance package.
    
    Args:
        segments: List of segments
        hr_max: Maximum heart rate
        ftp: Functional threshold power
        
    Returns:
        RaceGuidance object
    """
    hr_guidance = generate_hr_guidance(segments, hr_max, ftp)
    feel_adjustments = generate_feel_adjustments(segments, ftp)
    
    # Identify key segments (steepest climbs, fastest descents)
    key_segments = []
    
    # Find steepest climb
    max_grade_seg = max(segments, key=lambda s: s.avg_grade)
    if max_grade_seg.avg_grade > 3:
        key_segments.append(
            f"Seg {max_grade_seg.index+1}: Steepest climb ({max_grade_seg.avg_grade:.1f}%) - "
            f"maintain {int(max_grade_seg.avg_power)}W"
        )
    
    # Find longest segment
    longest_seg = max(segments, key=lambda s: s.length_m)
    key_segments.append(
        f"Seg {longest_seg.index+1}: Longest segment ({longest_seg.length_km:.1f}km) - "
        f"stay steady at {int(longest_seg.avg_power)}W"
    )
    
    # Pacing summary
    total_time = segments[-1].cumulative_time_s if segments else 0
    avg_power = np.mean([s.avg_power for s in segments]) if segments else 0
    
    mins = int(total_time // 60)
    secs = int(total_time % 60)
    
    pacing_summary = (
        f"Target finish: {mins}:{secs:02d} | "
        f"Average power: {avg_power:.0f}W ({avg_power/ftp*100:.0f}% FTP)"
    )
    
    return RaceGuidance(
        hr_guidance=hr_guidance,
        feel_adjustments=feel_adjustments,
        key_segments=key_segments,
        pacing_summary=pacing_summary
    )


def format_race_card(
    segments: List[Segment],
    guidance: RaceGuidance
) -> str:
    """
    Format a printable race card.
    
    Args:
        segments: List of segments
        guidance: Race guidance
        
    Returns:
        Formatted string for printing
    """
    lines = []
    lines.append("=" * 60)
    lines.append("RACE PACING CARD")
    lines.append("=" * 60)
    lines.append("")
    lines.append(guidance.pacing_summary)
    lines.append("")
    
    # Segment table
    lines.append("-" * 60)
    lines.append(f"{'Seg':>3} | {'Dist':>6} | {'Grade':>6} | {'Power':>6} | {'HR':>9} | {'Time':>6}")
    lines.append("-" * 60)
    
    for seg, hr in zip(segments, guidance.hr_guidance):
        lines.append(
            f"{seg.index+1:>3} | "
            f"{seg.length_m/1000:>5.1f}k | "
            f"{seg.avg_grade:>5.1f}% | "
            f"{seg.avg_power:>5.0f}W | "
            f"{hr.hr_low:>3}-{hr.hr_high:<3} | "
            f"{seg.segment_time_str:>6}"
        )
    
    lines.append("-" * 60)
    lines.append("")
    
    # Key segments
    if guidance.key_segments:
        lines.append("KEY SEGMENTS:")
        for key in guidance.key_segments:
            lines.append(f"  • {key}")
        lines.append("")
    
    # Feel adjustments
    if guidance.feel_adjustments:
        lines.append("ADJUSTMENT RULES:")
        for adj in guidance.feel_adjustments:
            lines.append(f"  At {adj.checkpoint_name}:")
            lines.append(f"    GOOD legs: {adj.if_good}")
            lines.append(f"    OK legs:   {adj.if_ok}")
            lines.append(f"    BAD legs:  {adj.if_bad}")
        lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def export_plan_csv(
    course: Course,
    sim_result: SimulationResult,
    filename: str
) -> None:
    """
    Export full pacing plan to CSV.
    
    Args:
        course: Course data
        sim_result: Simulation result
        filename: Output filename
    """
    data = []
    
    for i, (point, sim_point) in enumerate(zip(course.points, sim_result.points)):
        data.append({
            'distance_m': point.distance_m,
            'elevation_m': point.elevation_m,
            'grade_pct': point.grade_pct,
            'lat': point.lat,
            'lon': point.lon,
            'bearing_deg': point.bearing_deg,
            'target_power_w': sim_point.power_w,
            'predicted_speed_ms': sim_point.speed_ms,
            'predicted_speed_kmh': sim_point.speed_ms * 3.6,
            'cumulative_time_s': sim_point.time_s,
            'segment_time_s': sim_point.segment_time_s,
            'headwind_ms': sim_point.headwind_ms,
            'air_density': sim_point.air_density,
            'cda_used': sim_point.cda_used
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def export_segments_csv(
    segments: List[Segment],
    filename: str
) -> None:
    """
    Export segment summary to CSV.
    
    Args:
        segments: List of segments
        filename: Output filename
    """
    df = segments_to_dataframe(segments)
    df.to_csv(filename, index=False)
