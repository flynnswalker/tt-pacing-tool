"""
TT Pacing Tool - Streamlit Application

Main entry point for the time trial pacing optimization tool.
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import io

# Import local modules
from gpx_io import load_course_from_string, Course
from weather import (
    fetch_weather, create_fallback_weather, WeatherData, 
    EnvironmentConfig, compute_air_density
)
from physics import PhysicsConfig, AeroConfig, simulate_constant_power
from pdc import PDCModel, fit_pdc, default_anchors_from_ftp, check_feasibility, format_violations
from optimizer import optimize_pacing, OptimizationConfig, OptimizationResult
from segmentation import auto_segment, split_segment_at_distance, merge_segments, Segment
from reporting import (
    create_power_plot, create_speed_plot, create_elevation_profile,
    create_splits_chart, create_comparison_plot, segments_to_dataframe,
    generate_race_guidance, format_race_card, export_plan_csv
)


# Page configuration
st.set_page_config(
    page_title="TT Pacing Tool",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize session state variables."""
    if 'course' not in st.session_state:
        st.session_state.course = None
    if 'weather' not in st.session_state:
        st.session_state.weather = None
    if 'pdc' not in st.session_state:
        st.session_state.pdc = None
    if 'optimization_result' not in st.session_state:
        st.session_state.optimization_result = None
    if 'segments' not in st.session_state:
        st.session_state.segments = None


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS or H:MM:SS."""
    if seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}:{mins:02d}:{secs:02d}"


def sidebar_rider_params() -> dict:
    """Render rider parameters in sidebar."""
    st.sidebar.header("Rider Parameters")
    
    mass = st.sidebar.number_input(
        "Total Mass (kg)",
        min_value=50.0,
        max_value=150.0,
        value=75.0,
        step=0.5,
        help="Rider + bike + gear"
    )
    
    ftp = st.sidebar.number_input(
        "FTP (W)",
        min_value=100,
        max_value=500,
        value=250,
        step=5,
        help="Functional Threshold Power"
    )
    
    hr_max = st.sidebar.number_input(
        "Max HR",
        min_value=150,
        max_value=220,
        value=185,
        step=1,
        help="Maximum heart rate"
    )
    
    return {'mass': mass, 'ftp': ftp, 'hr_max': hr_max}


def sidebar_power_anchors(ftp: float) -> dict:
    """Render power anchor inputs in sidebar."""
    st.sidebar.header("Power Anchors")
    
    use_defaults = st.sidebar.checkbox("Use defaults from FTP", value=True)
    
    if use_defaults:
        anchors = default_anchors_from_ftp(ftp)
        st.sidebar.caption(f"5s: {anchors[5]:.0f}W | 1m: {anchors[60]:.0f}W | 5m: {anchors[300]:.0f}W")
        st.sidebar.caption(f"20m: {anchors[1200]:.0f}W | 60m: {anchors[3600]:.0f}W")
    else:
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            p5s = st.number_input("5s", min_value=100, max_value=2000, value=int(ftp*2.5))
            p1m = st.number_input("1m", min_value=100, max_value=1500, value=int(ftp*1.5))
            p5m = st.number_input("5m", min_value=100, max_value=800, value=int(ftp*1.15))
        
        with col2:
            p20m = st.number_input("20m", min_value=100, max_value=600, value=int(ftp*1.02))
            p60m = st.number_input("60m", min_value=100, max_value=500, value=int(ftp*0.95))
        
        anchors = {5: p5s, 60: p1m, 300: p5m, 1200: p20m, 3600: p60m}
    
    # CSV import option
    uploaded_csv = st.sidebar.file_uploader("Or import CSV", type=['csv'], key='anchor_csv')
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            if 'duration_s' in df.columns and 'power_w' in df.columns:
                anchors = dict(zip(df['duration_s'].astype(int), df['power_w']))
                st.sidebar.success(f"Loaded {len(anchors)} anchors from CSV")
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {e}")
    
    return anchors


def sidebar_equipment() -> dict:
    """Render equipment parameters in sidebar."""
    st.sidebar.header("Equipment")
    
    with st.sidebar.expander("Aerodynamics", expanded=False):
        cda_flat = st.number_input(
            "CdA Flat (m²)",
            min_value=0.15,
            max_value=0.40,
            value=0.25,
            step=0.01,
            help="Drag area in TT position"
        )
        
        cda_climb = st.number_input(
            "CdA Climb (m²)",
            min_value=0.20,
            max_value=0.50,
            value=0.32,
            step=0.01,
            help="Drag area in climbing position"
        )
        
        grade_threshold = st.number_input(
            "Grade Threshold (%)",
            min_value=0.0,
            max_value=15.0,
            value=6.0,
            step=0.5,
            help="Switch to climb CdA above this grade"
        )
        
        speed_threshold = st.number_input(
            "Speed Threshold (km/h)",
            min_value=10.0,
            max_value=40.0,
            value=22.0,
            step=1.0,
            help="Switch to climb CdA below this speed"
        )
    
    with st.sidebar.expander("Rolling Resistance", expanded=False):
        crr = st.number_input(
            "Crr",
            min_value=0.002,
            max_value=0.010,
            value=0.004,
            step=0.0005,
            format="%.4f",
            help="Rolling resistance coefficient"
        )
    
    return {
        'cda_flat': cda_flat,
        'cda_climb': cda_climb,
        'grade_threshold': grade_threshold,
        'speed_threshold_ms': speed_threshold / 3.6,
        'crr': crr
    }


def sidebar_environment() -> dict:
    """Render environment settings in sidebar."""
    st.sidebar.header("Environment")
    
    use_forecast = st.sidebar.checkbox("Fetch weather forecast", value=False)
    
    if use_forecast:
        race_date = st.sidebar.date_input("Race Date", value=datetime.now().date())
        race_time = st.sidebar.time_input("Start Time", value=datetime.now().time())
        race_datetime = datetime.combine(race_date, race_time)
        
        return {
            'use_forecast': True,
            'race_datetime': race_datetime
        }
    else:
        with st.sidebar.expander("Manual Weather", expanded=True):
            temp = st.number_input("Temperature (°C)", value=20.0, step=1.0)
            pressure = st.number_input("Pressure (hPa)", value=1013.0, step=1.0)
            humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
            wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, value=0.0, step=1.0)
            wind_dir = st.number_input("Wind Direction (°)", min_value=0, max_value=359, value=0,
                                       help="Direction wind is coming FROM (0=N, 90=E)")
        
        return {
            'use_forecast': False,
            'temp': temp,
            'pressure': pressure,
            'humidity': humidity,
            'wind_speed_ms': wind_speed / 3.6,
            'wind_dir': wind_dir
        }


def sidebar_advanced() -> dict:
    """Render advanced settings in sidebar."""
    with st.sidebar.expander("Advanced Settings", expanded=False):
        smoothing_window = st.number_input(
            "Elevation Smoothing Window",
            min_value=3,
            max_value=15,
            value=5,
            step=2,
            help="Points for Savitzky-Golay filter"
        )
        
        regularization = st.slider(
            "Pacing Smoothness",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help="Higher = smoother power changes"
        )
        
        grade_seg_threshold = st.number_input(
            "Segment Grade Threshold (%)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Grade change to trigger new segment"
        )
        
        min_segment_m = st.number_input(
            "Min Segment Length (m)",
            min_value=200,
            max_value=2000,
            value=500,
            step=100
        )
    
    return {
        'smoothing_window': smoothing_window,
        'regularization': regularization,
        'grade_seg_threshold': grade_seg_threshold,
        'min_segment_m': min_segment_m
    }


def render_course_tab(course: Course):
    """Render the Course tab content."""
    st.subheader("Course Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Distance", f"{course.total_distance_m/1000:.2f} km")
    with col2:
        st.metric("Elevation Gain", f"{course.total_elevation_gain_m:.0f} m")
    with col3:
        st.metric("Elevation Loss", f"{course.total_elevation_loss_m:.0f} m")
    with col4:
        avg_grade = (course.total_elevation_gain_m / course.total_distance_m) * 100
        st.metric("Avg Uphill Grade", f"{avg_grade:.1f}%")
    
    # Elevation profile
    st.subheader("Elevation Profile")
    
    distances_km = course.get_distances() / 1000
    elevations = course.get_elevations()
    grades = course.get_grades()
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.05)
    
    fig.add_trace(
        go.Scatter(x=distances_km, y=elevations, mode='lines', name='Elevation',
                   fill='tozeroy', line=dict(color='#795548'),
                   fillcolor='rgba(121, 85, 72, 0.3)'),
        row=1, col=1
    )
    
    colors = ['#4CAF50' if g < 0 else '#F44336' if g > 5 else '#FFC107' for g in grades]
    fig.add_trace(
        go.Bar(x=distances_km, y=grades, name='Grade', marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    
    fig.update_layout(height=450, showlegend=False)
    fig.update_xaxes(title_text="Distance (km)", row=2, col=1)
    fig.update_yaxes(title_text="Elevation (m)", row=1, col=1)
    fig.update_yaxes(title_text="Grade (%)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)


def render_plan_tab(result: OptimizationResult, course: Course):
    """Render the Plan tab content."""
    st.subheader("Optimized Pacing Plan")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Predicted Time", format_time(result.total_time_s))
    with col2:
        st.metric("Baseline Time", format_time(result.baseline_time_s))
    with col3:
        st.metric("Time Saved", f"{result.time_saved_s:.1f}s ({result.time_saved_pct:.1f}%)")
    with col4:
        st.metric("Avg Power", f"{result.simulation.avg_power_w:.0f} W")
    
    # Violations warning
    if result.violations:
        st.warning(format_violations(result.violations))
    else:
        st.success("Plan respects all PDC limits")
    
    # Power plot
    st.subheader("Power Profile")
    segments = st.session_state.segments
    fig = create_power_plot(course, result.simulation, segments)
    st.plotly_chart(fig, use_container_width=True)
    
    # Speed plot
    st.subheader("Speed Profile")
    fig = create_speed_plot(course, result.simulation)
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparison plot
    st.subheader("Optimized vs Constant Power")
    fig = create_comparison_plot(course, result.simulation, result.baseline_simulation)
    st.plotly_chart(fig, use_container_width=True)


def render_segments_tab(course: Course, result: OptimizationResult, advanced: dict):
    """Render the Segments tab content."""
    st.subheader("Segment Analysis")
    
    # Auto-segment button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Re-segment Course"):
            st.session_state.segments = auto_segment(
                course, result.simulation,
                grade_change_threshold=advanced['grade_seg_threshold'],
                min_segment_m=advanced['min_segment_m']
            )
            st.rerun()
    
    segments = st.session_state.segments
    
    if not segments:
        st.info("Click 'Re-segment Course' to generate segments")
        return
    
    # Segment table
    df = segments_to_dataframe(segments)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Splits chart
    st.subheader("Segment Times")
    fig = create_splits_chart(segments)
    st.plotly_chart(fig, use_container_width=True)
    
    # Manual editing
    st.subheader("Edit Segments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Split Segment**")
        split_distance = st.number_input(
            "Split at distance (km)",
            min_value=0.0,
            max_value=course.total_distance_m / 1000,
            value=1.0,
            step=0.1
        )
        if st.button("Split"):
            new_segments = split_segment_at_distance(
                segments, course, result.simulation,
                split_distance * 1000
            )
            if len(new_segments) > len(segments):
                st.session_state.segments = new_segments
                st.success(f"Split segment at {split_distance:.1f} km")
                st.rerun()
            else:
                st.warning("Could not split at this distance")
    
    with col2:
        st.write("**Merge Segments**")
        if len(segments) > 1:
            merge_idx = st.selectbox(
                "Merge segment with next",
                options=list(range(len(segments) - 1)),
                format_func=lambda i: f"Segment {i+1} + {i+2}"
            )
            if st.button("Merge"):
                new_segments = merge_segments(
                    segments, course, result.simulation,
                    merge_idx, merge_idx + 1
                )
                st.session_state.segments = new_segments
                st.success(f"Merged segments {merge_idx+1} and {merge_idx+2}")
                st.rerun()


def render_guidance_tab(segments: list, rider: dict):
    """Render the Guidance tab content."""
    st.subheader("Race Guidance")
    
    if not segments:
        st.info("Generate a plan first to see race guidance")
        return
    
    guidance = generate_race_guidance(
        segments,
        hr_max=rider['hr_max'],
        ftp=rider['ftp']
    )
    
    # Summary
    st.info(guidance.pacing_summary)
    
    # HR Guidance table
    st.subheader("Heart Rate Targets")
    hr_data = []
    for hr in guidance.hr_guidance:
        hr_data.append({
            'Segment': hr.segment_name,
            'HR Range': f"{hr.hr_low}-{hr.hr_high}",
            'Zone': hr.intensity_zone,
            'Notes': hr.notes
        })
    st.dataframe(pd.DataFrame(hr_data), use_container_width=True, hide_index=True)
    
    # Key segments
    st.subheader("Key Segments")
    for key in guidance.key_segments:
        st.write(f"• {key}")
    
    # Feel adjustments
    st.subheader("Adjustment Rules")
    for adj in guidance.feel_adjustments:
        with st.expander(f"At {adj.checkpoint_name}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"**GOOD legs:** {adj.if_good}")
            with col2:
                st.info(f"**OK legs:** {adj.if_ok}")
            with col3:
                st.warning(f"**BAD legs:** {adj.if_bad}")
    
    # Printable race card
    st.subheader("Race Card")
    race_card = format_race_card(segments, guidance)
    st.code(race_card, language=None)
    
    st.download_button(
        "Download Race Card",
        race_card,
        file_name="race_card.txt",
        mime="text/plain"
    )


def main():
    """Main application entry point."""
    init_session_state()
    
    st.title("🚴 TT Pacing Tool")
    st.caption("Time-optimal pacing for time trials and hill climbs")
    
    # Sidebar inputs
    rider = sidebar_rider_params()
    anchors = sidebar_power_anchors(rider['ftp'])
    equipment = sidebar_equipment()
    environment = sidebar_environment()
    advanced = sidebar_advanced()
    
    # GPX upload
    st.sidebar.header("Course")
    uploaded_gpx = st.sidebar.file_uploader("Load GPX File", type=['gpx'])
    
    if uploaded_gpx:
        try:
            gpx_content = uploaded_gpx.read().decode('utf-8')
            course = load_course_from_string(
                gpx_content,
                step_m=50.0,
                smoothing_window=advanced['smoothing_window']
            )
            st.session_state.course = course
            st.sidebar.success(f"Loaded: {course.total_distance_m/1000:.1f} km")
        except Exception as e:
            st.sidebar.error(f"Error loading GPX: {e}")
    
    # Generate button
    generate_clicked = st.sidebar.button("Generate Plan", type="primary", use_container_width=True)
    
    if generate_clicked and st.session_state.course is not None:
        course = st.session_state.course
        
        with st.spinner("Optimizing pacing..."):
            # Build configs
            aero_config = AeroConfig(
                cda_flat=equipment['cda_flat'],
                cda_climb=equipment['cda_climb'],
                grade_threshold=equipment['grade_threshold'],
                speed_threshold=equipment['speed_threshold_ms']
            )
            
            physics_config = PhysicsConfig(
                mass_kg=rider['mass'],
                crr=equipment['crr'],
                aero=aero_config
            )
            
            # Get weather
            if environment['use_forecast']:
                weather = fetch_weather(
                    course.start_lat,
                    course.start_lon,
                    environment['race_datetime']
                )
            else:
                env_config = EnvironmentConfig(
                    use_forecast=False,
                    fallback_temp_c=environment['temp'],
                    fallback_pressure_hpa=environment['pressure'],
                    fallback_humidity_pct=environment['humidity'],
                    fallback_wind_speed_ms=environment['wind_speed_ms'],
                    fallback_wind_direction_deg=environment['wind_dir']
                )
                weather = create_fallback_weather(
                    course.start_lat,
                    course.start_lon,
                    env_config
                )
            st.session_state.weather = weather
            
            # Fit PDC
            pdc = fit_pdc(anchors)
            st.session_state.pdc = pdc
            
            # Optimize
            opt_config = OptimizationConfig(
                regularization=advanced['regularization'],
                verbose=False
            )
            
            result = optimize_pacing(
                course, weather, physics_config, pdc, opt_config
            )
            st.session_state.optimization_result = result
            
            # Auto-segment
            segments = auto_segment(
                course, result.simulation,
                grade_change_threshold=advanced['grade_seg_threshold'],
                min_segment_m=advanced['min_segment_m']
            )
            st.session_state.segments = segments
    
    # Main content area with tabs
    course = st.session_state.course
    result = st.session_state.optimization_result
    segments = st.session_state.segments
    
    if course is None:
        st.info("👈 Upload a GPX file to get started")
        
        # Show sample usage
        with st.expander("How to use this tool"):
            st.markdown("""
            1. **Upload a GPX file** of your time trial or hill climb course
            2. **Set your rider parameters** (mass, FTP, HR max)
            3. **Enter your power anchors** or use defaults based on FTP
            4. **Configure equipment** (CdA, rolling resistance)
            5. **Set weather conditions** or fetch forecast
            6. **Click "Generate Plan"** to optimize your pacing
            7. **Review segments** and adjust as needed
            8. **Export your race card** to memorize before the event
            """)
        return
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Course", "Plan", "Segments", "Guidance"])
    
    with tab1:
        render_course_tab(course)
    
    with tab2:
        if result:
            render_plan_tab(result, course)
        else:
            st.info("Click 'Generate Plan' to optimize your pacing")
    
    with tab3:
        if result:
            render_segments_tab(course, result, advanced)
        else:
            st.info("Generate a plan first to see segments")
    
    with tab4:
        if segments:
            render_guidance_tab(segments, rider)
        else:
            st.info("Generate a plan first to see race guidance")
    
    # Export options in sidebar
    if result:
        st.sidebar.header("Export")
        
        # CSV export
        csv_buffer = io.StringIO()
        export_data = []
        for i, (point, sim_point) in enumerate(zip(course.points, result.simulation.points)):
            export_data.append({
                'distance_m': point.distance_m,
                'elevation_m': point.elevation_m,
                'grade_pct': point.grade_pct,
                'target_power_w': sim_point.power_w,
                'predicted_speed_kmh': sim_point.speed_ms * 3.6,
                'cumulative_time_s': sim_point.time_s
            })
        df = pd.DataFrame(export_data)
        csv_data = df.to_csv(index=False)
        
        st.sidebar.download_button(
            "Download Full Plan (CSV)",
            csv_data,
            file_name="pacing_plan.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
