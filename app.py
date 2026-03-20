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
from optimizer import optimize_pacing, OptimizationConfig, OptimizationResult, compare_bikes
from segmentation import auto_segment, split_segment_at_distance, merge_segments, Segment
from reporting import (
    create_power_plot, create_speed_plot, create_elevation_profile,
    create_splits_chart, create_comparison_plot, segments_to_dataframe,
    generate_race_guidance, format_race_card, export_plan_csv,
    create_segment_overlay_chart
)
from fit_export import create_fit_workout, get_workout_summary
import json
import os


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
    if 'loaded_profile' not in st.session_state:
        st.session_state.loaded_profile = None
    if 'panel_minimized' not in st.session_state:
        st.session_state.panel_minimized = False


def get_sample_courses() -> dict:
    """Scan sample_data/ folder for GPX files."""
    sample_dir = "sample_data"
    courses = {}
    if os.path.exists(sample_dir):
        for filename in sorted(os.listdir(sample_dir)):
            if filename.lower().endswith('.gpx'):
                name = filename[:-4].replace('_', ' ').replace('-', ' ').title()
                courses[name] = os.path.join(sample_dir, filename)
    return courses


def get_sample_gpx(filepath: str) -> str:
    """Load a GPX file content."""
    if filepath and os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return f.read()
    return None


def get_sample_profiles() -> dict:
    """Scan profiles/ folder for JSON profile files."""
    profile_dir = "profiles"
    profiles = {}
    if os.path.exists(profile_dir):
        for filename in sorted(os.listdir(profile_dir)):
            if filename.lower().endswith('.json'):
                name = filename[:-5].replace('_', ' ').replace('-', ' ').title()
                profiles[name] = os.path.join(profile_dir, filename)
    return profiles


def create_user_profile(rider: dict, anchors: dict, equipment: dict, environment: dict, advanced: dict) -> dict:
    """Create a user profile dictionary from current settings."""
    return {
        "version": 1,
        "rider": rider,
        "anchors": anchors,
        "equipment": equipment,
        "environment": {k: v for k, v in environment.items() if k != 'race_datetime'},
        "advanced": advanced
    }


def profile_to_json(profile: dict) -> str:
    """Convert profile to JSON string."""
    return json.dumps(profile, indent=2)


def apply_profile_defaults(profile: dict):
    """Write profile values into session state widget keys before widgets render."""
    rider = profile.get('rider', {})
    anchors = profile.get('anchors', {})
    equipment = profile.get('equipment', {})
    environment = profile.get('environment', {})
    advanced = profile.get('advanced', {})

    def _a(key):  # anchor lookup handles int or str keys
        return anchors.get(key, anchors.get(str(key)))

    # Rider
    if 'rider_weight' in rider: st.session_state['w_rider_weight'] = float(rider['rider_weight'])
    if 'bike_weight' in rider:  st.session_state['w_bike_weight']  = float(rider['bike_weight'])
    if 'ftp'          in rider: st.session_state['w_ftp']           = int(rider['ftp'])
    if 'hr_max'       in rider: st.session_state['w_hr_max']        = int(rider['hr_max'])

    # Power anchors
    if _a(5):    st.session_state['w_p5s']  = int(_a(5))
    if _a(60):   st.session_state['w_p1m']  = int(_a(60))
    if _a(300):  st.session_state['w_p5m']  = int(_a(300))
    if _a(1200): st.session_state['w_p20m'] = int(_a(1200))
    if _a(3600): st.session_state['w_p60m'] = int(_a(3600))

    # Equipment
    bike_type = equipment.get('bike_type', 'tt')
    st.session_state['w_bike_type'] = 'TT Bike' if bike_type == 'tt' else 'Road Bike'
    if 'cda_aero'          in equipment: st.session_state['w_cda_aero']        = float(equipment['cda_aero'])
    if 'cda_non_aero'      in equipment: st.session_state['w_cda_non_aero']    = float(equipment['cda_non_aero'])
    if 'cda_road'          in equipment: st.session_state['w_cda_road']         = float(equipment['cda_road'])
    if 'grade_threshold'   in equipment: st.session_state['w_grade_threshold']  = float(equipment['grade_threshold'])
    if 'speed_threshold_ms' in equipment:
        st.session_state['w_speed_threshold'] = float(equipment['speed_threshold_ms'] * 3.6)
    if 'crr' in equipment: st.session_state['w_crr'] = float(equipment['crr'])

    # Advanced
    if 'smoothing_window'      in advanced: st.session_state['w_smoothing']         = int(advanced['smoothing_window'])
    if 'regularization'        in advanced: st.session_state['w_regularization']    = float(advanced['regularization'])
    if 'grade_seg_threshold'   in advanced: st.session_state['w_grade_seg']         = float(advanced['grade_seg_threshold'])
    if 'min_segment_m'         in advanced: st.session_state['w_min_seg_m']         = int(advanced['min_segment_m'])
    if 'min_segment_duration_s' in advanced: st.session_state['w_min_seg_dur']      = int(advanced['min_segment_duration_s'])
    if 'target_segments'       in advanced: st.session_state['w_target_segs']       = int(advanced['target_segments'])

    # Environment
    if not environment.get('use_forecast', False):
        if 'temp'          in environment: st.session_state['w_temp']       = float(environment['temp'])
        if 'pressure'      in environment: st.session_state['w_pressure']   = float(environment['pressure'])
        if 'humidity'      in environment: st.session_state['w_humidity']   = int(environment['humidity'])
        if 'wind_speed_ms' in environment: st.session_state['w_wind_speed'] = float(environment['wind_speed_ms'] * 3.6)
        if 'wind_dir'      in environment: st.session_state['w_wind_dir']   = int(environment['wind_dir'])


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


def sidebar_inputs():
    """Render all sidebar inputs as three tabs: Rider, Equipment, Course.
    Returns (rider, anchors, equipment, environment, advanced, generate_clicked,
             compare_clicked, alt_bike_weight, alt_cda, alt_cda_aero, alt_cda_non_aero,
             save_placeholder)
    """
    # ── Profile loading: apply any pending profile before widgets render ──
    if st.session_state.get('pending_profile'):
        apply_profile_defaults(st.session_state.pop('pending_profile'))

    tab_course, tab_rider, tab_equip = st.tabs(["Course", "Rider", "Equipment"])

    # ── COURSE TAB ───────────────────────────────────────────────────
    with tab_course:
        sample_courses = get_sample_courses()
        if sample_courses:
            course_options = ["-- Select Course --"] + list(sample_courses.keys())
            selected_sample = st.selectbox("Saved courses", course_options,
                                           key="sample_course", label_visibility="collapsed")
            if selected_sample != "-- Select Course --":
                gpx_content = get_sample_gpx(sample_courses[selected_sample])
                if gpx_content:
                    try:
                        # smoothing_window not yet known here; use session state or default
                        sw = st.session_state.get('w_smoothing', 5)
                        course_obj = load_course_from_string(gpx_content, step_m=50.0, smoothing_window=sw)
                        st.session_state.course = course_obj
                        st.success(f"Loaded: {course_obj.total_distance_m/1000:.1f} km")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.caption("Add .gpx files to sample_data/ to see them here")

        uploaded_gpx = st.file_uploader("Upload GPX", type=['gpx'], label_visibility="collapsed")
        if uploaded_gpx:
            try:
                sw = st.session_state.get('w_smoothing', 5)
                gpx_content = uploaded_gpx.read().decode('utf-8')
                course_obj = load_course_from_string(gpx_content, step_m=50.0, smoothing_window=sw)
                st.session_state.course = course_obj
                st.success(f"Loaded: {course_obj.total_distance_m/1000:.1f} km")
            except Exception as e:
                st.error(f"Error: {e}")

        st.divider()
        st.markdown("**Weather**")
        use_forecast = st.checkbox("Fetch weather forecast", value=False, key='w_use_forecast',
                                   help="Only works within ~14 days of race date")
        if use_forecast:
            default_date = datetime(2026, 5, 16).date()
            race_date = st.date_input("Race Date", value=default_date)
            race_time = st.time_input("Start Time", value=datetime.strptime("09:15", "%H:%M").time())
            race_datetime = datetime.combine(race_date, race_time)
            days_out = (race_date - datetime.now().date()).days
            if days_out > 14:
                st.warning(f"Race is {days_out} days away — forecast may be unreliable.")
            environment = {'use_forecast': True, 'race_datetime': race_datetime}
        else:
            col1, col2 = st.columns(2)
            with col1:
                temp       = st.number_input("Temp (°C)",      value=20.0,   step=1.0, key='w_temp')
                pressure   = st.number_input("Pressure (hPa)", value=1013.0, step=1.0, key='w_pressure')
                humidity   = st.number_input("Humidity (%)",   value=50,     step=1,   key='w_humidity',
                                             min_value=0, max_value=100)
            with col2:
                wind_speed = st.number_input("Wind (km/h)",  value=0.0, step=1.0, key='w_wind_speed',
                                             min_value=0.0)
                wind_dir   = st.number_input("Wind Dir (°)", value=0,   step=1,   key='w_wind_dir',
                                             min_value=0, max_value=359,
                                             help="Direction wind comes FROM (0=N, 90=E)")
            environment = {
                'use_forecast': False,
                'temp': temp, 'pressure': pressure, 'humidity': humidity,
                'wind_speed_ms': wind_speed / 3.6, 'wind_dir': wind_dir,
            }

    # ── RIDER TAB ────────────────────────────────────────────────────
    with tab_rider:
        st.markdown("**Load Profile**")
        sample_profiles = get_sample_profiles()
        if sample_profiles:
            prof_options = ["-- Select --"] + list(sample_profiles.keys())
            selected_prof = st.selectbox("Saved profiles", prof_options,
                                         key="profile_dropdown", label_visibility="collapsed")
            if selected_prof != "-- Select --":
                if st.session_state.get('last_applied_profile') != selected_prof:
                    with open(sample_profiles[selected_prof], 'r', encoding='utf-8') as f:
                        pdata = json.load(f)
                    st.session_state['pending_profile'] = pdata
                    st.session_state['last_applied_profile'] = selected_prof
                    st.rerun()
        else:
            st.caption("Add .json files to profiles/ to see them here")

        uploaded_profile = st.file_uploader("Upload profile (.json)", type=['json'],
                                             key='profile_upload', label_visibility="collapsed")
        if uploaded_profile:
            upload_id = uploaded_profile.name + str(uploaded_profile.size)
            if st.session_state.get('last_upload_id') != upload_id:
                try:
                    pdata = json.loads(uploaded_profile.read().decode('utf-8'))
                    st.session_state['pending_profile'] = pdata
                    st.session_state['last_upload_id'] = upload_id
                    st.session_state['last_applied_profile'] = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading profile: {e}")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            rider_weight = st.number_input("Rider (kg)", min_value=40.0, max_value=150.0,
                                           value=80.0, step=0.5, key='w_rider_weight')
        with col2:
            bike_weight = st.number_input("Bike+Gear (kg)", min_value=5.0, max_value=20.0,
                                          value=8.0, step=0.5, key='w_bike_weight')
        total_mass = rider_weight + bike_weight
        st.caption(f"Total system mass: {total_mass:.1f} kg")

        ftp = st.number_input("FTP (W)", min_value=100, max_value=500,
                               value=350, step=5, key='w_ftp')
        st.caption(f"{ftp/rider_weight:.2f} W/kg")

        hr_max = st.number_input("Max HR", min_value=130, max_value=220,
                                  value=195, step=1, key='w_hr_max')

        st.divider()
        st.markdown("**Power Anchors**")
        use_defaults = st.checkbox("Use defaults from FTP", value=False, key='w_anchor_defaults')
        if use_defaults:
            anchors = default_anchors_from_ftp(ftp)
            st.caption(f"5s: {anchors[5]:.0f}W | 1m: {anchors[60]:.0f}W | 5m: {anchors[300]:.0f}W")
            st.caption(f"20m: {anchors[1200]:.0f}W | 60m: {anchors[3600]:.0f}W")
        else:
            col1, col2 = st.columns(2)
            with col1:
                p5s  = st.number_input("5s",  min_value=100, max_value=2000, value=1300, key='w_p5s')
                p1m  = st.number_input("1m",  min_value=100, max_value=1500, value=585,  key='w_p1m')
                p5m  = st.number_input("5m",  min_value=100, max_value=800,  value=415,  key='w_p5m')
            with col2:
                p20m = st.number_input("20m", min_value=100, max_value=600,  value=357,  key='w_p20m')
                p60m = st.number_input("60m", min_value=100, max_value=500,  value=337,  key='w_p60m')
            anchors = {5: p5s, 60: p1m, 300: p5m, 1200: p20m, 3600: p60m}
            uploaded_csv = st.file_uploader("Import CSV", type=['csv'], key='anchor_csv')
            if uploaded_csv:
                try:
                    df = pd.read_csv(uploaded_csv)
                    if 'duration_s' in df.columns and 'power_w' in df.columns:
                        anchors = dict(zip(df['duration_s'].astype(int), df['power_w']))
                        st.success(f"Loaded {len(anchors)} anchors")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()
        st.markdown("**Save Profile**")
        save_placeholder = st.empty()

    rider = {
        'mass': total_mass, 'rider_weight': rider_weight,
        'bike_weight': bike_weight, 'ftp': ftp, 'hr_max': hr_max,
    }

    # ── EQUIPMENT TAB ─────────────────────────────────────────────────
    with tab_equip:
        bike_type_label = st.radio("Bike Type", ["TT Bike", "Road Bike"], index=0,
                                   key='w_bike_type',
                                   help="TT: aero/non-aero switching; Road: single CdA")

        with st.expander("Aerodynamics", expanded=True):
            if bike_type_label == "TT Bike":
                col1, col2 = st.columns(2)
                with col1:
                    cda_aero = st.number_input("CdA Aero (m²)", min_value=0.15, max_value=0.35,
                                               value=0.22, step=0.01, key='w_cda_aero',
                                               help="Full aero position")
                with col2:
                    cda_non_aero = st.number_input("CdA Bars (m²)", min_value=0.20, max_value=0.40,
                                                   value=0.28, step=0.01, key='w_cda_non_aero',
                                                   help="On bars/drops")
                grade_threshold = st.number_input("Grade Threshold (%)", min_value=0.0, max_value=15.0,
                                                  value=5.0, step=0.5, key='w_grade_threshold',
                                                  help="Switch to bars CdA above this grade")
                speed_threshold = st.number_input("Speed Threshold (km/h)", min_value=10.0, max_value=40.0,
                                                  value=22.0, step=1.0, key='w_speed_threshold',
                                                  help="Switch to bars CdA below this speed")
                cda_road = 0.32
            else:
                cda_road = st.number_input("CdA (m²)", min_value=0.25, max_value=0.45,
                                           value=0.32, step=0.01, key='w_cda_road')
                cda_aero = 0.22
                cda_non_aero = 0.28
                grade_threshold = 5.0
                speed_threshold = 22.0

        with st.expander("Rolling Resistance"):
            crr = st.number_input("Crr", min_value=0.002, max_value=0.010,
                                  value=0.0029, step=0.0001, format="%.4f", key='w_crr')

        with st.expander("Advanced Settings"):
            smoothing_window = st.number_input("Elevation Smoothing Window", min_value=3, max_value=15,
                                               value=5, step=2, key='w_smoothing')
            regularization = st.slider("Pacing Smoothness", min_value=0.01, max_value=2.0,
                                       value=0.1, step=0.05, key='w_regularization',
                                       help="Higher = smoother power changes between segments")
            grade_seg_threshold = st.number_input("Segment Grade Threshold (%)", min_value=0.5,
                                                  max_value=5.0, value=2.0, step=0.5, key='w_grade_seg')
            min_segment_m = st.number_input("Min Segment Length (m)", min_value=100, max_value=2000,
                                            value=200, step=50, key='w_min_seg_m')
            min_segment_duration_s = st.number_input("Min Segment Duration (s)", min_value=0,
                                                     max_value=120, value=30, step=5, key='w_min_seg_dur')
            target_segments = st.number_input("Target # of Segments", min_value=3, max_value=15,
                                              value=6, step=1, key='w_target_segs')

        with st.expander("Compare Bikes"):
            st.caption("Compare TT bike vs Road bike for this course")
            if bike_type_label == "TT Bike":
                st.write("**Road Bike Config (alternate):**")
                alt_bike_weight = st.number_input("Road Bike Weight (kg)", min_value=5.0, max_value=15.0,
                                                  value=7.5, step=0.5, key="alt_bike_weight")
                alt_cda = st.number_input("Road Bike CdA (m²)", min_value=0.25, max_value=0.45,
                                          value=0.32, step=0.01, key="alt_cda")
                alt_cda_aero = alt_cda_non_aero = None
            else:
                st.write("**TT Bike Config (alternate):**")
                alt_bike_weight = st.number_input("TT Bike Weight (kg)", min_value=5.0, max_value=15.0,
                                                  value=8.5, step=0.5, key="alt_bike_weight")
                col1, col2 = st.columns(2)
                with col1:
                    alt_cda_aero = st.number_input("CdA Aero (m²)", min_value=0.15, max_value=0.35,
                                                   value=0.22, step=0.01, key="alt_cda_aero")
                with col2:
                    alt_cda_non_aero = st.number_input("CdA Bars (m²)", min_value=0.20, max_value=0.40,
                                                        value=0.28, step=0.01, key="alt_cda_non_aero")
                alt_cda = None
            compare_clicked = st.button("Compare Bikes", use_container_width=True)

    equipment = {
        'bike_type': 'tt' if bike_type_label == "TT Bike" else 'road',
        'cda_aero': cda_aero, 'cda_non_aero': cda_non_aero, 'cda_road': cda_road,
        'grade_threshold': grade_threshold, 'speed_threshold_ms': speed_threshold / 3.6, 'crr': crr,
    }
    advanced = {
        'smoothing_window': smoothing_window, 'regularization': regularization,
        'grade_seg_threshold': grade_seg_threshold, 'min_segment_m': min_segment_m,
        'min_segment_duration_s': min_segment_duration_s, 'target_segments': target_segments,
    }

    # ── Generate Plan button ──────────────────────────────────────────
    st.divider()
    generate_clicked = st.button("Generate Plan", type="primary", use_container_width=True)

    # ── Fill save-profile placeholder ────────────────────────────────
    current_profile = create_user_profile(rider, anchors, equipment, environment, advanced)
    save_placeholder.download_button(
        "Download Profile",
        profile_to_json(current_profile),
        file_name="tt_pacing_profile.json",
        mime="application/json",
        use_container_width=True,
    )

    return rider, anchors, equipment, environment, advanced, generate_clicked, \
           compare_clicked, alt_bike_weight, alt_cda, alt_cda_aero, alt_cda_non_aero




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
    
    # Summary metrics - row 1
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Predicted Time", format_time(result.total_time_s))
    with col2:
        np_val = result.simulation.normalized_power_w
        st.metric(
            "Baseline Time", 
            format_time(result.baseline_time_s),
            help=f"Time if you held constant {np_val:.0f}W (same NP as optimized plan). The difference shows the benefit of varying power with terrain."
        )
    with col3:
        st.metric("Time Saved", f"{result.time_saved_s:.1f}s ({result.time_saved_pct:.1f}%)")
    with col4:
        st.metric("Avg Power", f"{result.simulation.avg_power_w:.0f} W")
    with col5:
        np_val = result.simulation.normalized_power_w
        vi = result.simulation.variability_index
        st.metric(
            "Normalized Power", 
            f"{np_val:.0f} W",
            delta=f"VI: {vi:.2f}",
            delta_color="off",
            help="NP accounts for the higher physiological cost of variable power. VI (Variability Index) = NP/Avg. VI close to 1.0 means steady pacing."
        )
    
    # Violations warning with explanation
    if result.violations:
        st.warning(format_violations(result.violations))
        with st.expander("What are PDC violations?"):
            st.markdown("""
            **PDC (Power Duration Curve) violations** occur when the optimized plan exceeds what you can 
            physiologically sustain based on your power anchors.
            
            For example, if your 20-minute max is 357W, but the plan asks you to average 370W over any 
            20-minute window, that's a violation.
            
            **Minor violations (1-5%)** may be acceptable if you're willing to dig deeper than your 
            known limits. **Large violations (>10%)** indicate the plan is unrealistic and you should 
            either adjust your expectations or improve your fitness data.
            
            The optimizer tries to avoid violations but may accept small ones if it significantly 
            improves time.
            """)
    else:
        st.success("Plan respects all PDC limits - this effort is physiologically achievable based on your power data")
    
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


def render_gearing_tab(segments: list, sim_result):
    """Render the Gearing tab — cadence distribution for different min gear ratios."""
    st.subheader("Gearing Analysis")
    st.caption("Shows % of riding time spent in each cadence range for different minimum gear ratios (easiest gear).")

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        wheel_circumference_m = st.number_input(
            "Wheel circumference (m)", min_value=1.9, max_value=2.3,
            value=2.096, step=0.001, format="%.3f",
            help="700c x 25mm ≈ 2.096m"
        )
    with col2:
        st.markdown("**Gear ratio** = chainring ÷ cog")

    # Gear options: (label, ratio) sorted ascending by ratio
    gear_options_raw = [
        ("34/34", 34/34),
        ("36/34", 36/34),
        ("34/32", 34/32),
        ("36/32", 36/32),
        ("34/30", 34/30),
        ("36/30", 36/30),
        ("34/28", 34/28),
        ("36/28", 36/28),
        ("34/26", 34/26),
        ("39/28", 39/28),
        ("50/34", 50/34),
    ]
    gear_options_raw.sort(key=lambda x: x[1])
    gear_options = {f"{label} ({ratio:.2f})": ratio for label, ratio in gear_options_raw}

    cadence_buckets = [
        ("<50",  0,  50),
        ("50–60", 50, 60),
        ("60–70", 60, 70),
        ("70–80", 70, 80),
        ("80–90", 80, 90),
        ("90+",  90, float("inf")),
    ]

    # Build time-per-point array across all segment points
    # Use segment_time_s (time spent traversing each point interval)
    all_speeds = []   # m/s
    all_times = []    # seconds per interval

    for seg in segments:
        sim_points = sim_result.points[seg.start_idx:seg.end_idx]
        for p in sim_points:
            all_speeds.append(p.speed_ms)
            all_times.append(p.segment_time_s)

    all_speeds = np.array(all_speeds)
    all_times = np.array(all_times)
    total_time = all_times.sum()

    if total_time == 0:
        st.warning("No time data available.")
        return

    # Build table: rows = gear options, columns = cadence buckets
    rows = []
    for label, ratio in gear_options.items():
        development = ratio * wheel_circumference_m  # metres per revolution
        # cadence (rpm) = speed (m/s) / development (m/rev) * 60
        cadences = all_speeds / development * 60

        row = {"Min Gear": label}
        for bucket_label, lo, hi in cadence_buckets:
            mask = (cadences >= lo) & (cadences < hi)
            secs = int(all_times[mask].sum())
            row[bucket_label] = f"{secs // 60}:{secs % 60:02d}"
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Min Gear")
    st.dataframe(df, use_container_width=True)

    st.caption(
        "Time (m:ss) spent in each cadence range, assuming you are always in your minimum (easiest) gear. "
        "High time in <60 rpm buckets suggests you may need an easier gear for this course."
    )


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
                min_segment_m=advanced['min_segment_m'],
                target_segments=advanced['target_segments'],
                min_segment_duration_s=advanced['min_segment_duration_s']
            )
            st.rerun()
    
    segments = st.session_state.segments
    
    if not segments:
        st.info("Click 'Re-segment Course' to generate segments")
        return
    
    # Segment overlay visualization - the main view
    st.subheader("Pacing Plan Overview")
    fig = create_segment_overlay_chart(course, result.simulation, segments)
    st.plotly_chart(fig, use_container_width=True)
    
    # Segment table
    st.subheader("Segment Details")
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

    has_plan = st.session_state.optimization_result is not None
    minimized = st.session_state.panel_minimized

    if minimized:
        col_in, col_out = st.columns([1, 40])
    elif has_plan:
        col_in, col_out = st.columns([1, 3])
    else:
        col_in, col_out = st.columns([1, 1])

    with col_in:
        # Toggle button — always visible
        toggle_label = "▶" if minimized else "◀"
        if st.button(toggle_label, help="Expand/collapse input panel", use_container_width=True):
            st.session_state.panel_minimized = not minimized
            st.rerun()

        if not minimized:
            (rider, anchors, equipment, environment, advanced,
             generate_clicked, compare_clicked,
             alt_bike_weight, alt_cda, alt_cda_aero, alt_cda_non_aero) = sidebar_inputs()

            # Cache inputs so they survive panel minimization
            st.session_state['_cached_inputs'] = (
                rider, anchors, equipment, environment, advanced,
                alt_bike_weight, alt_cda, alt_cda_aero, alt_cda_non_aero
            )

            # Auto-minimize after Generate Plan is clicked
            if generate_clicked:
                st.session_state.panel_minimized = True
        else:
            # Restore last known inputs from cache rather than using hardcoded defaults
            cached = st.session_state.get('_cached_inputs')
            if cached:
                (rider, anchors, equipment, environment, advanced,
                 alt_bike_weight, alt_cda, alt_cda_aero, alt_cda_non_aero) = cached
            else:
                rider = {'mass': 88, 'rider_weight': 80, 'bike_weight': 8, 'ftp': 350, 'hr_max': 195}
                anchors = {}
                equipment = {'bike_type': 'tt', 'cda_aero': 0.22, 'cda_non_aero': 0.28,
                             'cda_road': 0.32, 'grade_threshold': 5.0, 'speed_threshold_ms': 6.11, 'crr': 0.0029}
                advanced = {'smoothing_window': 5, 'regularization': 0.1, 'grade_seg_threshold': 2.0,
                            'min_segment_m': 200, 'min_segment_duration_s': 30, 'target_segments': 6}
                environment = {'use_forecast': False, 'temp': 20, 'pressure': 1013,
                               'humidity': 50, 'wind_speed_ms': 0, 'wind_dir': 0}
                alt_bike_weight = alt_cda = alt_cda_aero = alt_cda_non_aero = None
            generate_clicked = compare_clicked = False

    with col_out:
        _main_content(rider, anchors, equipment, environment, advanced,
                      generate_clicked, compare_clicked,
                      alt_bike_weight, alt_cda, alt_cda_aero, alt_cda_non_aero)


def _main_content(rider, anchors, equipment, environment, advanced,
                  generate_clicked, compare_clicked,
                  alt_bike_weight, alt_cda, alt_cda_aero, alt_cda_non_aero):
    """Render the right-hand results column."""

    if generate_clicked and st.session_state.course is not None:
        course = st.session_state.course
        
        # Create a container for the optimization log
        log_container = st.container()
        
        with log_container:
            st.subheader("Optimization Progress")
            log_expander = st.expander("View optimization log", expanded=True)
            log_placeholder = log_expander.empty()
            status_placeholder = st.empty()
            
            # Store log messages
            log_messages = []
            
            def log_callback(msg: str):
                """Callback to update the log display."""
                log_messages.append(msg)
                # Show last 20 messages
                log_text = "\n".join(log_messages[-25:])
                log_placeholder.code(log_text, language=None)
            
            def progress_callback(update):
                """Callback for optimization progress."""
                status_placeholder.info(
                    f"Iteration {update.iteration}: Best time {update.total_time_s:.1f}s "
                    f"(saving {update.improvement_s:.1f}s) - Elapsed: {update.elapsed_s:.1f}s"
                )
            
            status_placeholder.info("Starting optimization...")
            
            # Build configs
            log_callback("Building configuration...")
            aero_config = AeroConfig(
                bike_type=equipment['bike_type'],
                cda_aero=equipment['cda_aero'],
                cda_non_aero=equipment['cda_non_aero'],
                cda_road=equipment['cda_road'],
                grade_threshold=equipment['grade_threshold'],
                speed_threshold=equipment['speed_threshold_ms']
            )
            
            physics_config = PhysicsConfig(
                mass_kg=rider['mass'],
                crr=equipment['crr'],
                aero=aero_config
            )
            log_callback(f"  Rider mass: {rider['mass']}kg, Crr: {equipment['crr']}")
            if equipment['bike_type'] == 'tt':
                log_callback(f"  TT bike: CdA aero={equipment['cda_aero']}, CdA bars={equipment['cda_non_aero']}")
            else:
                log_callback(f"  Road bike: CdA={equipment['cda_road']}")
            
            # Get weather
            log_callback("Fetching weather data...")
            if environment['use_forecast']:
                weather = fetch_weather(
                    course.start_lat,
                    course.start_lon,
                    environment['race_datetime']
                )
                log_callback(f"  Forecast: {weather.temperature_c:.1f}°C, wind {weather.wind_speed_ms:.1f}m/s")
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
                log_callback(f"  Manual: {weather.temperature_c:.1f}°C, wind {weather.wind_speed_ms:.1f}m/s")
            log_callback(f"  Air density: {weather.air_density:.3f} kg/m³")
            st.session_state.weather = weather
            
            # Fit PDC
            log_callback("Fitting power-duration curve...")
            pdc = fit_pdc(anchors)
            st.session_state.pdc = pdc
            log_callback(f"  FTP estimate: {pdc.ftp_estimate():.0f}W")
            log_callback(f"  5s max: {pdc.max_power(5):.0f}W, 1m max: {pdc.max_power(60):.0f}W")
            
            # Optimize
            opt_config = OptimizationConfig(
                regularization=advanced['regularization'],
                verbose=False,
                progress_callback=progress_callback
            )
            
            log_callback("")
            result = optimize_pacing(
                course, weather, physics_config, pdc, opt_config,
                log_callback=log_callback
            )
            st.session_state.optimization_result = result
            
            # Auto-segment
            log_callback("")
            log_callback("Auto-segmenting course...")
            segments = auto_segment(
                course, result.simulation,
                grade_change_threshold=advanced['grade_seg_threshold'],
                min_segment_m=advanced['min_segment_m'],
                target_segments=advanced['target_segments'],
                min_segment_duration_s=advanced['min_segment_duration_s']
            )
            st.session_state.segments = segments
            log_callback(f"  Created {len(segments)} segments")
            
            status_placeholder.success(
                f"Optimization complete! Time: {result.total_time_s:.1f}s "
                f"(saved {result.time_saved_s:.1f}s / {result.time_saved_pct:.1f}%)"
            )
            st.rerun()
    
    # Handle bike comparison
    if compare_clicked and st.session_state.course is not None:
        course = st.session_state.course
        
        st.subheader("Bike Comparison")
        comparison_log = st.expander("Comparison Log", expanded=True)
        log_placeholder = comparison_log.empty()
        log_messages = []
        
        def log_callback(msg: str):
            log_messages.append(msg)
            log_placeholder.code("\n".join(log_messages[-20:]), language=None)
        
        # Build weather (use session state if available, otherwise create fallback)
        if st.session_state.weather:
            weather = st.session_state.weather
        else:
            env_config = EnvironmentConfig(
                use_forecast=False,
                fallback_temp_c=environment.get('temp', 20),
                fallback_pressure_hpa=environment.get('pressure', 1013),
                fallback_humidity_pct=environment.get('humidity', 50),
                fallback_wind_speed_ms=environment.get('wind_speed_ms', 0),
                fallback_wind_direction_deg=environment.get('wind_dir', 0)
            )
            weather = create_fallback_weather(course.start_lat, course.start_lon, env_config)
        
        # Build PDC
        pdc = fit_pdc(anchors)
        
        # Build both bike configs
        if equipment['bike_type'] == 'tt':
            # Current is TT, alternate is Road
            tt_aero = AeroConfig(
                bike_type='tt',
                cda_aero=equipment['cda_aero'],
                cda_non_aero=equipment['cda_non_aero'],
                grade_threshold=equipment['grade_threshold'],
                speed_threshold=equipment['speed_threshold_ms']
            )
            road_aero = AeroConfig(
                bike_type='road',
                cda_road=alt_cda
            )
            tt_mass = rider['mass']
            road_mass = rider['rider_weight'] + alt_bike_weight
        else:
            # Current is Road, alternate is TT
            road_aero = AeroConfig(
                bike_type='road',
                cda_road=equipment['cda_road']
            )
            tt_aero = AeroConfig(
                bike_type='tt',
                cda_aero=alt_cda_aero,
                cda_non_aero=alt_cda_non_aero,
                grade_threshold=5.0,
                speed_threshold=6.0
            )
            road_mass = rider['mass']
            tt_mass = rider['rider_weight'] + alt_bike_weight
        
        tt_physics = PhysicsConfig(mass_kg=tt_mass, crr=equipment['crr'], aero=tt_aero)
        road_physics = PhysicsConfig(mass_kg=road_mass, crr=equipment['crr'], aero=road_aero)
        
        opt_config = OptimizationConfig(regularization=advanced['regularization'], verbose=False)
        
        # Run comparison
        comparison = compare_bikes(
            course, weather, tt_physics, road_physics, pdc, opt_config, log_callback
        )
        
        # Display results
        st.write("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("TT Bike", f"{comparison['tt_time']/60:.1f} min")
        with col2:
            st.metric("Road Bike", f"{comparison['road_time']/60:.1f} min")
        with col3:
            delta = comparison['time_delta']
            if comparison['recommended'] == 'tt':
                st.metric("Recommendation", "TT Bike", delta=f"{abs(delta):.1f}s faster")
            elif comparison['recommended'] == 'road':
                st.metric("Recommendation", "Road Bike", delta=f"{abs(delta):.1f}s faster")
            else:
                st.metric("Recommendation", "Either", delta=f"Within {abs(delta):.1f}s")
        
        st.info(comparison['reason'])
    
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Course", "Plan", "Segments", "Gearing", "Guidance"])

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
        if segments and result:
            render_gearing_tab(segments, result.simulation)
        else:
            st.info("Generate a plan first to see gearing analysis")

    with tab5:
        if segments:
            render_guidance_tab(segments, rider)
        else:
            st.info("Generate a plan first to see race guidance")
    
    # Export options (below results tabs)
    if result:
        st.divider()
        st.subheader("Export")
        ecol1, ecol2 = st.columns(2)

        # CSV export
        export_data = []
        for point, sim_point in zip(course.points, result.simulation.points):
            export_data.append({
                'distance_m': point.distance_m,
                'elevation_m': point.elevation_m,
                'grade_pct': point.grade_pct,
                'target_power_w': sim_point.power_w,
                'predicted_speed_kmh': sim_point.speed_ms * 3.6,
                'cumulative_time_s': sim_point.time_s,
            })
        csv_data = pd.DataFrame(export_data).to_csv(index=False)
        with ecol1:
            st.download_button("Download Full Plan (CSV)", csv_data,
                               file_name="pacing_plan.csv", mime="text/csv",
                               use_container_width=True)

        # Garmin FIT workout export
        if segments:
            workout_name = "TT Pacing"
            if hasattr(course, 'name') and course.name:
                workout_name = course.name[:15]
            power_range = st.slider("Target Power Range (+/- W)", min_value=5, max_value=30,
                                    value=10, step=5, help="Power target tolerance on Garmin")
            try:
                fit_data = create_fit_workout(segments, workout_name=workout_name,
                                              power_range_watts=power_range)
                with ecol2:
                    st.download_button("Download Garmin Workout (.fit)", fit_data,
                                       file_name=f"{workout_name.replace(' ', '_')}_workout.fit",
                                       mime="application/octet-stream", use_container_width=True)
                with st.expander("Workout Preview"):
                    st.text(get_workout_summary(segments))
                    st.caption("Transfer via Garmin Connect or copy to `/Garmin/NewFiles/`")
            except Exception as e:
                st.error(f"Error generating workout: {e}")


if __name__ == "__main__":
    main()
