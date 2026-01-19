# TT Pacing Tool

A local desktop Python application that generates time-optimal pacing plans for cycling time trials and hill climbs from GPX course files.

## Features

- **GPX Course Import**: Load any GPX file and automatically resample at 50m resolution with configurable elevation smoothing
- **Weather Integration**: Fetch real-time weather data from Open-Meteo API or use manual fallback values
- **Physics-Based Simulation**: Full aerodynamic model including gravity, rolling resistance, drag, and wind effects
- **Power Duration Curve**: Define your power capabilities via anchor points (5s, 1m, 5m, 20m, 60m)
- **Optimal Pacing**: Generate time-minimizing power plans that respect your physiological limits
- **Segment Analysis**: Auto-segment course by grade changes with manual refinement
- **Race Guidance**: HR bands and legs-feel heuristics for pre-race memorization

## Installation

```bash
cd tt-pacing-tool
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`.

## Module Structure

- `app.py` - Streamlit UI entry point
- `gpx_io.py` - GPX parsing, resampling, smoothing
- `weather.py` - Open-Meteo client, air density, fallback
- `physics.py` - Force model, CdA switching, forward simulation
- `pdc.py` - Power-duration curve fitting
- `optimizer.py` - Objective function, constraints, solver
- `segmentation.py` - Auto/manual sector logic
- `reporting.py` - Plots, tables, export, heuristics

## Quick Start

1. Load a GPX file of your course
2. Enter your rider parameters (mass, FTP, HRmax)
3. Input your power anchor points or import from CSV
4. Set equipment parameters (CdA, Crr)
5. Optionally fetch weather for your race day
6. Click "Generate Plan" to optimize your pacing strategy
7. Review segments and adjust boundaries as needed
8. Export your race guidance card

## Physics Model

The simulation includes:
- Gravitational force based on grade
- Rolling resistance (constant Crr)
- Aerodynamic drag with CdA switching (flat vs climbing position)
- Wind projection onto route bearing
- Drivetrain efficiency losses

## License

MIT
