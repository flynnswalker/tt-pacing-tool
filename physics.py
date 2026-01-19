"""
Physics Module - Force model, CdA switching, forward simulation.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from scipy.optimize import brentq

from gpx_io import Course, CoursePoint
from weather import WeatherData, WindModel


# Physical constants
GRAVITY = 9.80665  # m/s²


@dataclass
class AeroConfig:
    """Aerodynamic configuration."""
    cda_flat: float = 0.25       # CdA in TT/aero position (m²)
    cda_climb: float = 0.32      # CdA in climbing position (m²)
    grade_threshold: float = 6.0  # Switch to climb CdA above this grade (%)
    speed_threshold: float = 6.0  # Switch to climb CdA below this speed (m/s, ~22 km/h)


@dataclass
class PhysicsConfig:
    """Physics simulation configuration."""
    mass_kg: float = 75.0           # Total mass (rider + bike + gear)
    crr: float = 0.004              # Rolling resistance coefficient
    drivetrain_eff: float = 0.97    # Drivetrain efficiency
    p_min: float = 50.0             # Minimum power (coasting/descents)
    aero: AeroConfig = field(default_factory=AeroConfig)
    
    @property
    def rider_mass_kg(self) -> float:
        """Estimate rider mass (total - typical bike weight)."""
        return max(self.mass_kg - 8.0, 50.0)


@dataclass
class SimulationPoint:
    """Result at a single simulation point."""
    distance_m: float
    elevation_m: float
    grade_pct: float
    power_w: float
    speed_ms: float
    time_s: float           # Time to reach this point
    segment_time_s: float   # Time for this segment
    cda_used: float
    headwind_ms: float
    air_density: float


@dataclass
class SimulationResult:
    """Complete simulation result."""
    points: List[SimulationPoint]
    total_time_s: float
    total_distance_m: float
    avg_power_w: float
    avg_speed_ms: float
    
    def get_times(self) -> np.ndarray:
        return np.array([p.time_s for p in self.points])
    
    def get_speeds(self) -> np.ndarray:
        return np.array([p.speed_ms for p in self.points])
    
    def get_powers(self) -> np.ndarray:
        return np.array([p.power_w for p in self.points])
    
    def get_distances(self) -> np.ndarray:
        return np.array([p.distance_m for p in self.points])


def get_cda(
    grade_pct: float,
    speed_ms: float,
    config: AeroConfig
) -> float:
    """
    Determine CdA based on grade and speed.
    
    Uses climbing position when grade is steep OR speed is low.
    
    Args:
        grade_pct: Current grade in percent
        speed_ms: Current speed in m/s
        config: Aero configuration
        
    Returns:
        CdA value in m²
    """
    if grade_pct > config.grade_threshold or speed_ms < config.speed_threshold:
        return config.cda_climb
    return config.cda_flat


def compute_gravity_force(
    mass_kg: float,
    grade_pct: float
) -> float:
    """
    Compute gravitational force component along road.
    
    Positive = resistance (uphill), Negative = assistance (downhill)
    
    Args:
        mass_kg: Total mass in kg
        grade_pct: Grade in percent
        
    Returns:
        Force in Newtons
    """
    # Convert grade to angle
    theta = math.atan(grade_pct / 100.0)
    return mass_kg * GRAVITY * math.sin(theta)


def compute_rolling_resistance(
    mass_kg: float,
    grade_pct: float,
    crr: float
) -> float:
    """
    Compute rolling resistance force.
    
    Args:
        mass_kg: Total mass in kg
        grade_pct: Grade in percent
        crr: Rolling resistance coefficient
        
    Returns:
        Force in Newtons
    """
    theta = math.atan(grade_pct / 100.0)
    normal_force = mass_kg * GRAVITY * math.cos(theta)
    return crr * normal_force


def compute_aero_drag(
    speed_ms: float,
    headwind_ms: float,
    air_density: float,
    cda: float
) -> float:
    """
    Compute aerodynamic drag force.
    
    Args:
        speed_ms: Ground speed in m/s
        headwind_ms: Headwind component (positive = headwind)
        air_density: Air density in kg/m³
        cda: Drag area in m²
        
    Returns:
        Force in Newtons
    """
    # Air speed is ground speed plus headwind
    air_speed = speed_ms + headwind_ms
    
    # Drag is always positive (resistance)
    # Direction of drag force depends on relative wind
    if air_speed >= 0:
        return 0.5 * air_density * cda * air_speed ** 2
    else:
        # Tailwind exceeds ground speed - drag assists
        return -0.5 * air_density * cda * air_speed ** 2


def compute_total_resistance(
    speed_ms: float,
    grade_pct: float,
    headwind_ms: float,
    air_density: float,
    config: PhysicsConfig
) -> float:
    """
    Compute total resistance force.
    
    Args:
        speed_ms: Ground speed in m/s
        grade_pct: Grade in percent
        headwind_ms: Headwind component
        air_density: Air density in kg/m³
        config: Physics configuration
        
    Returns:
        Total resistance force in Newtons
    """
    cda = get_cda(grade_pct, speed_ms, config.aero)
    
    f_gravity = compute_gravity_force(config.mass_kg, grade_pct)
    f_rolling = compute_rolling_resistance(config.mass_kg, grade_pct, config.crr)
    f_aero = compute_aero_drag(speed_ms, headwind_ms, air_density, cda)
    
    return f_gravity + f_rolling + f_aero


def compute_required_power(
    speed_ms: float,
    grade_pct: float,
    headwind_ms: float,
    air_density: float,
    config: PhysicsConfig
) -> float:
    """
    Compute power required to maintain a given speed.
    
    Args:
        speed_ms: Ground speed in m/s
        grade_pct: Grade in percent
        headwind_ms: Headwind component
        air_density: Air density in kg/m³
        config: Physics configuration
        
    Returns:
        Required power at the pedals in Watts
    """
    if speed_ms <= 0:
        return 0.0
    
    total_force = compute_total_resistance(
        speed_ms, grade_pct, headwind_ms, air_density, config
    )
    
    # Power at wheel = force × velocity
    power_wheel = total_force * speed_ms
    
    # Power at pedals (account for drivetrain losses)
    if power_wheel >= 0:
        # Positive power: pedal power > wheel power
        power_pedal = power_wheel / config.drivetrain_eff
    else:
        # Negative power (descending): coasting
        power_pedal = 0.0
    
    return max(0.0, power_pedal)


def compute_speed_from_power(
    power_w: float,
    grade_pct: float,
    headwind_ms: float,
    air_density: float,
    config: PhysicsConfig,
    initial_guess: float = 10.0
) -> float:
    """
    Compute steady-state speed for a given power output.
    
    Uses root-finding to solve: power = resistance_force × speed
    
    Args:
        power_w: Power at pedals in Watts
        grade_pct: Grade in percent
        headwind_ms: Headwind component
        air_density: Air density in kg/m³
        config: Physics configuration
        initial_guess: Starting speed for solver
        
    Returns:
        Speed in m/s
    """
    # Effective power at wheel
    power_wheel = power_w * config.drivetrain_eff
    
    # Define equation to solve: power_wheel - F_total × v = 0
    def equation(v):
        if v <= 0:
            return -power_wheel
        
        cda = get_cda(grade_pct, v, config.aero)
        
        f_gravity = compute_gravity_force(config.mass_kg, grade_pct)
        f_rolling = compute_rolling_resistance(config.mass_kg, grade_pct, config.crr)
        f_aero = compute_aero_drag(v, headwind_ms, air_density, cda)
        
        f_total = f_gravity + f_rolling + f_aero
        
        return power_wheel - f_total * v
    
    # Speed bounds
    v_min = 0.5   # Minimum speed (avoid singularities)
    v_max = 30.0  # Maximum realistic speed (108 km/h)
    
    # Check if solution exists in range
    f_min = equation(v_min)
    f_max = equation(v_max)
    
    if f_min * f_max > 0:
        # No sign change - return boundary speed
        if f_min > 0:
            # Power exceeds resistance even at v_max
            return v_max
        else:
            # Power insufficient even at v_min
            # This can happen on steep uphills
            return v_min
    
    try:
        speed = brentq(equation, v_min, v_max, xtol=0.001)
        return speed
    except ValueError:
        # Fallback to initial guess
        return initial_guess


def compute_terminal_velocity(
    grade_pct: float,
    headwind_ms: float,
    air_density: float,
    config: PhysicsConfig,
    p_min: Optional[float] = None
) -> float:
    """
    Compute terminal (coasting) velocity on a descent.
    
    Args:
        grade_pct: Grade in percent (negative for descent)
        headwind_ms: Headwind component
        air_density: Air density in kg/m³
        config: Physics configuration
        p_min: Minimum power (defaults to config.p_min)
        
    Returns:
        Terminal velocity in m/s
    """
    if p_min is None:
        p_min = config.p_min
    
    # On descents, find speed where forces balance
    return compute_speed_from_power(
        p_min, grade_pct, headwind_ms, air_density, config
    )


def simulate_segment(
    power_w: float,
    distance_m: float,
    grade_pct: float,
    headwind_ms: float,
    air_density: float,
    config: PhysicsConfig,
    initial_speed_ms: float = 0.0
) -> Tuple[float, float]:
    """
    Simulate a single segment with given power.
    
    Uses quasi-steady-state approximation (assumes speed adjusts quickly).
    
    Args:
        power_w: Power at pedals in Watts
        distance_m: Segment distance in meters
        grade_pct: Grade in percent
        headwind_ms: Headwind component
        air_density: Air density in kg/m³
        config: Physics configuration
        initial_speed_ms: Speed entering segment (for future transient sim)
        
    Returns:
        Tuple of (final_speed_ms, segment_time_s)
    """
    # Enforce minimum power
    effective_power = max(power_w, config.p_min)
    
    # Compute steady-state speed
    speed = compute_speed_from_power(
        effective_power, grade_pct, headwind_ms, air_density, config
    )
    
    # Clamp speed to reasonable bounds
    speed = max(speed, 1.0)  # Minimum 1 m/s
    speed = min(speed, 25.0)  # Maximum ~90 km/h
    
    # Time for segment
    time_s = distance_m / speed
    
    return speed, time_s


def simulate_course(
    course: Course,
    power_array: np.ndarray,
    weather: WeatherData,
    config: PhysicsConfig
) -> SimulationResult:
    """
    Simulate entire course with given power profile.
    
    Args:
        course: Course object with resampled points
        power_array: Power for each segment in Watts
        weather: Weather conditions
        config: Physics configuration
        
    Returns:
        SimulationResult with full detail
    """
    wind_model = WindModel(weather)
    
    n_points = len(course.points)
    if len(power_array) != n_points:
        raise ValueError(f"Power array length {len(power_array)} != course points {n_points}")
    
    sim_points = []
    cumulative_time = 0.0
    prev_speed = 0.0
    
    for i, point in enumerate(course.points):
        power = power_array[i]
        
        # Get segment distance (distance to next point)
        if i < n_points - 1:
            segment_dist = course.points[i+1].distance_m - point.distance_m
        else:
            segment_dist = 0.0
        
        # Get environment at this point
        headwind = wind_model.get_headwind_at_point(
            point.lat, point.lon, point.elevation_m, point.bearing_deg
        )
        air_density = wind_model.get_air_density_at_point(
            point.lat, point.lon, point.elevation_m
        )
        
        # Get CdA for this point
        cda = get_cda(point.grade_pct, prev_speed if prev_speed > 0 else 5.0, config.aero)
        
        # Simulate segment
        if segment_dist > 0:
            speed, segment_time = simulate_segment(
                power, segment_dist, point.grade_pct, headwind, air_density, config
            )
        else:
            speed = prev_speed if prev_speed > 0 else 5.0
            segment_time = 0.0
        
        sim_points.append(SimulationPoint(
            distance_m=point.distance_m,
            elevation_m=point.elevation_m,
            grade_pct=point.grade_pct,
            power_w=power,
            speed_ms=speed,
            time_s=cumulative_time,
            segment_time_s=segment_time,
            cda_used=cda,
            headwind_ms=headwind,
            air_density=air_density
        ))
        
        cumulative_time += segment_time
        prev_speed = speed
    
    # Compute averages
    total_time = cumulative_time
    total_distance = course.total_distance_m
    avg_power = np.mean(power_array)
    avg_speed = total_distance / total_time if total_time > 0 else 0.0
    
    return SimulationResult(
        points=sim_points,
        total_time_s=total_time,
        total_distance_m=total_distance,
        avg_power_w=avg_power,
        avg_speed_ms=avg_speed
    )


def simulate_constant_power(
    course: Course,
    power_w: float,
    weather: WeatherData,
    config: PhysicsConfig
) -> SimulationResult:
    """
    Simulate course at constant power.
    
    Args:
        course: Course object
        power_w: Constant power in Watts
        weather: Weather conditions
        config: Physics configuration
        
    Returns:
        SimulationResult
    """
    power_array = np.full(len(course.points), power_w)
    return simulate_course(course, power_array, weather, config)


def estimate_time_at_power(
    course: Course,
    power_w: float,
    weather: WeatherData,
    config: PhysicsConfig
) -> float:
    """
    Quick estimate of course time at constant power.
    
    Args:
        course: Course object
        power_w: Constant power in Watts
        weather: Weather conditions
        config: Physics configuration
        
    Returns:
        Estimated time in seconds
    """
    result = simulate_constant_power(course, power_w, weather, config)
    return result.total_time_s
