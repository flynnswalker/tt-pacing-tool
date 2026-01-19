"""
Weather Module - Open-Meteo API client, air density calculation, wind projection.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple
import requests


@dataclass
class WeatherData:
    """Weather conditions at a location."""
    wind_speed_ms: float         # Wind speed in m/s
    wind_direction_deg: float    # Direction wind is coming FROM (0=N, 90=E)
    temperature_c: float         # Temperature in Celsius
    humidity_pct: float          # Relative humidity (0-100)
    pressure_hpa: float          # Atmospheric pressure in hPa
    air_density: float           # Computed air density in kg/m³
    
    @property
    def wind_speed_kmh(self) -> float:
        return self.wind_speed_ms * 3.6
    
    @property
    def temperature_f(self) -> float:
        return self.temperature_c * 9/5 + 32


@dataclass
class EnvironmentConfig:
    """Configuration for environment/weather handling."""
    use_forecast: bool = True
    fallback_temp_c: float = 20.0
    fallback_pressure_hpa: float = 1013.25
    fallback_humidity_pct: float = 50.0
    fallback_wind_speed_ms: float = 0.0
    fallback_wind_direction_deg: float = 0.0


# Physical constants
R_DRY = 287.05      # Specific gas constant for dry air (J/(kg·K))
R_VAPOR = 461.495   # Specific gas constant for water vapor (J/(kg·K))


def compute_air_density(
    temp_c: float,
    pressure_hpa: float,
    humidity_pct: float
) -> float:
    """
    Compute air density using the ideal gas law with humidity correction.
    
    Uses the formula for humid air density accounting for water vapor
    partial pressure.
    
    Args:
        temp_c: Temperature in Celsius
        pressure_hpa: Atmospheric pressure in hPa (mbar)
        humidity_pct: Relative humidity (0-100)
        
    Returns:
        Air density in kg/m³
    """
    # Convert to SI units
    temp_k = temp_c + 273.15
    pressure_pa = pressure_hpa * 100
    humidity_frac = humidity_pct / 100.0
    
    # Saturation vapor pressure (Tetens formula)
    e_sat = 6.1078 * 10 ** (7.5 * temp_c / (temp_c + 237.3)) * 100  # Pa
    
    # Actual vapor pressure
    e_vapor = humidity_frac * e_sat
    
    # Partial pressure of dry air
    p_dry = pressure_pa - e_vapor
    
    # Density using ideal gas law for mixture
    # rho = p_dry / (R_dry * T) + e_vapor / (R_vapor * T)
    rho = (p_dry / (R_DRY * temp_k)) + (e_vapor / (R_VAPOR * temp_k))
    
    return rho


def standard_atmosphere_density(altitude_m: float) -> Tuple[float, float, float]:
    """
    Compute standard atmosphere values at a given altitude.
    
    Uses the International Standard Atmosphere (ISA) model.
    
    Args:
        altitude_m: Altitude in meters above sea level
        
    Returns:
        Tuple of (temperature_c, pressure_hpa, density_kg_m3)
    """
    # Sea level values
    T0 = 288.15  # K (15°C)
    P0 = 101325  # Pa
    
    # Temperature lapse rate (troposphere)
    L = 0.0065  # K/m
    
    # Gravity and gas constant
    g = 9.80665  # m/s²
    M = 0.0289644  # kg/mol (molar mass of air)
    R = 8.31447  # J/(mol·K)
    
    # Temperature at altitude
    T = T0 - L * altitude_m
    temp_c = T - 273.15
    
    # Pressure at altitude (barometric formula)
    P = P0 * (T / T0) ** (g * M / (R * L))
    pressure_hpa = P / 100
    
    # Density (ideal gas law)
    rho = P * M / (R * T)
    
    return temp_c, pressure_hpa, rho


def fallback_air_density(
    altitude_m: float,
    temp_c: Optional[float] = None,
    pressure_hpa: Optional[float] = None,
    humidity_pct: float = 50.0
) -> float:
    """
    Compute air density when forecast unavailable.
    
    Uses standard atmosphere as baseline, with optional user overrides.
    
    Args:
        altitude_m: Altitude in meters
        temp_c: Optional temperature override
        pressure_hpa: Optional pressure override
        humidity_pct: Humidity assumption
        
    Returns:
        Air density in kg/m³
    """
    std_temp, std_pressure, _ = standard_atmosphere_density(altitude_m)
    
    actual_temp = temp_c if temp_c is not None else std_temp
    actual_pressure = pressure_hpa if pressure_hpa is not None else std_pressure
    
    return compute_air_density(actual_temp, actual_pressure, humidity_pct)


def project_wind(
    wind_speed_ms: float,
    wind_direction_deg: float,
    bearing_deg: float
) -> float:
    """
    Project wind onto route bearing to get headwind component.
    
    Positive result = headwind (resistance)
    Negative result = tailwind (assistance)
    
    Args:
        wind_speed_ms: Wind speed in m/s
        wind_direction_deg: Direction wind is coming FROM (0=N, 90=E)
        bearing_deg: Direction of travel (0=N, 90=E)
        
    Returns:
        Headwind component in m/s (positive = headwind)
    """
    # Wind direction is where wind comes FROM
    # Bearing is where we're going TO
    # Headwind = wind coming from the direction we're going
    
    # Calculate relative angle
    # If riding north (0°) and wind from north (0°), full headwind
    # If riding north (0°) and wind from south (180°), full tailwind
    relative_angle = wind_direction_deg - bearing_deg
    
    # Headwind component
    headwind = wind_speed_ms * math.cos(math.radians(relative_angle))
    
    return headwind


def fetch_weather(
    lat: float,
    lon: float,
    start_time: Optional[datetime] = None,
    config: Optional[EnvironmentConfig] = None
) -> WeatherData:
    """
    Fetch weather data from Open-Meteo API.
    
    Args:
        lat: Latitude of location
        lon: Longitude of location
        start_time: Event start time for forecast (None = current)
        config: Environment configuration with fallback values
        
    Returns:
        WeatherData object with conditions
    """
    if config is None:
        config = EnvironmentConfig()
    
    if not config.use_forecast:
        # Use fallback values
        return create_fallback_weather(lat, lon, config)
    
    try:
        # Build API URL
        base_url = "https://api.open-meteo.com/v1/forecast"
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m",
            "timezone": "auto"
        }
        
        # Add forecast days if needed
        if start_time:
            # Open-Meteo provides up to 16 days forecast
            params["forecast_days"] = 16
        
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Parse response
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        humidities = hourly.get("relative_humidity_2m", [])
        pressures = hourly.get("surface_pressure", [])
        wind_speeds = hourly.get("wind_speed_10m", [])
        wind_dirs = hourly.get("wind_direction_10m", [])
        
        if not times:
            raise ValueError("No forecast data available")
        
        # Find closest time to start_time
        if start_time:
            target_str = start_time.strftime("%Y-%m-%dT%H:00")
            try:
                idx = times.index(target_str)
            except ValueError:
                # Find nearest
                idx = 0
                for i, t in enumerate(times):
                    if t <= target_str:
                        idx = i
        else:
            idx = 0  # Current conditions
        
        # Extract values with bounds checking
        temp_c = temps[idx] if idx < len(temps) else config.fallback_temp_c
        humidity = humidities[idx] if idx < len(humidities) else config.fallback_humidity_pct
        pressure = pressures[idx] if idx < len(pressures) else config.fallback_pressure_hpa
        wind_speed_kmh = wind_speeds[idx] if idx < len(wind_speeds) else config.fallback_wind_speed_ms * 3.6
        wind_dir = wind_dirs[idx] if idx < len(wind_dirs) else config.fallback_wind_direction_deg
        
        # Convert wind speed from km/h to m/s
        wind_speed_ms = wind_speed_kmh / 3.6
        
        # Compute air density
        air_density = compute_air_density(temp_c, pressure, humidity)
        
        return WeatherData(
            wind_speed_ms=wind_speed_ms,
            wind_direction_deg=wind_dir,
            temperature_c=temp_c,
            humidity_pct=humidity,
            pressure_hpa=pressure,
            air_density=air_density
        )
        
    except Exception as e:
        print(f"Weather fetch failed: {e}, using fallback values")
        return create_fallback_weather(lat, lon, config)


def create_fallback_weather(
    lat: float,
    lon: float,
    config: EnvironmentConfig
) -> WeatherData:
    """
    Create weather data from fallback configuration.
    
    Args:
        lat: Latitude (not used but kept for consistency)
        lon: Longitude (not used but kept for consistency)
        config: Environment configuration
        
    Returns:
        WeatherData with fallback values
    """
    air_density = compute_air_density(
        config.fallback_temp_c,
        config.fallback_pressure_hpa,
        config.fallback_humidity_pct
    )
    
    return WeatherData(
        wind_speed_ms=config.fallback_wind_speed_ms,
        wind_direction_deg=config.fallback_wind_direction_deg,
        temperature_c=config.fallback_temp_c,
        humidity_pct=config.fallback_humidity_pct,
        pressure_hpa=config.fallback_pressure_hpa,
        air_density=air_density
    )


class WindModel:
    """
    Wind model interface for per-location wind variation.
    
    V1 implementation uses constant wind across course.
    Future versions can implement gridded/interpolated wind.
    """
    
    def __init__(self, weather: WeatherData):
        """
        Initialize wind model.
        
        Args:
            weather: Base weather data
        """
        self.base_weather = weather
    
    def get_wind_at_point(
        self,
        lat: float,
        lon: float,
        elevation_m: float
    ) -> Tuple[float, float]:
        """
        Get wind speed and direction at a specific point.
        
        V1: Returns constant values from base weather.
        Future: Could interpolate from grid or adjust for terrain.
        
        Args:
            lat: Latitude
            lon: Longitude
            elevation_m: Elevation at point
            
        Returns:
            Tuple of (wind_speed_ms, wind_direction_deg)
        """
        # V1: Constant wind across course
        return (self.base_weather.wind_speed_ms, 
                self.base_weather.wind_direction_deg)
    
    def get_headwind_at_point(
        self,
        lat: float,
        lon: float,
        elevation_m: float,
        bearing_deg: float
    ) -> float:
        """
        Get headwind component at a specific point.
        
        Args:
            lat: Latitude
            lon: Longitude
            elevation_m: Elevation at point
            bearing_deg: Direction of travel
            
        Returns:
            Headwind component in m/s (positive = headwind)
        """
        wind_speed, wind_dir = self.get_wind_at_point(lat, lon, elevation_m)
        return project_wind(wind_speed, wind_dir, bearing_deg)
    
    def get_air_density_at_point(
        self,
        lat: float,
        lon: float,
        elevation_m: float
    ) -> float:
        """
        Get air density at a specific point, adjusted for altitude.
        
        Args:
            lat: Latitude
            lon: Longitude
            elevation_m: Elevation at point
            
        Returns:
            Air density in kg/m³
        """
        # Adjust pressure for altitude difference
        # Simple exponential decay approximation
        sea_level_pressure = self.base_weather.pressure_hpa
        
        # Scale height of atmosphere ~8500m
        scale_height = 8500
        pressure_at_altitude = sea_level_pressure * math.exp(-elevation_m / scale_height)
        
        return compute_air_density(
            self.base_weather.temperature_c,
            pressure_at_altitude,
            self.base_weather.humidity_pct
        )
