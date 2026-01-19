"""
Optimizer Module - Objective function, regularization, L-BFGS-B solver.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import numpy as np
from scipy.optimize import minimize, Bounds

from gpx_io import Course
from weather import WeatherData
from physics import PhysicsConfig, simulate_course, SimulationResult
from pdc import PDCModel, compute_pdc_penalty, check_feasibility


@dataclass
class OptimizationConfig:
    """Configuration for the optimizer."""
    regularization: float = 0.1       # Penalty weight for power variability
    pdc_penalty_weight: float = 1000.0  # Penalty weight for PDC violations
    max_iterations: int = 500         # Maximum solver iterations
    tolerance: float = 1e-6           # Convergence tolerance
    initial_power_frac: float = 0.95  # Initial power as fraction of FTP
    verbose: bool = False


@dataclass 
class OptimizationResult:
    """Result from optimization."""
    power_plan: np.ndarray           # Optimized power for each segment
    simulation: SimulationResult      # Simulation with optimized plan
    baseline_simulation: SimulationResult  # Constant-power baseline
    time_saved_s: float              # Time improvement over baseline
    time_saved_pct: float            # Percentage improvement
    iterations: int                   # Solver iterations used
    converged: bool                   # Whether solver converged
    violations: list                  # PDC violations in final plan
    
    @property
    def total_time_s(self) -> float:
        return self.simulation.total_time_s
    
    @property
    def baseline_time_s(self) -> float:
        return self.baseline_simulation.total_time_s


def create_objective_function(
    course: Course,
    weather: WeatherData,
    physics_config: PhysicsConfig,
    pdc: PDCModel,
    opt_config: OptimizationConfig
) -> Callable[[np.ndarray], float]:
    """
    Create the objective function for optimization.
    
    The objective minimizes:
    - Total time
    - Plus regularization penalty for power variability
    - Plus soft penalty for PDC violations
    
    Args:
        course: Course to optimize
        weather: Weather conditions
        physics_config: Physics parameters
        pdc: Power duration curve
        opt_config: Optimization configuration
        
    Returns:
        Objective function that takes power array and returns scalar cost
    """
    def objective(power_array: np.ndarray) -> float:
        # Simulate the course with this power plan
        sim = simulate_course(course, power_array, weather, physics_config)
        
        # Base objective: total time
        total_time = sim.total_time_s
        
        # Regularization: penalize large power changes
        power_changes = np.diff(power_array)
        smoothness_penalty = opt_config.regularization * np.sum(power_changes ** 2)
        
        # PDC violation penalty
        times = sim.get_times()
        pdc_penalty = compute_pdc_penalty(
            power_array, times, pdc,
            penalty_weight=opt_config.pdc_penalty_weight
        )
        
        return total_time + smoothness_penalty + pdc_penalty
    
    return objective


def create_bounds(
    n_segments: int,
    pdc: PDCModel,
    physics_config: PhysicsConfig,
    expected_duration_s: float
) -> Bounds:
    """
    Create power bounds for optimization.
    
    Args:
        n_segments: Number of segments
        pdc: Power duration curve
        physics_config: Physics parameters (for p_min)
        expected_duration_s: Expected total duration
        
    Returns:
        scipy Bounds object
    """
    p_min = physics_config.p_min
    
    # Upper bound: maximum power for expected duration
    # Use slightly shorter duration to allow some margin
    p_max = pdc.max_power(max(60, expected_duration_s * 0.8))
    
    # Allow higher power for short bursts
    p_max = min(p_max * 1.2, pdc.max_power(60))
    
    lower = np.full(n_segments, p_min)
    upper = np.full(n_segments, p_max)
    
    return Bounds(lower, upper)


def estimate_initial_duration(
    course: Course,
    ftp: float,
    weather: WeatherData,
    physics_config: PhysicsConfig
) -> float:
    """
    Estimate initial duration using constant power simulation.
    
    Args:
        course: Course to estimate
        ftp: Functional threshold power
        weather: Weather conditions
        physics_config: Physics parameters
        
    Returns:
        Estimated duration in seconds
    """
    from physics import simulate_constant_power
    
    # Use 95% FTP for initial estimate
    power = ftp * 0.95
    result = simulate_constant_power(course, power, weather, physics_config)
    return result.total_time_s


def optimize_pacing(
    course: Course,
    weather: WeatherData,
    physics_config: PhysicsConfig,
    pdc: PDCModel,
    opt_config: Optional[OptimizationConfig] = None
) -> OptimizationResult:
    """
    Optimize pacing for a course.
    
    Uses L-BFGS-B to minimize time while respecting PDC constraints.
    
    Args:
        course: Course to optimize
        weather: Weather conditions  
        physics_config: Physics parameters
        pdc: Power duration curve
        opt_config: Optimization configuration
        
    Returns:
        OptimizationResult with optimized plan
    """
    if opt_config is None:
        opt_config = OptimizationConfig()
    
    n_segments = len(course.points)
    ftp = pdc.ftp_estimate()
    
    # Step 1: Estimate initial duration
    initial_duration = estimate_initial_duration(
        course, ftp, weather, physics_config
    )
    
    if opt_config.verbose:
        print(f"Initial duration estimate: {initial_duration:.1f}s ({initial_duration/60:.1f}min)")
    
    # Step 2: Create baseline (constant power) simulation
    baseline_power = ftp * opt_config.initial_power_frac
    from physics import simulate_constant_power
    baseline_sim = simulate_constant_power(course, baseline_power, weather, physics_config)
    
    if opt_config.verbose:
        print(f"Baseline time at {baseline_power:.0f}W: {baseline_sim.total_time_s:.1f}s")
    
    # Step 3: Initialize power array
    # Start with constant power, slightly reduced to allow headroom
    initial_power = np.full(n_segments, baseline_power)
    
    # Apply grade-based initial adjustments
    grades = course.get_grades()
    
    # Increase power on climbs, decrease on descents
    grade_adjustment = np.clip(grades / 10.0, -0.2, 0.3)  # ±20-30% based on grade
    initial_power = initial_power * (1 + grade_adjustment * 0.5)
    
    # Ensure within reasonable bounds
    initial_power = np.clip(initial_power, physics_config.p_min, pdc.max_power(60))
    
    # Step 4: Create objective function and bounds
    objective = create_objective_function(
        course, weather, physics_config, pdc, opt_config
    )
    
    bounds = create_bounds(
        n_segments, pdc, physics_config, initial_duration
    )
    
    # Step 5: Run optimization
    if opt_config.verbose:
        print("Starting optimization...")
    
    options = {
        'maxiter': opt_config.max_iterations,
        'ftol': opt_config.tolerance,
    }
    
    result = minimize(
        objective,
        initial_power,
        method='L-BFGS-B',
        bounds=bounds,
        options=options
    )
    
    optimized_power = result.x
    
    # Step 6: Apply smoothing to reduce chatter
    # Use a simple rolling average to smooth the optimized power
    window = min(5, n_segments // 10)
    if window >= 3:
        kernel = np.ones(window) / window
        padded = np.pad(optimized_power, (window//2, window//2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')[:n_segments]
        optimized_power = smoothed
    
    # Ensure bounds are still respected
    optimized_power = np.clip(optimized_power, bounds.lb, bounds.ub)
    
    # Step 7: Simulate with final optimized plan
    final_sim = simulate_course(course, optimized_power, weather, physics_config)
    
    # Step 8: Check for violations
    violations = check_feasibility(
        optimized_power, 
        final_sim.get_times(), 
        pdc
    )
    
    # Step 9: Compute improvement
    time_saved = baseline_sim.total_time_s - final_sim.total_time_s
    time_saved_pct = (time_saved / baseline_sim.total_time_s) * 100
    
    if opt_config.verbose:
        print(f"Optimized time: {final_sim.total_time_s:.1f}s")
        print(f"Time saved: {time_saved:.1f}s ({time_saved_pct:.1f}%)")
        if violations:
            print(f"Violations: {len(violations)}")
    
    return OptimizationResult(
        power_plan=optimized_power,
        simulation=final_sim,
        baseline_simulation=baseline_sim,
        time_saved_s=time_saved,
        time_saved_pct=time_saved_pct,
        iterations=result.nit,
        converged=result.success,
        violations=violations
    )


def iterative_optimize(
    course: Course,
    weather: WeatherData,
    physics_config: PhysicsConfig,
    pdc: PDCModel,
    opt_config: Optional[OptimizationConfig] = None,
    max_outer_iterations: int = 3
) -> OptimizationResult:
    """
    Iteratively optimize, updating duration estimates.
    
    This helps because PDC constraints depend on duration, which
    is unknown a priori.
    
    Args:
        course: Course to optimize
        weather: Weather conditions
        physics_config: Physics parameters
        pdc: Power duration curve
        opt_config: Optimization configuration
        max_outer_iterations: Maximum outer loop iterations
        
    Returns:
        OptimizationResult with optimized plan
    """
    if opt_config is None:
        opt_config = OptimizationConfig()
    
    best_result = None
    
    for i in range(max_outer_iterations):
        if opt_config.verbose:
            print(f"\n=== Outer iteration {i+1}/{max_outer_iterations} ===")
        
        result = optimize_pacing(
            course, weather, physics_config, pdc, opt_config
        )
        
        if best_result is None or result.total_time_s < best_result.total_time_s:
            best_result = result
        
        # Check for convergence
        if best_result is not None:
            time_diff = abs(result.total_time_s - best_result.total_time_s)
            if time_diff < 1.0:  # Less than 1 second change
                if opt_config.verbose:
                    print("Converged (time change < 1s)")
                break
    
    return best_result


def quick_optimize(
    course: Course,
    weather: WeatherData,
    physics_config: PhysicsConfig,
    pdc: PDCModel
) -> OptimizationResult:
    """
    Quick optimization with default settings.
    
    Args:
        course: Course to optimize
        weather: Weather conditions
        physics_config: Physics parameters
        pdc: Power duration curve
        
    Returns:
        OptimizationResult
    """
    config = OptimizationConfig(
        regularization=0.1,
        max_iterations=200,
        verbose=False
    )
    
    return optimize_pacing(course, weather, physics_config, pdc, config)


def compare_strategies(
    course: Course,
    weather: WeatherData,
    physics_config: PhysicsConfig,
    pdc: PDCModel,
    powers_to_test: Optional[list] = None
) -> dict:
    """
    Compare different pacing strategies.
    
    Args:
        course: Course to analyze
        weather: Weather conditions
        physics_config: Physics parameters
        pdc: Power duration curve
        powers_to_test: List of constant powers to test
        
    Returns:
        Dictionary with comparison results
    """
    from physics import simulate_constant_power
    
    ftp = pdc.ftp_estimate()
    
    if powers_to_test is None:
        powers_to_test = [
            ftp * 0.85,
            ftp * 0.90,
            ftp * 0.95,
            ftp * 1.00,
            ftp * 1.05,
        ]
    
    results = {
        'constant_power': {},
        'optimized': None
    }
    
    # Test constant power strategies
    for power in powers_to_test:
        sim = simulate_constant_power(course, power, weather, physics_config)
        results['constant_power'][f"{power:.0f}W"] = {
            'power': power,
            'time_s': sim.total_time_s,
            'avg_speed_kmh': sim.avg_speed_ms * 3.6
        }
    
    # Run optimization
    opt_result = quick_optimize(course, weather, physics_config, pdc)
    results['optimized'] = {
        'avg_power': opt_result.simulation.avg_power_w,
        'time_s': opt_result.total_time_s,
        'time_saved_s': opt_result.time_saved_s,
        'time_saved_pct': opt_result.time_saved_pct,
        'violations': len(opt_result.violations)
    }
    
    return results
