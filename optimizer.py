"""
Optimizer Module - Objective function, regularization, L-BFGS-B solver.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, List
import numpy as np
from scipy.optimize import minimize, Bounds
import time

from gpx_io import Course
from weather import WeatherData
from physics import PhysicsConfig, simulate_course, SimulationResult
from pdc import PDCModel, compute_pdc_penalty, check_feasibility


@dataclass
class ProgressUpdate:
    """Progress update during optimization."""
    iteration: int
    total_time_s: float
    objective_value: float
    improvement_s: float
    elapsed_s: float
    message: str


@dataclass
class OptimizationConfig:
    """Configuration for the optimizer."""
    regularization: float = 0.1       # Penalty weight for power variability
    pdc_penalty_weight: float = 1000.0  # Penalty weight for PDC violations
    max_iterations: int = 500         # Maximum solver iterations
    tolerance: float = 1e-6           # Convergence tolerance
    initial_power_frac: float = 0.95  # Initial power as fraction of FTP
    verbose: bool = False
    progress_callback: Optional[Callable[[ProgressUpdate], None]] = None


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


class ObjectiveTracker:
    """Tracks objective function evaluations for progress reporting."""
    
    def __init__(self, opt_config: OptimizationConfig, baseline_time: float):
        self.opt_config = opt_config
        self.baseline_time = baseline_time
        self.iteration = 0
        self.best_time = float('inf')
        self.start_time = time.time()
        self.last_report_time = 0
        self.eval_count = 0
    
    def report(self, total_time: float, objective_value: float):
        """Report progress if callback is set."""
        self.eval_count += 1
        
        # Only report every ~20 evaluations or if significant improvement
        current_time = time.time()
        time_since_report = current_time - self.last_report_time
        improved = total_time < self.best_time - 0.5
        
        if improved:
            self.best_time = total_time
        
        # Report every 2 seconds or on improvement
        if self.opt_config.progress_callback and (time_since_report > 2.0 or improved):
            self.last_report_time = current_time
            self.iteration += 1
            
            improvement = self.baseline_time - self.best_time
            elapsed = current_time - self.start_time
            
            update = ProgressUpdate(
                iteration=self.iteration,
                total_time_s=self.best_time,
                objective_value=objective_value,
                improvement_s=improvement,
                elapsed_s=elapsed,
                message=f"Eval #{self.eval_count}: Best time {self.best_time:.1f}s (saving {improvement:.1f}s)"
            )
            self.opt_config.progress_callback(update)


def create_objective_function(
    course: Course,
    weather: WeatherData,
    physics_config: PhysicsConfig,
    pdc: PDCModel,
    opt_config: OptimizationConfig,
    tracker: Optional[ObjectiveTracker] = None
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
        tracker: Optional progress tracker
        
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
        
        obj_value = total_time + smoothness_penalty + pdc_penalty
        
        # Report progress
        if tracker:
            tracker.report(total_time, obj_value)
        
        return obj_value
    
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
    opt_config: Optional[OptimizationConfig] = None,
    log_callback: Optional[Callable[[str], None]] = None
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
        log_callback: Optional callback for log messages
        
    Returns:
        OptimizationResult with optimized plan
    """
    if opt_config is None:
        opt_config = OptimizationConfig()
    
    def log(msg: str):
        if log_callback:
            log_callback(msg)
        if opt_config.verbose:
            print(msg)
    
    n_segments = len(course.points)
    ftp = pdc.ftp_estimate()
    
    log(f"Starting optimization for {course.total_distance_m/1000:.1f}km course")
    log(f"Course has {n_segments} segments at 50m resolution")
    
    # Step 1: Estimate initial duration
    log("Step 1/7: Estimating initial duration...")
    initial_duration = estimate_initial_duration(
        course, ftp, weather, physics_config
    )
    log(f"  Initial estimate: {initial_duration:.1f}s ({initial_duration/60:.1f} min)")
    
    # Step 2: Create baseline (constant power) simulation
    log("Step 2/7: Running baseline simulation...")
    baseline_power = ftp * opt_config.initial_power_frac
    from physics import simulate_constant_power
    baseline_sim = simulate_constant_power(course, baseline_power, weather, physics_config)
    log(f"  Baseline at {baseline_power:.0f}W: {baseline_sim.total_time_s:.1f}s ({baseline_sim.total_time_s/60:.1f} min)")
    
    # Step 3: Initialize power array
    log("Step 3/7: Initializing power array with grade adjustments...")
    initial_power = np.full(n_segments, baseline_power)
    
    # Apply grade-based initial adjustments
    grades = course.get_grades()
    grade_adjustment = np.clip(grades / 10.0, -0.2, 0.3)
    initial_power = initial_power * (1 + grade_adjustment * 0.5)
    initial_power = np.clip(initial_power, physics_config.p_min, pdc.max_power(60))
    log(f"  Power range: {initial_power.min():.0f}W - {initial_power.max():.0f}W")
    
    # Step 4: Create objective function and bounds
    log("Step 4/7: Setting up optimizer...")
    
    # Create tracker for progress updates
    tracker = ObjectiveTracker(opt_config, baseline_sim.total_time_s)
    
    objective = create_objective_function(
        course, weather, physics_config, pdc, opt_config, tracker
    )
    
    bounds = create_bounds(
        n_segments, pdc, physics_config, initial_duration
    )
    log(f"  Power bounds: {bounds.lb[0]:.0f}W - {bounds.ub[0]:.0f}W")
    log(f"  Max iterations: {opt_config.max_iterations}")
    
    # Step 5: Run optimization
    log("Step 5/7: Running L-BFGS-B optimization...")
    log("  (This may take a minute for long courses)")
    
    start_time = time.time()
    
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
    
    opt_elapsed = time.time() - start_time
    log(f"  Optimization completed in {opt_elapsed:.1f}s")
    log(f"  Converged: {result.success}, Iterations: {result.nit}")
    
    optimized_power = result.x
    
    # Step 6: Apply smoothing
    log("Step 6/7: Smoothing power profile...")
    window = min(5, n_segments // 10)
    if window >= 3:
        kernel = np.ones(window) / window
        padded = np.pad(optimized_power, (window//2, window//2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')[:n_segments]
        optimized_power = smoothed
    
    optimized_power = np.clip(optimized_power, bounds.lb, bounds.ub)
    log(f"  Final power range: {optimized_power.min():.0f}W - {optimized_power.max():.0f}W")
    
    # Step 7: Final simulation and validation
    log("Step 7/7: Validating final plan...")
    final_sim = simulate_course(course, optimized_power, weather, physics_config)
    
    violations = check_feasibility(
        optimized_power, 
        final_sim.get_times(), 
        pdc
    )
    
    time_saved = baseline_sim.total_time_s - final_sim.total_time_s
    time_saved_pct = (time_saved / baseline_sim.total_time_s) * 100
    
    log(f"")
    log(f"=== OPTIMIZATION COMPLETE ===")
    log(f"  Optimized time: {final_sim.total_time_s:.1f}s ({final_sim.total_time_s/60:.1f} min)")
    log(f"  Time saved: {time_saved:.1f}s ({time_saved_pct:.1f}%)")
    log(f"  PDC violations: {len(violations)}")
    
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
