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
    regularization: float = 0.5       # Penalty weight for power variability (increased for smoother plans)
    pdc_penalty_weight: float = 10000.0  # Penalty weight for PDC violations (10x higher)
    max_iterations: int = 30          # Maximum solver iterations (most gains come very early)
    tolerance: float = 1.0            # Convergence tolerance (1 second is plenty)
    initial_power_frac: float = 0.95  # Initial power as fraction of FTP
    verbose: bool = False
    progress_callback: Optional[Callable[[ProgressUpdate], None]] = None
    early_stop_threshold: float = 3.0  # Stop if improvement < 3 seconds over recent evals
    max_time_s: float = 30.0          # Maximum optimization time in seconds


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


class EarlyStopException(Exception):
    """Raised to terminate optimization early."""
    pass


class ObjectiveTracker:
    """Tracks objective function evaluations and enforces termination."""
    
    def __init__(self, opt_config: OptimizationConfig, baseline_time: float):
        self.opt_config = opt_config
        self.baseline_time = baseline_time
        self.iteration = 0
        self.best_time = float('inf')
        self.best_x = None
        self.start_time = time.time()
        self.last_report_time = 0
        self.last_improvement_time = time.time()
        self.eval_count = 0
        self.stop_reason = ""
    
    def check_and_report(self, power_array: np.ndarray, total_time: float, objective_value: float):
        """
        Check termination conditions and report progress.
        
        Raises EarlyStopException if we should terminate.
        """
        self.eval_count += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Check for significant improvement (> 0.5 seconds)
        improved = total_time < self.best_time - 0.5
        
        if improved:
            self.best_time = total_time
            self.best_x = power_array.copy()
            self.last_improvement_time = current_time
        elif self.best_x is None:
            # Store first result even if not "improved"
            self.best_x = power_array.copy()
            self.best_time = total_time
        
        # Report every 2 seconds or on improvement
        time_since_report = current_time - self.last_report_time
        if self.opt_config.progress_callback and (time_since_report > 2.0 or improved):
            self.last_report_time = current_time
            self.iteration += 1
            
            improvement = self.baseline_time - self.best_time
            
            update = ProgressUpdate(
                iteration=self.iteration,
                total_time_s=self.best_time,
                objective_value=objective_value,
                improvement_s=improvement,
                elapsed_s=elapsed,
                message=f"Best: {self.best_time:.1f}s (saving {improvement:.1f}s) - {elapsed:.0f}s elapsed"
            )
            self.opt_config.progress_callback(update)
        
        # === TERMINATION CHECKS ===
        time_since_improvement = current_time - self.last_improvement_time
        
        # Stop if exceeded max time
        if elapsed > self.opt_config.max_time_s:
            self.stop_reason = f"Max time ({self.opt_config.max_time_s:.0f}s) reached"
            raise EarlyStopException(self.stop_reason)
        
        # Stop if no improvement for 10 seconds (after initial settling)
        if time_since_improvement > 10.0 and elapsed > 5.0:
            self.stop_reason = f"No improvement for {time_since_improvement:.0f}s"
            raise EarlyStopException(self.stop_reason)


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
    
    This function also checks termination conditions via the tracker,
    which will raise EarlyStopException when it's time to stop.
    
    Args:
        course: Course to optimize
        weather: Weather conditions
        physics_config: Physics parameters
        pdc: Power duration curve
        opt_config: Optimization configuration
        tracker: Progress tracker (also handles termination)
        
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
        
        # Check termination and report progress (may raise EarlyStopException)
        if tracker:
            tracker.check_and_report(power_array, total_time, obj_value)
        
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
    
    Individual segments can spike above race-duration power (that's the point of
    variable pacing), but the PDC penalty will ensure NP stays within limits.
    
    Args:
        n_segments: Number of segments
        pdc: Power duration curve
        physics_config: Physics parameters (for p_min)
        expected_duration_s: Expected total duration
        
    Returns:
        scipy Bounds object
    """
    p_min = physics_config.p_min
    
    # Upper bound: allow spikes up to ~5-minute power for short climbs
    # The PDC penalty ensures overall NP stays within race-duration limits
    p_max = pdc.max_power(300)  # 5-minute power as upper bound for any segment
    
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
    
    # Step 2: Initial estimate for optimization setup
    log("Step 2/7: Setting up initial power estimate...")
    from physics import simulate_constant_power
    initial_power_level = pdc.max_power(initial_duration) * 0.95
    log(f"  Starting optimization from {initial_power_level:.0f}W")
    
    # Step 3: Initialize power array
    log("Step 3/7: Initializing power array with grade adjustments...")
    initial_power = np.full(n_segments, initial_power_level)
    
    # Apply grade-based initial adjustments
    grades = course.get_grades()
    grade_adjustment = np.clip(grades / 10.0, -0.2, 0.3)
    initial_power = initial_power * (1 + grade_adjustment * 0.5)
    initial_power = np.clip(initial_power, physics_config.p_min, pdc.max_power(60))
    log(f"  Power range: {initial_power.min():.0f}W - {initial_power.max():.0f}W")
    
    # Step 4: Create objective function and bounds
    log("Step 4/7: Setting up optimizer...")
    
    # Create tracker for progress updates (use initial estimate as reference)
    initial_sim = simulate_constant_power(course, initial_power_level, weather, physics_config)
    tracker = ObjectiveTracker(opt_config, initial_sim.total_time_s)
    
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
    log(f"  Max iterations: {opt_config.max_iterations}, tolerance: {opt_config.tolerance}s")
    log(f"  Max time: {opt_config.max_time_s:.0f}s, early stop if no improvement for 10s")
    
    start_time = time.time()
    
    options = {
        'maxiter': opt_config.max_iterations,
        'ftol': 1e-6,  # Use tight tolerance; we control stopping via time/improvement
    }
    
    n_iterations = 0
    try:
        result = minimize(
            objective,
            initial_power,
            method='L-BFGS-B',
            bounds=bounds,
            options=options
        )
        optimized_power = result.x
        converged = result.success
        n_iterations = result.nit
        log(f"  Optimizer converged naturally after {n_iterations} iterations")
    except EarlyStopException as e:
        # Early stopping triggered - use best solution found
        optimized_power = tracker.best_x if tracker.best_x is not None else initial_power
        converged = True  # We stopped intentionally
        n_iterations = tracker.eval_count
        log(f"  Early stop: {e}")
    
    opt_elapsed = time.time() - start_time
    log(f"  Completed in {opt_elapsed:.1f}s after {tracker.eval_count} evaluations")
    
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
    
    # Calculate baseline: what if you held the optimized NP as constant power?
    # This is the fair comparison - same physiological cost, but constant vs variable pacing
    optimized_np = final_sim.normalized_power_w
    baseline_sim = simulate_constant_power(course, optimized_np, weather, physics_config)
    
    time_saved = baseline_sim.total_time_s - final_sim.total_time_s
    time_saved_pct = (time_saved / baseline_sim.total_time_s) * 100 if baseline_sim.total_time_s > 0 else 0
    
    log(f"")
    log(f"=== OPTIMIZATION COMPLETE ===")
    log(f"  Optimized time: {final_sim.total_time_s:.1f}s ({final_sim.total_time_s/60:.1f} min)")
    log(f"  Avg power: {final_sim.avg_power_w:.0f}W, NP: {optimized_np:.0f}W")
    log(f"  Baseline (constant {optimized_np:.0f}W): {baseline_sim.total_time_s:.1f}s")
    log(f"  Time saved by variable pacing: {time_saved:.1f}s ({time_saved_pct:.1f}%)")
    log(f"  PDC violations: {len(violations)}")
    
    return OptimizationResult(
        power_plan=optimized_power,
        simulation=final_sim,
        baseline_simulation=baseline_sim,
        time_saved_s=time_saved,
        time_saved_pct=time_saved_pct,
        iterations=n_iterations,
        converged=converged,
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


def compare_bikes(
    course: Course,
    weather: WeatherData,
    tt_physics_config: PhysicsConfig,
    road_physics_config: PhysicsConfig,
    pdc: PDCModel,
    opt_config: Optional[OptimizationConfig] = None,
    log_callback: Optional[Callable[[str], None]] = None
) -> dict:
    """
    Compare TT bike vs Road bike for a course.
    
    Runs optimization for both bike configurations and returns comparison.
    
    Args:
        course: Course to optimize
        weather: Weather conditions
        tt_physics_config: Physics config for TT bike
        road_physics_config: Physics config for Road bike
        pdc: Power duration curve
        opt_config: Optimization configuration
        log_callback: Optional callback for log messages
        
    Returns:
        Dictionary with comparison results
    """
    def log(msg: str):
        if log_callback:
            log_callback(msg)
    
    log("=== BIKE COMPARISON ===")
    log("")
    log("Optimizing for TT bike...")
    tt_result = optimize_pacing(
        course, weather, tt_physics_config, pdc, opt_config, log_callback=None
    )
    log(f"  TT bike time: {tt_result.total_time_s:.1f}s ({tt_result.total_time_s/60:.1f} min)")
    
    log("")
    log("Optimizing for Road bike...")
    road_result = optimize_pacing(
        course, weather, road_physics_config, pdc, opt_config, log_callback=None
    )
    log(f"  Road bike time: {road_result.total_time_s:.1f}s ({road_result.total_time_s/60:.1f} min)")
    
    # Determine recommendation
    time_delta = tt_result.total_time_s - road_result.total_time_s
    
    if time_delta < -5:  # TT is more than 5 seconds faster
        recommended = 'tt'
        reason = f"TT bike is {abs(time_delta):.1f}s faster"
    elif time_delta > 5:  # Road is more than 5 seconds faster
        recommended = 'road'
        reason = f"Road bike is {abs(time_delta):.1f}s faster"
    else:
        # Within 5 seconds - essentially equal
        recommended = 'either'
        reason = f"Both bikes within {abs(time_delta):.1f}s - choose based on comfort"
    
    log("")
    log(f"Recommendation: {recommended.upper()} - {reason}")
    
    return {
        'tt_result': tt_result,
        'road_result': road_result,
        'tt_time': tt_result.total_time_s,
        'road_time': road_result.total_time_s,
        'time_delta': time_delta,
        'recommended': recommended,
        'reason': reason
    }
