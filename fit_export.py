"""
FIT Export Module - Generate Garmin FIT workout files from pacing segments.

Creates structured workout files that Garmin bike computers can load and execute,
providing live power targets during the race.
"""

from typing import List
from datetime import datetime


from fit_tool.fit_file_builder import FitFileBuilder
from fit_tool.profile.messages.file_id_message import FileIdMessage
from fit_tool.profile.messages.workout_message import WorkoutMessage
from fit_tool.profile.messages.workout_step_message import (
    WorkoutStepMessage,
    WorkoutStepDuration,
    WorkoutStepTarget,
    WorkoutStepDurationValueField,
    Intensity,
)
from fit_tool.profile.profile_type import FileType, Sport, Manufacturer

from segmentation import Segment


def create_fit_workout(
    segments: List[Segment],
    workout_name: str = "TT Pacing Plan",
    power_range_watts: int = 10,
) -> bytes:
    """
    Create a FIT workout file from pacing segments.
    
    Each segment becomes a workout step with:
    - Duration: Distance-based (segment length)
    - Target: Power with configurable +/- range
    
    Args:
        segments: List of Segment objects with power targets
        workout_name: Name for the workout (shown on device)
        power_range_watts: +/- watts for power target range (default 10W)
        
    Returns:
        FIT file as bytes, ready to write to file or download
    """
    if not segments:
        raise ValueError("No segments provided")
    
    builder = FitFileBuilder(auto_define=True)
    
    # File ID message - identifies this as a Workout file
    file_id = FileIdMessage()
    file_id.type = FileType.WORKOUT
    file_id.manufacturer = Manufacturer.DEVELOPMENT.value
    file_id.product = 1
    file_id.serial_number = 12345
    # fit-tool expects time in milliseconds since Unix epoch
    file_id.time_created = int(datetime.now().timestamp() * 1000)
    builder.add(file_id)
    
    # Workout message - metadata about the workout
    workout = WorkoutMessage()
    workout.workout_name = workout_name[:20]  # Garmin limits name length
    workout.sport = Sport.CYCLING
    workout.num_valid_steps = len(segments)
    builder.add(workout)
    
    # Add a workout step for each segment
    for i, segment in enumerate(segments):
        step = WorkoutStepMessage()
        step.message_index = i
        
        # Step name (shown on device when step is active)
        step_name = f"Seg {i+1}: {int(segment.avg_power)}W"
        if segment.avg_grade > 2:
            step_name += " climb"
        elif segment.avg_grade < -2:
            step_name += " desc"
        step.workout_step_name = step_name[:20]  # Limit length
        
        # Duration: Distance-based
        # Note: fit-tool has a bug with subfield selection, so we set the raw
        # encoded value directly. FIT stores distance in centimeters (scale=100).
        step.duration_type = WorkoutStepDuration.DISTANCE
        duration_field = step.get_field(WorkoutStepDurationValueField.ID)
        duration_field.set_encoded_value(0, int(segment.length_m * 100))
        
        # Target: Power with range
        step.target_type = WorkoutStepTarget.POWER
        step.target_value = 0  # 0 = use custom targets
        
        # Power targets in watts (FIT uses watts directly for custom targets)
        power_low = max(0, int(segment.avg_power) - power_range_watts)
        power_high = int(segment.avg_power) + power_range_watts
        step.custom_target_power_low = power_low
        step.custom_target_power_high = power_high
        
        # Intensity (affects display color on some devices)
        if segment.avg_power > 400:
            step.intensity = Intensity.WARMUP  # Red/hard indication
        else:
            step.intensity = Intensity.ACTIVE  # Normal
        
        builder.add(step)
    
    # Build the FIT file and convert to bytes
    fit_file = builder.build()
    return fit_file.to_bytes()


def create_fit_workout_with_cooldown(
    segments: List[Segment],
    workout_name: str = "TT Pacing Plan",
    power_range_watts: int = 10,
    warmup_duration_s: int = 0,
    cooldown_duration_s: int = 300,
) -> bytes:
    """
    Create a FIT workout file with optional warmup and cooldown steps.
    
    Args:
        segments: List of Segment objects with power targets
        workout_name: Name for the workout (shown on device)
        power_range_watts: +/- watts for power target range
        warmup_duration_s: Warmup duration in seconds (0 to skip)
        cooldown_duration_s: Cooldown duration in seconds (0 to skip)
        
    Returns:
        FIT file as bytes
    """
    if not segments:
        raise ValueError("No segments provided")
    
    builder = FitFileBuilder(auto_define=True)
    
    # File ID message
    file_id = FileIdMessage()
    file_id.type = FileType.WORKOUT
    file_id.manufacturer = Manufacturer.DEVELOPMENT.value
    file_id.product = 1
    file_id.serial_number = 12345
    # fit-tool expects time in milliseconds since Unix epoch
    file_id.time_created = int(datetime.now().timestamp() * 1000)
    builder.add(file_id)
    
    # Calculate total steps
    total_steps = len(segments)
    if warmup_duration_s > 0:
        total_steps += 1
    if cooldown_duration_s > 0:
        total_steps += 1
    
    # Workout message
    workout = WorkoutMessage()
    workout.workout_name = workout_name[:20]
    workout.sport = Sport.CYCLING
    workout.num_valid_steps = total_steps
    builder.add(workout)
    
    step_index = 0
    
    # Optional warmup step (time-based, open target)
    if warmup_duration_s > 0:
        warmup = WorkoutStepMessage()
        warmup.message_index = step_index
        warmup.workout_step_name = "Warmup"
        # Duration: Time-based (FIT stores in milliseconds, scale=1000)
        warmup.duration_type = WorkoutStepDuration.TIME
        warmup_duration_field = warmup.get_field(WorkoutStepDurationValueField.ID)
        warmup_duration_field.set_encoded_value(0, warmup_duration_s * 1000)
        warmup.target_type = WorkoutStepTarget.OPEN
        warmup.intensity = Intensity.WARMUP
        builder.add(warmup)
        step_index += 1
    
    # Main workout steps (distance-based with power targets)
    for i, segment in enumerate(segments):
        step = WorkoutStepMessage()
        step.message_index = step_index
        
        step_name = f"Seg {i+1}: {int(segment.avg_power)}W"
        if segment.avg_grade > 2:
            step_name += " climb"
        elif segment.avg_grade < -2:
            step_name += " desc"
        step.workout_step_name = step_name[:20]
        
        # Duration: Distance-based (FIT stores in centimeters, scale=100)
        step.duration_type = WorkoutStepDuration.DISTANCE
        duration_field = step.get_field(WorkoutStepDurationValueField.ID)
        duration_field.set_encoded_value(0, int(segment.length_m * 100))
        
        step.target_type = WorkoutStepTarget.POWER
        step.target_value = 0
        
        power_low = max(0, int(segment.avg_power) - power_range_watts)
        power_high = int(segment.avg_power) + power_range_watts
        step.custom_target_power_low = power_low
        step.custom_target_power_high = power_high
        
        step.intensity = Intensity.ACTIVE
        builder.add(step)
        step_index += 1
    
    # Optional cooldown step (time-based, open target)
    if cooldown_duration_s > 0:
        cooldown = WorkoutStepMessage()
        cooldown.message_index = step_index
        cooldown.workout_step_name = "Cooldown"
        # Duration: Time-based (FIT stores in milliseconds, scale=1000)
        cooldown.duration_type = WorkoutStepDuration.TIME
        cooldown_duration_field = cooldown.get_field(WorkoutStepDurationValueField.ID)
        cooldown_duration_field.set_encoded_value(0, cooldown_duration_s * 1000)
        cooldown.target_type = WorkoutStepTarget.OPEN
        cooldown.intensity = Intensity.COOLDOWN
        builder.add(cooldown)
    
    # Build the FIT file and convert to bytes
    fit_file = builder.build()
    return fit_file.to_bytes()


def get_workout_summary(segments: List[Segment]) -> str:
    """
    Generate a text summary of what the workout will contain.
    
    Args:
        segments: List of segments
        
    Returns:
        Human-readable summary string
    """
    if not segments:
        return "No segments"
    
    lines = [
        f"Workout with {len(segments)} steps:",
        ""
    ]
    
    total_distance = 0
    for i, seg in enumerate(segments):
        total_distance += seg.length_m
        grade_note = ""
        if seg.avg_grade > 2:
            grade_note = f" (+{seg.avg_grade:.1f}%)"
        elif seg.avg_grade < -2:
            grade_note = f" ({seg.avg_grade:.1f}%)"
        
        lines.append(
            f"  {i+1}. {seg.length_m/1000:.2f}km @ {int(seg.avg_power)}W{grade_note}"
        )
    
    lines.append("")
    lines.append(f"Total: {total_distance/1000:.2f}km")
    
    return "\n".join(lines)
