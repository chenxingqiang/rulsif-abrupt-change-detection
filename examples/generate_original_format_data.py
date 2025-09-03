#!/usr/bin/env python3
"""
Generate Synthetic Driving Data in Original Format

This script generates data that matches the exact format used in the original
Aggressive_Drive_detection.ipynb notebook.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta


def generate_original_format_data(driver_id, n_samples=10000):
    """
    Generate data in the exact format used by the original notebook.
    
    Columns: ['Car_ID', 'Time', 'Car_Orientation', 'Pitch_Rate', 'Roll_Rate', 
              'Acceleration', 'Velocity', 'Steering_Wheel_Angle', 'Yaw_Rate']
    """
    # Car ID (driver number)
    car_id = [int(driver_id)] * n_samples
    
    # Time - starts from 0, increments by 1, with potential resets to 0
    time = np.arange(n_samples)
    
    # Add time resets to simulate multiple driving sessions
    # This matches the original notebook's behavior of finding Time==0 points
    reset_points = []
    if n_samples > 5000:
        # Add 2-3 time resets for longer sequences
        num_resets = np.random.randint(2, 4)
        reset_positions = np.sort(np.random.choice(range(1000, n_samples-1000), num_resets, replace=False))
        reset_points = [0] + list(reset_positions) + [n_samples]
        
        # Reset time at these points
        for i in range(1, len(reset_points)):
            start_idx = reset_points[i-1]
            end_idx = reset_points[i]
            time[start_idx:end_idx] = np.arange(end_idx - start_idx)
    
    # Car_Orientation (degrees) - vehicle heading direction
    # Normal driving: small variations around a base direction
    base_orientation = np.random.uniform(0, 360)
    orientation = base_orientation + np.random.normal(0, 5, n_samples)
    orientation = np.mod(orientation, 360)  # Keep within 0-360 degrees
    
    # Pitch_Rate (degrees/second) - vehicle pitch rotation rate
    # Related to acceleration/deceleration
    pitch_rate = np.random.normal(0, 2, n_samples)
    # Increase during acceleration/deceleration events
    acceleration_events = np.random.random(n_samples) < 0.01  # 1% events
    pitch_rate[acceleration_events] += np.random.uniform(5, 15, np.sum(acceleration_events))
    pitch_rate = np.clip(pitch_rate, -20, 20)
    
    # Roll_Rate (degrees/second) - vehicle roll rotation rate
    # Related to steering and turning
    roll_rate = np.random.normal(0, 1.5, n_samples)
    # Increase during turning events
    turning_events = np.random.random(n_samples) < 0.01  # 1% events
    roll_rate[turning_events] += np.random.uniform(3, 10, np.sum(turning_events))
    roll_rate = np.clip(roll_rate, -15, 15)
    
    # Acceleration (m/s²) - vehicle acceleration
    # Base acceleration with realistic variations
    base_accel = np.random.normal(0, 0.5, n_samples)
    
    # Add acceleration patterns
    # Gradual acceleration/deceleration
    for i in range(0, n_samples, 500):  # Every 500 samples
        if np.random.random() < 0.3:  # 30% chance
            # Acceleration phase
            duration = np.random.randint(50, 150)
            end_idx = min(i + duration, n_samples)
            base_accel[i:end_idx] += np.linspace(0, np.random.uniform(2, 4), end_idx - i)
        elif np.random.random() < 0.3:  # 30% chance
            # Deceleration phase
            duration = np.random.randint(50, 150)
            end_idx = min(i + duration, n_samples)
            base_accel[i:end_idx] -= np.linspace(0, np.random.uniform(2, 4), end_idx - i)
    
    # Add sudden acceleration events (aggressive driving)
    sudden_accel_events = np.random.random(n_samples) < 0.005  # 0.5% events
    base_accel[sudden_accel_events] += np.random.uniform(5, 12, np.sum(sudden_accel_events))
    
    # Add sudden braking events
    sudden_brake_events = np.random.random(n_samples) < 0.005  # 0.5% events
    base_accel[sudden_brake_events] -= np.random.uniform(6, 15, np.sum(sudden_brake_events))
    
    acceleration = np.clip(base_accel, -15, 15)
    
    # Velocity (km/h) - vehicle speed
    # Integrate acceleration to get velocity
    velocity = np.zeros(n_samples)
    current_velocity = np.random.uniform(40, 80)  # Start with reasonable speed
    
    for i in range(n_samples):
        # Update velocity based on acceleration
        current_velocity += acceleration[i] * 3.6  # Convert m/s² to km/h
        current_velocity = np.clip(current_velocity, 0, 130)  # Realistic speed range
        velocity[i] = current_velocity
        
        # Add some random variation
        velocity[i] += np.random.normal(0, 1)
        velocity[i] = np.clip(velocity[i], 0, 130)
    
    # Steering_Wheel_Angle (degrees) - steering wheel rotation
    # Base steering with small variations
    base_steering = np.random.normal(0, 3, n_samples)
    
    # Add steering patterns
    # Gradual turns
    for i in range(0, n_samples, 300):  # Every 300 samples
        if np.random.random() < 0.4:  # 40% chance
            duration = np.random.randint(30, 100)
            end_idx = min(i + duration, n_samples)
            turn_angle = np.random.uniform(-20, 20)
            base_steering[i:end_idx] += np.linspace(0, turn_angle, end_idx - i)
    
    # Add sharp steering events (aggressive driving)
    sharp_steering_events = np.random.random(n_samples) < 0.003  # 0.3% events
    base_steering[sharp_steering_events] += np.random.uniform(-30, 30, np.sum(sharp_steering_events))
    
    steering_wheel_angle = np.clip(base_steering, -45, 45)
    
    # Yaw_Rate (degrees/second) - vehicle yaw rotation rate
    # Related to steering and velocity
    yaw_rate = (steering_wheel_angle * velocity * 0.001 + 
                np.random.normal(0, 2, n_samples))
    
    # Increase during turning events
    yaw_rate[turning_events] += np.random.uniform(5, 15, np.sum(turning_events))
    yaw_rate = np.clip(yaw_rate, -25, 25)
    
    # Create DataFrame with exact column names from original notebook
    df = pd.DataFrame({
        'Car_ID': car_id,
        'Time': time,
        'Car_Orientation': orientation,
        'Pitch_Rate': pitch_rate,
        'Roll_Rate': roll_rate,
        'Acceleration': acceleration,
        'Velocity': velocity,
        'Steering_Wheel_Angle': steering_wheel_angle,
        'Yaw_Rate': yaw_rate
    })
    
    return df, reset_points


def save_original_format_data(df, reset_points, output_path, driver_id):
    """Save data in the original format."""
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save main file
    main_file = os.path.join(output_path, f'{driver_id}.csv')
    df.to_csv(main_file, index=False)
    
    # Save segmented files (matching original notebook behavior)
    if len(reset_points) > 2:
        for i in range(1, len(reset_points)):
            start_idx = reset_points[i-1]
            end_idx = reset_points[i]
            
            if end_idx - start_idx > 151:  # Only save sequences longer than 151
                segment_data = df.iloc[start_idx:end_idx].copy()
                
                # Reset time to start from 0 for each segment
                segment_data['Time'] = np.arange(len(segment_data))
                
                # Save with appropriate naming convention
                segment_file = os.path.join(output_path, f'OrderRight_{driver_id}_{i:04d}.csv')
                segment_data.to_csv(segment_file, index=False)
                
                print(f"  Saved segment {i}: {len(segment_data)} samples")
    
    print(f"Saved {driver_id}.csv with {len(df)} samples and {len(reset_points)-1} segments")


def main():
    """Generate data for all drivers in original format."""
    # Driver IDs from the original notebook
    driver_ids = ['10', '56', '64', '72', '84',
                  '11', '56_0', '64_0', '73', '85',
                  '12', '57', '65', '74', '86',
                  '40', '60', '66', '77', '87',
                  '45', '60_0', '67', '78', '89',
                  '50', '61', '68', '80',
                  '51', '62', '69', '81',
                  '52', '63', '70', '82',
                  '55', '63_2', '71', '83']
    
    # Configuration
    output_dir = './original_format_data/'
    n_samples = 8000  # Similar to original data length
    
    print(f"Generating original format data for {len(driver_ids)} drivers...")
    print(f"Each driver will have {n_samples:,} samples")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Generate data for each driver
    all_reset_points = []
    for i, driver_id in enumerate(driver_ids, 1):
        print(f"Processing driver {driver_id} ({i}/{len(driver_ids)})...")
        
        # Generate data
        df, reset_points = generate_original_format_data(driver_id, n_samples)
        
        # Save data
        save_original_format_data(df, reset_points, output_dir, driver_id)
        
        # Collect reset points for summary
        all_reset_points.extend([(driver_id, len(reset_points)-1)])
    
    # Generate summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Total drivers processed: {len(driver_ids)}")
    print(f"Total samples generated: {len(driver_ids) * n_samples:,}")
    
    # Segment statistics
    total_segments = sum(segments for _, segments in all_reset_points)
    print(f"Total segments created: {total_segments}")
    print(f"Average segments per driver: {total_segments / len(driver_ids):.1f}")
    
    print(f"\nData saved to: {os.path.abspath(output_dir)}")
    print("\nThis data format matches the original Aggressive_Drive_detection.ipynb notebook!")
    print("You can now use these CSV files with your original notebook code.")


if __name__ == "__main__":
    main()
