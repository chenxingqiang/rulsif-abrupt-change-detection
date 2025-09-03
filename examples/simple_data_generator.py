#!/usr/bin/env python3
"""
Simple Synthetic Driving Data Generator
"""

import numpy as np
import pandas as pd
import os


def generate_driving_data(driver_id, n_samples=5000):
    """Generate synthetic driving data for a driver."""
    # Time series
    time = np.arange(n_samples)
    
    # Base speed with variations
    base_speed = 60 + np.random.normal(0, 10)
    speed = base_speed + 10 * np.sin(time * 0.01) + np.random.normal(0, 5, n_samples)
    speed = np.clip(speed, 20, 120)
    
    # Acceleration
    acceleration = np.gradient(speed, 1)
    acceleration = np.clip(acceleration, -15, 15)
    
    # RPM (correlated with speed)
    rpm = speed * 15 + np.random.normal(0, 200, n_samples)
    rpm = np.clip(rpm, 800, 6000)
    
    # Fuel consumption
    fuel = speed * 0.1 + np.abs(acceleration) * 0.3 + np.random.normal(0, 0.5, n_samples)
    fuel = np.clip(fuel, 0, 15)
    
    # Engine temperature
    temp = 90 + (rpm - 2000) * 0.005 + np.random.normal(0, 2, n_samples)
    temp = np.clip(temp, 70, 110)
    
    # Brake pressure
    brake = np.where(acceleration < -3, np.abs(acceleration) * 8, np.random.normal(0, 3, n_samples))
    brake = np.clip(brake, 0, 100)
    
    # Steering angle
    steering = np.random.normal(0, 8, n_samples)
    steering = np.clip(steering, -25, 25)
    
    # Lateral acceleration
    lateral = steering * speed * 0.001 + np.random.normal(0, 0.3, n_samples)
    lateral = np.clip(lateral, -3, 3)
    
    # Throttle
    throttle = np.where(acceleration > 0, np.clip(acceleration * 8 + 25, 0, 100), 
                        np.random.normal(25, 8, n_samples))
    throttle = np.clip(throttle, 0, 100)
    
    # Gear
    gear = np.ones(n_samples)
    gear[speed < 25] = 1
    gear[(speed >= 25) & (speed < 45)] = 2
    gear[(speed >= 45) & (speed < 65)] = 3
    gear[(speed >= 65) & (speed < 85)] = 4
    gear[(speed >= 85) & (speed < 105)] = 5
    gear[speed >= 105] = 6
    
    # Add aggressive events randomly
    aggressive_mask = np.random.random(n_samples) < 0.02  # 2% aggressive events
    
    # Sudden acceleration events
    accel_events = np.random.random(n_samples) < 0.005
    speed[accel_events] += np.random.uniform(25, 40, np.sum(accel_events))
    
    # Hard braking events
    brake_events = np.random.random(n_samples) < 0.005
    speed[brake_events] -= np.random.uniform(30, 45, np.sum(brake_events))
    brake[brake_events] = 100
    
    # Sharp turning events
    turn_events = np.random.random(n_samples) < 0.005
    steering[turn_events] += np.random.uniform(20, 35, np.sum(turn_events)) * np.random.choice([-1, 1], np.sum(turn_events))
    
    # Clip final values
    speed = np.clip(speed, 0, 130)
    acceleration = np.gradient(speed, 1)
    acceleration = np.clip(acceleration, -20, 20)
    
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='S'),
        'speed': speed,
        'acceleration': acceleration,
        'rpm': rpm,
        'fuel_consumption': fuel,
        'engine_temperature': temp,
        'brake_pressure': brake,
        'steering_angle': steering,
        'lateral_acceleration': lateral,
        'throttle_position': throttle,
        'gear': gear,
        'driver_id': driver_id,
        'aggressive_event': aggressive_mask
    })


def main():
    """Generate data for all drivers."""
    # Driver IDs from your notebook
    driver_ids = ['10', '56', '64', '72', '84',
                  '11', '56_0', '64_0', '73', '85',
                  '12', '57', '65', '74', '86',
                  '40', '60', '66', '77', '87',
                  '45', '60_0', '67', '78', '89',
                  '50', '61', '68', '80',
                  '51', '62', '69', '81',
                  '52', '63', '70', '82',
                  '55', '63_2', '71', '83']
    
    # Create output directory
    output_dir = './synthetic_driving_data/'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating data for {len(driver_ids)} drivers...")
    
    # Generate data for each driver
    for i, driver_id in enumerate(driver_ids, 1):
        print(f"Processing driver {driver_id} ({i}/{len(driver_ids)})...")
        
        # Generate data
        df = generate_driving_data(driver_id, n_samples=5000)
        
        # Save to CSV
        output_file = os.path.join(output_dir, f'{driver_id}.csv')
        df.to_csv(output_file, index=False)
        
        # Count aggressive events
        aggressive_count = df['aggressive_event'].sum()
        print(f"  Generated {len(df)} samples with {aggressive_count} aggressive events")
    
    print(f"\nData generation complete! Files saved to: {os.path.abspath(output_dir)}")
    print("You can now use these CSV files with your aggressive driving detection notebook.")


if __name__ == "__main__":
    main()
