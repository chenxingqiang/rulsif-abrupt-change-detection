#!/usr/bin/env python3
"""
Generate Synthetic Driving Data for Anomaly Detection
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta


def generate_normal_driving_pattern(n_samples, base_speed=60, noise_level=5):
    """Generate normal driving patterns with realistic variations."""
    timestamps = [datetime.now() + timedelta(seconds=i) for i in range(n_samples)]
    
    # Speed with realistic variations
    speed = np.zeros(n_samples)
    current_speed = base_speed
    
    for i in range(n_samples):
        if i % 100 == 0:
            target_speed = base_speed + np.random.normal(0, 15)
            target_speed = np.clip(target_speed, 20, 120)
        
        speed_diff = target_speed - current_speed
        current_speed += speed_diff * 0.02 + np.random.normal(0, noise_level)
        current_speed = np.clip(current_speed, 0, 130)
        speed[i] = current_speed
    
    # Other metrics
    acceleration = np.gradient(speed, 1)
    rpm = speed * 15 + np.random.normal(0, 200)
    rpm = np.clip(rpm, 800, 6000)
    
    fuel_consumption = (speed * 0.1 + np.abs(acceleration) * 0.5 + 
                        np.random.normal(0, 0.5))
    fuel_consumption = np.clip(fuel_consumption, 0, 20)
    
    engine_temp = 90 + (rpm - 2000) * 0.01 + np.random.normal(0, 2)
    engine_temp = np.clip(engine_temp, 70, 110)
    
    brake_pressure = np.where(acceleration < -2, 
                             np.abs(acceleration) * 10 + np.random.normal(0, 5),
                             np.random.normal(0, 2))
    brake_pressure = np.clip(brake_pressure, 0, 100)
    
    steering_angle = np.random.normal(0, 5, n_samples)
    steering_angle = np.clip(steering_angle, -30, 30)
    
    lateral_accel = (steering_angle * speed * 0.001 + np.random.normal(0, 0.5))
    lateral_accel = np.clip(lateral_accel, -5, 5)
    
    throttle = np.where(acceleration > 0, 
                        np.clip(acceleration * 10 + 20, 0, 100),
                        np.random.normal(20, 5))
    throttle = np.clip(throttle, 0, 100)
    
    # Gear based on speed
    gear = np.ones(n_samples)
    gear[speed < 20] = 1
    gear[(speed >= 20) & (speed < 40)] = 2
    gear[(speed >= 40) & (speed < 60)] = 3
    gear[(speed >= 60) & (speed < 80)] = 4
    gear[(speed >= 80) & (speed < 100)] = 5
    gear[speed >= 100] = 6
    
    return {
        'timestamp': timestamps,
        'speed': speed,
        'acceleration': acceleration,
        'rpm': rpm,
        'fuel_consumption': fuel_consumption,
        'engine_temperature': engine_temp,
        'brake_pressure': brake_pressure,
        'steering_angle': steering_angle,
        'lateral_acceleration': lateral_accel,
        'throttle_position': throttle,
        'gear': gear
    }


def inject_aggressive_driving_events(data, event_probability=0.05):
    """Inject aggressive driving events into normal driving data."""
    n_samples = len(data['speed'])
    aggressive_events = []
    
    # Generate random event locations
    event_locations = np.random.choice([True, False], n_samples, 
                                     p=[event_probability, 1-event_probability])
    
    for i in range(n_samples):
        if event_locations[i]:
            event_type = np.random.choice(['sudden_acceleration', 'hard_braking', 
                                         'sharp_turning', 'high_speed'])
            
            if event_type == 'sudden_acceleration':
                data['speed'][i:i+10] += np.random.uniform(20, 40)
                data['acceleration'][i:i+10] += np.random.uniform(5, 15)
                data['throttle_position'][i:i+10] = 100
                data['rpm'][i:i+10] += np.random.uniform(1000, 2000)
                
            elif event_type == 'hard_braking':
                data['speed'][i:i+15] -= np.random.uniform(30, 50)
                data['acceleration'][i:i+15] -= np.random.uniform(8, 20)
                data['brake_pressure'][i:i+15] = 100
                
            elif event_type == 'sharp_turning':
                data['steering_angle'][i:i+20] += np.random.uniform(20, 40) * np.random.choice([-1, 1])
                data['lateral_acceleration'][i:i+20] += np.random.uniform(2, 5) * np.random.choice([-1, 1])
                
            elif event_type == 'high_speed':
                data['speed'][i:i+30] += np.random.uniform(30, 50)
                data['rpm'][i:i+30] += np.random.uniform(1500, 2500)
                data['fuel_consumption'][i:i+30] += np.random.uniform(5, 10)
            
            aggressive_events.append({
                'timestamp': data['timestamp'][i],
                'event_type': event_type,
                'index': i
            })
    
    # Clip values to realistic ranges
    data['speed'] = np.clip(data['speed'], 0, 130)
    data['acceleration'] = np.clip(data['acceleration'], -20, 20)
    data['rpm'] = np.clip(data['rpm'], 800, 6000)
    data['fuel_consumption'] = np.clip(data['fuel_consumption'], 0, 20)
    data['engine_temperature'] = np.clip(data['engine_temperature'], 70, 110)
    data['brake_pressure'] = np.clip(data['brake_pressure'], 0, 100)
    data['steering_angle'] = np.clip(data['steering_angle'], -45, 45)
    data['lateral_acceleration'] = np.clip(data['lateral_acceleration'], -8, 8)
    data['throttle_position'] = np.clip(data['throttle_position'], 0, 100)
    
    return data, aggressive_events


def generate_driver_profile(driver_id, n_samples=10000):
    """Generate driving data for a specific driver with unique characteristics."""
    # Driver-specific characteristics
    driver_profiles = {
        '10': {'base_speed': 55, 'noise_level': 3, 'aggressive_prob': 0.02},   # Conservative
        '11': {'base_speed': 65, 'noise_level': 4, 'aggressive_prob': 0.04},   # Normal
        '12': {'base_speed': 75, 'noise_level': 6, 'aggressive_prob': 0.06},   # Slightly aggressive
        '40': {'base_speed': 70, 'noise_level': 5, 'aggressive_prob': 0.05},   # Normal
        '45': {'base_speed': 80, 'noise_level': 7, 'aggressive_prob': 0.08},   # Aggressive
        '50': {'base_speed': 60, 'noise_level': 4, 'aggressive_prob': 0.03},   # Normal
        '51': {'base_speed': 85, 'noise_level': 8, 'aggressive_prob': 0.10},   # Very aggressive
        '52': {'base_speed': 55, 'noise_level': 3, 'aggressive_prob': 0.02},   # Conservative
        '55': {'base_speed': 70, 'noise_level': 5, 'aggressive_prob': 0.05},   # Normal
        '56': {'base_speed': 75, 'noise_level': 6, 'aggressive_prob': 0.07},   # Slightly aggressive
        '56_0': {'base_speed': 65, 'noise_level': 4, 'aggressive_prob': 0.04}, # Normal
        '57': {'base_speed': 80, 'noise_level': 7, 'aggressive_prob': 0.09},   # Aggressive
        '60': {'base_speed': 60, 'noise_level': 4, 'aggressive_prob': 0.03},   # Normal
        '60_0': {'base_speed': 70, 'noise_level': 5, 'aggressive_prob': 0.05}, # Normal
        '61': {'base_speed': 75, 'noise_level': 6, 'aggressive_prob': 0.06},   # Slightly aggressive
        '62': {'base_speed': 85, 'noise_level': 8, 'aggressive_prob': 0.11},   # Very aggressive
        '63': {'base_speed': 65, 'noise_level': 4, 'aggressive_prob': 0.04},   # Normal
        '63_2': {'base_speed': 70, 'noise_level': 5, 'aggressive_prob': 0.05}, # Normal
        '64': {'base_speed': 75, 'noise_level': 6, 'aggressive_prob': 0.07},   # Slightly aggressive
        '64_0': {'base_speed': 60, 'noise_level': 4, 'aggressive_prob': 0.03}, # Normal
        '65': {'base_speed': 80, 'noise_level': 7, 'aggressive_prob': 0.08},   # Aggressive
        '66': {'base_speed': 70, 'noise_level': 5, 'aggressive_prob': 0.05},   # Normal
        '67': {'base_speed': 75, 'noise_level': 6, 'aggressive_prob': 0.06},   # Slightly aggressive
        '68': {'base_speed': 65, 'noise_level': 4, 'aggressive_prob': 0.04},   # Normal
        '69': {'base_speed': 80, 'noise_level': 7, 'aggressive_prob': 0.09},   # Aggressive
        '70': {'base_speed': 70, 'noise_level': 5, 'aggressive_prob': 0.05},   # Normal
        '71': {'base_speed': 75, 'noise_level': 6, 'aggressive_prob': 0.07},   # Slightly aggressive
        '72': {'base_speed': 60, 'noise_level': 4, 'aggressive_prob': 0.03},   # Normal
        '73': {'base_speed': 85, 'noise_level': 8, 'aggressive_prob': 0.10},   # Very aggressive
        '74': {'base_speed': 70, 'noise_level': 5, 'aggressive_prob': 0.05},   # Normal
        '77': {'base_speed': 75, 'noise_level': 6, 'aggressive_prob': 0.06},   # Slightly aggressive
        '78': {'base_speed': 65, 'noise_level': 4, 'aggressive_prob': 0.04},   # Normal
        '80': {'base_speed': 80, 'noise_level': 7, 'aggressive_prob': 0.08},   # Aggressive
        '81': {'base_speed': 70, 'noise_level': 5, 'aggressive_prob': 0.05},   # Normal
        '82': {'base_speed': 75, 'noise_level': 6, 'aggressive_prob': 0.07},   # Slightly aggressive
        '83': {'base_speed': 60, 'noise_level': 4, 'aggressive_prob': 0.03},   # Normal
        '84': {'base_speed': 85, 'noise_level': 8, 'aggressive_prob': 0.11},   # Very aggressive
        '85': {'base_speed': 70, 'noise_level': 5, 'aggressive_prob': 0.05},   # Normal
        '86': {'base_speed': 75, 'noise_level': 6, 'aggressive_prob': 0.06},   # Slightly aggressive
        '87': {'base_speed': 65, 'noise_level': 4, 'aggressive_prob': 0.04},   # Normal
        '89': {'base_speed': 80, 'noise_level': 7, 'aggressive_prob': 0.09}    # Aggressive
    }
    
    # Get driver profile or use default
    profile = driver_profiles.get(driver_id, {
        'base_speed': 65, 'noise_level': 5, 'aggressive_prob': 0.05
    })
    
    # Generate normal driving data
    data = generate_normal_driving_pattern(
        n_samples=n_samples,
        base_speed=profile['base_speed'],
        noise_level=profile['noise_level']
    )
    
    # Inject aggressive driving events
    data, events = inject_aggressive_driving_events(
        data, event_probability=profile['aggressive_prob']
    )
    
    # Add driver ID
    data['driver_id'] = [driver_id] * n_samples
    
    return data, events


def save_driving_data(data, events, output_path, driver_id):
    """Save driving data to CSV file."""
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save main data
    output_file = os.path.join(output_path, f'{driver_id}.csv')
    df.to_csv(output_file, index=False)
    
    # Save events summary
    if events:
        events_df = pd.DataFrame(events)
        events_file = os.path.join(output_path, f'{driver_id}_events.csv')
        events_df.to_csv(events_file, index=False)
    
    print(f"Saved {driver_id}.csv with {len(df)} samples and {len(events)} events")


def main():
    """Main function to generate all driving data files."""
    # Configuration
    data_path = './synthetic_driving_data/'
    n_samples = 10000  # 10K samples per driver
    
    # Create output directory
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
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
    
    print(f"Generating synthetic driving data for {len(driver_ids)} drivers...")
    print(f"Each driver will have {n_samples:,} samples")
    print(f"Output directory: {data_path}")
    print("-" * 60)
    
    # Generate data for each driver
    all_events = []
    for i, driver_id in enumerate(driver_ids, 1):
        print(f"Processing driver {driver_id} ({i}/{len(driver_ids)})...")
        
        # Generate data
        data, events = generate_driver_profile(driver_id, n_samples)
        
        # Save data
        save_driving_data(data, events, data_path, driver_id)
        
        # Collect events for summary
        all_events.extend(events)
    
    # Generate summary statistics
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Total drivers processed: {len(driver_ids)}")
    print(f"Total samples generated: {len(driver_ids) * n_samples:,}")
    print(f"Total aggressive events: {len(all_events)}")
    print(f"Average events per driver: {len(all_events) / len(driver_ids):.1f}")
    
    # Event type distribution
    event_types = [event['event_type'] for event in all_events]
    event_counts = pd.Series(event_types).value_counts()
    print("\nEvent type distribution:")
    for event_type, count in event_counts.items():
        print(f"  {event_type}: {count} ({count/len(all_events)*100:.1f}%)")
    
    print(f"\nData saved to: {os.path.abspath(data_path)}")
    print("\nYou can now use this data with your aggressive driving detection notebook!")


if __name__ == "__main__":
    main()
