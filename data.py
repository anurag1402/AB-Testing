import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Set parameters
total_samples = 20000  # Total number of users to simulate
control_conversion_rate = 0.035  # 3.5% conversion rate for control group
treatment_conversion_rate = 0.042  # 4.2% conversion rate for treatment group
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 1, 31)

# Create user IDs
user_ids = [f"user_{i}" for i in range(1, total_samples + 1)]

# Assign variants (50/50 split)
variants = np.random.choice(['A', 'B'], size=total_samples)

# Generate timestamps across the test period
date_range = (end_date - start_date).days
timestamps = [start_date + timedelta(
    days=random.randint(0, date_range),
    hours=random.randint(0, 23),
    minutes=random.randint(0, 59),
    seconds=random.randint(0, 59)
) for _ in range(total_samples)]

# Generate conversions based on group assignment
conversions = []
for variant in variants:
    if variant == 'A':
        # Control group conversion
        conversions.append(1 if random.random() < control_conversion_rate else 0)
    else:
        # Treatment group conversion
        conversions.append(1 if random.random() < treatment_conversion_rate else 0)

# Additional metrics that might be useful
session_duration = []
pages_viewed = []
for i in range(total_samples):
    # Users who converted tend to have longer sessions and view more pages
    if conversions[i] == 1:
        session_duration.append(random.randint(120, 900))  # 2-15 minutes
        pages_viewed.append(random.randint(3, 15))
    else:
        session_duration.append(random.randint(10, 300))  # 10 sec - 5 minutes
        pages_viewed.append(random.randint(1, 8))

# Create device types with realistic distribution
devices = np.random.choice(['desktop', 'mobile', 'tablet'], 
                          p=[0.55, 0.35, 0.10], 
                          size=total_samples)

# Create traffic sources
sources = np.random.choice(['direct', 'organic', 'social', 'email', 'paid'], 
                          p=[0.30, 0.25, 0.20, 0.15, 0.10], 
                          size=total_samples)

# Create the DataFrame
df = pd.DataFrame({
    'user_id': user_ids,
    'variant': variants,
    'timestamp': timestamps,
    'converted': conversions,
    'session_duration_seconds': session_duration,
    'pages_viewed': pages_viewed,
    'device': devices,
    'traffic_source': sources
})

# Sort by timestamp to make it look more realistic
df = df.sort_values('timestamp')

# Save to CSV
df.to_csv('Website_Results.csv', index=False)

print(f"Dataset created with {total_samples} rows")
print(f"Control group size: {sum(variants == 'A')}")
print(f"Treatment group size: {sum(variants == 'B')}")
print(f"Control conversions: {sum([c for v, c in zip(variants, conversions) if v == 'A'])}")
print(f"Treatment conversions: {sum([c for v, c in zip(variants, conversions) if v == 'B'])}")
print("Saved to 'ab_test_data.csv'")

# Preview the first few rows
print("\nData preview:")
print(df.head())

# Calculate actual conversion rates to verify
control_actual_rate = sum([c for v, c in zip(variants, conversions) if v == 'A']) / sum(variants == 'A') * 100
treatment_actual_rate = sum([c for v, c in zip(variants, conversions) if v == 'B']) / sum(variants == 'B') * 100

print(f"\nActual control conversion rate: {control_actual_rate:.2f}%")
print(f"Actual treatment conversion rate: {treatment_actual_rate:.2f}%")
print(f"Actual uplift: {(treatment_actual_rate - control_actual_rate) / control_actual_rate * 100:.2f}%")