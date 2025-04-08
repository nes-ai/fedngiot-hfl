import os
import json
import random

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
network_config_path = os.path.join(REPO_ROOT, "config", "network_profiles.json")
device_config_path = os.path.join(REPO_ROOT, "config", "device_profiles.json")

bands = [
    {
        "carrier_freq": 3.5,
        "mean_delay": (0.25, 0.35),
        "drop_rate": (0.03, 0.07),
        "bandwidth": (0.4e9, 0.6e9),
        "antenna_count": 4,
        "spatial_dof": 2,
        "rate_capacity": (0.4e9, 0.7e9)
    },
    {
        "carrier_freq": 7.8,
        "mean_delay": (0.1, 0.2),
        "drop_rate": (0.01, 0.03),
        "bandwidth": (0.6e9, 1.0e9),
        "antenna_count": 8,
        "spatial_dof": 4,
        "rate_capacity": (1.0e9, 3.0e9)
    },
    {
        "carrier_freq": 15.0,
        "mean_delay": (0.05, 0.1),
        "drop_rate": (0.005, 0.015),
        "bandwidth": (1.0e9, 1.5e9),
        "antenna_count": 16,
        "spatial_dof": 8,
        "rate_capacity": (3.0e9, 6.0e9)
    }
]

# 위치 및 연산 능력 범위 정의
locations = [
    (round(random.uniform(0.0, 1.0), 4), round(random.uniform(0.0, 1.0), 4)) for _ in range(100)
]
compute_power_range = (1.0, 10.0)  # 단위: GFLOPS

# Generate profiles
num_clients = 100
network_profiles = []
device_profiles = []

for i in range(num_clients):
    band = random.choice(bands)
    network_profiles.append({
        "client_id": i,
        "carrier_freq": band["carrier_freq"],
        "mean_delay": round(random.uniform(*band["mean_delay"]), 4),
        "drop_rate": round(random.uniform(*band["drop_rate"]), 4),
        "bandwidth": int(random.uniform(*band["bandwidth"])),
        "antenna_count": band["antenna_count"],
        "spatial_dof": band["spatial_dof"],
        "rate_capacity": int(random.uniform(*band["rate_capacity"]))
    })

    device_profiles.append({
        "client_id": i,
        "location": locations[i],
        "compute_power": round(random.uniform(*compute_power_range), 2)
    })

with open(network_config_path, "w") as f:
    json.dump(network_profiles, f, indent=2)

with open(device_config_path, "w") as f:
    json.dump(device_profiles, f, indent=2)

print(f"Generated {num_clients} network profiles to {network_config_path}")
print(f"Generated {num_clients} device profiles to {device_config_path}")
