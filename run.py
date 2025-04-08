import os
import yaml
import json
from datetime import datetime
from utils.logger import init_logger
from core.controller import FLController
from data.dataset_loader import load_dataset
from models.cnn import SimpleCNN
from core.client import Client
from network.simulator import NetworkSimulator
import torch



def log_configuration(logger, config):
    logger.info("===== Experiment Configuration =====")
    yaml_config_str = yaml.dump(config, sort_keys=False, default_flow_style=False)
    for line in yaml_config_str.splitlines():
        logger.info(line)
    logger.info("====================================")

if __name__ == "__main__":
    config = yaml.safe_load(open("config/config.yaml"))
    num_clients = config['global']['num_clients']
    dataset_name = config['global'].get('dataset', 'MNIST')
    partition = config['global'].get('partition', 'iid')
    network_simulation = config['global'].get('network_simulation', True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("log", exist_ok=True)
    log_filename = f"log/{timestamp}_{dataset_name.lower()}_training.log"
    logger = init_logger(log_file=log_filename)
    log_configuration(logger, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loaders, test_loader = load_dataset(dataset_name, num_clients, partition=partition)
    
    if dataset_name.upper() == 'MNIST':
        model = SimpleCNN(input_channels=1, input_size=(28, 28), num_classes=10, n_conv_layers=1)
    elif dataset_name.upper() == 'FASHIONMNIST':
        model = SimpleCNN(input_channels=1, input_size=(28, 28), num_classes=10, n_conv_layers=2)
    elif dataset_name.upper() == 'CIFAR10':
        model = SimpleCNN(input_channels=3, input_size=(32, 32), num_classes=10, n_conv_layers=3)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported!")

    # Load network and device profile
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    network_profile_path = os.path.join(repo_root, "config", "network_profiles.json")
    with open(network_profile_path) as f:
        all_network_profiles = json.load(f)
    selected_network_profiles = all_network_profiles[:num_clients]

    device_profile_path = os.path.join(repo_root, "config", "device_profiles.json")
    with open(device_profile_path) as f:
        all_device_profiles = json.load(f)
    selected_device_profiles = all_device_profiles[:num_clients]

    # Setting clients
    clients = []
    for i, (dl, net_profile, dev_profile) in enumerate(zip(train_loaders, selected_network_profiles, selected_device_profiles)):
        net = NetworkSimulator(net_profile)
        if network_simulation: 
            client = Client(i, model, dl, device, dev_profile, network_simulator=net)
        else:
            client = Client(i, model, dl, device, dev_profile)
        clients.append(client)

    # HFL run
    controller = FLController(config, logger, test_loader, device)
    controller.clients = clients
    controller.run()