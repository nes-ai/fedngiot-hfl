import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from collections import defaultdict
import random
import numpy as np

def create_label_skew_splits(dataset, num_clients, classes_per_client=2):
    label_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label_indices[label].append(idx)

    for label in label_indices:
        random.shuffle(label_indices[label])

    client_indices = [[] for _ in range(num_clients)]
    all_labels = list(label_indices.keys())

    for client_id in range(num_clients):
        selected_labels = random.sample(all_labels, classes_per_client)
        for label in selected_labels:
            take = len(label_indices[label]) // num_clients
            client_indices[client_id].extend(label_indices[label][:take])
            label_indices[label] = label_indices[label][take:]

    return [Subset(dataset, indices) for indices in client_indices]

def create_dirichlet_splits(dataset, num_clients, alpha=0.5):
    labels = np.array([label for _, label in dataset])
    num_classes = len(set(labels))
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]
    for c, indices in enumerate(class_indices):
        np.random.shuffle(indices)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split = np.split(indices, proportions)
        for client_id, idx in enumerate(split):
            client_indices[client_id].extend(idx)

    return [Subset(dataset, indices) for indices in client_indices]

def load_dataset(dataset_name='MNIST', num_clients=10, batch_size=32, partition='iid'):
    transform = transforms.Compose([transforms.ToTensor()])

    if dataset_name.upper() == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)

    elif dataset_name.upper() == 'FASHIONMNIST':
        train_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform)

    elif dataset_name.upper() == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)

    else:
        raise ValueError(f"Dataset {dataset_name} not supported!")

    if partition == 'iid':
        split_size = len(train_dataset) // num_clients
        client_subsets = random_split(train_dataset, [split_size] * num_clients)

    elif partition == 'label_skew':
        client_subsets = create_label_skew_splits(train_dataset, num_clients, classes_per_client=2)

    elif partition == 'dirichlet':
        client_subsets = create_dirichlet_splits(train_dataset, num_clients, alpha=0.5)

    else:
        raise ValueError(f"Partition type {partition} not supported!")

    client_loaders = [DataLoader(data, batch_size=batch_size, shuffle=True) for data in client_subsets]
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return client_loaders, test_loader