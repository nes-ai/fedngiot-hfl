import random
from math import dist

class ClusterManager:
    def __init__(self, clients, num_clusters):
        self.clients = clients
        self.num_clusters = num_clusters
        self.clusters = {i: [] for i in range(num_clusters)}

    def assign_clients_random(self): # Simple random assignment
        for client in self.clients:
            cluster_id = random.randint(0, self.num_clusters - 1)  
            self.clusters[cluster_id].append(client)
        return self.clusters

    def assign_clients_delay(self):  # Network-aware clustering (mean_delay)
        sorted_clients = sorted(
            self.clients,
            key=lambda c: c.get_network_profile().get("mean_delay", 1.0)
        )

        clusters = {i: [] for i in range(self.num_clusters)}
        for i, client in enumerate(sorted_clients):
            cluster_id = i % self.num_clusters
            clusters[cluster_id].append(client)

        self.clusters = clusters
        return self.clusters

    def assign_clients_location(self, center=(0.5, 0.5)):  # Location-aware clustering
        sorted_clients = sorted(
            self.clients,
            key=lambda c: dist(c.location, center)
        )
        clusters = {i: [] for i in range(self.num_clusters)}
        for i, client in enumerate(sorted_clients):
            cluster_id = i % self.num_clusters
            clusters[cluster_id].append(client)
        self.clusters = clusters
        return self.clusters

    def assign_clients_compute(self):  # Compute power-based clustering
        sorted_clients = sorted(
            self.clients,
            key=lambda c: c.compute_power,
            reverse=True
        )
        clusters = {i: [] for i in range(self.num_clusters)}
        for i, client in enumerate(sorted_clients):
            cluster_id = i % self.num_clusters
            clusters[cluster_id].append(client)
        self.clusters = clusters
        return self.clusters

    def assign_clients_compute_location(self, alpha=0.5, center=(0.5, 0.5)):  # Hybrid of compute power & location
        sorted_clients = sorted(
            self.clients,
            key=lambda c: alpha * dist(c.location, center) - (1 - alpha) * c.compute_power
        )
        clusters = {i: [] for i in range(self.num_clusters)}
        for i, client in enumerate(sorted_clients):
            cluster_id = i % self.num_clusters
            clusters[cluster_id].append(client)
        self.clusters = clusters
        return self.clusters