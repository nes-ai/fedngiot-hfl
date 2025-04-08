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
    
    def _slice_sorted_clients(self, sorted_clients):
        cluster_size = len(sorted_clients) // self.num_clusters
        clusters = {i: [] for i in range(self.num_clusters)}
        for i in range(self.num_clusters):
            start = i * cluster_size
            end = (i + 1) * cluster_size if i < self.num_clusters - 1 else len(sorted_clients)
            clusters[i] = sorted_clients[start:end]
        return clusters
    
    def assign_clients_delay(self): # Network-aware clustering (mean_delay)
        sorted_clients = sorted(
            self.clients,
            key=lambda c: c.get_network_profile().get("mean_delay", 1.0)
        )
        self.clusters = self._slice_sorted_clients(sorted_clients)
        return self.clusters

    def assign_clients_location(self, center=(0.5, 0.5)): # Location-aware clustering
        sorted_clients = sorted(
            self.clients,
            key=lambda c: dist(c.location, center)
        )
        self.clusters = self._slice_sorted_clients(sorted_clients)
        return self.clusters

    def assign_clients_compute(self): # Compute power-based clustering
        sorted_clients = sorted(
            self.clients,
            key=lambda c: c.compute_power,
            reverse=True
        )
        self.clusters = self._slice_sorted_clients(sorted_clients)
        return self.clusters

    def assign_clients_compute_location(self, alpha=0.5, center=(0.5, 0.5)): # Hybrid of compute power & location
        sorted_clients = sorted(
            self.clients,
            key=lambda c: alpha * dist(c.location, center) - (1 - alpha) * c.compute_power
        )
        self.clusters = self._slice_sorted_clients(sorted_clients)
        return self.clusters
