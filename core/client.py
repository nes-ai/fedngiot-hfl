import torch
import torch.nn as nn
import torch.optim as optim

class Client:
    def __init__(self, client_id, model, data_loader, device, device_profile, network_simulator=None):
        self.id = client_id
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.network = network_simulator
        self.profile = network_simulator.__dict__ if network_simulator else {}

        self.location = device_profile.get("location", (0.0, 0.0))
        self.compute_power = device_profile.get("compute_power", 1.0)

    def train(self, epochs=1, lr=0.01):
        self.model.train()
        self.model.to(self.device)
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            for x, y in self.data_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                optimizer.step()

        if self.network:
            self.network.simulate_delay()
            if self.network.simulate_failure():
                print(f"[Client {self.id}] Network failure")
                return None

        return self.model.state_dict()

    def get_network_profile(self):
        return self.profile
