import random
import time

class NetworkSimulator:
    def __init__(self, profile):
        """
        profile: dict with keys like 'mean_delay', 'drop_rate', 'bandwidth'
        """
        self.mean_delay = profile.get('mean_delay', 0.1)
        self.drop_rate = profile.get('drop_rate', 0.0)
        self.bandwidth = profile.get('bandwidth', 1e6)  # bytes/sec

    def simulate_delay(self):
        delay = random.expovariate(1 / self.mean_delay)
        time.sleep(delay)
        return delay

    def simulate_failure(self):
        return random.random() < self.drop_rate

    def simulate_transfer_time(self, data_size):
        """
        data_size: bytes
        """
        transfer_time = data_size / self.bandwidth
        time.sleep(transfer_time)
        return transfer_time
