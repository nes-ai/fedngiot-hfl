from utils.metrics import accuracy, precision, recall, f1
from core.aggregator import fedavg, dual_mode_aggregate
from core.cluster import ClusterManager
from core.model_compression import prune_model ########## Need to make compression after adding device location and capability

class FLController:
    def __init__(self, config, logger, test_loader, device):
        self.config = config
        self.logger = logger
        self.clients = []
        self.test_loader = test_loader
        self.device = device

    def run(self):
        cluster_mgr = ClusterManager(self.clients, self.config['clustering']['num_clusters'])
        clusters = cluster_mgr.assign_clients_delay() # Option: assign_clients_random, assign_clients_delay

        strategy = self.config['clustering'].get('strategy', 'random')
        criteria = self.config['clustering'].get('cluster_criteria', [])
        
        if strategy == 'random':
            clusters = cluster_mgr.assign_clients_random()
        elif strategy == 'device-centric':
            if 'compute_power' in criteria and 'location' in criteria:
                clusters = cluster_mgr.assign_clients_compute_location()
            elif 'compute_power' in criteria:
                clusters = cluster_mgr.assign_clients_compute()
            elif 'location' in criteria:
                clusters = cluster_mgr.assign_clients_location()
            else:
                self.logger.warning("No valid device-centric clustering criteria found. Falling back to random.")
                clusters = cluster_mgr.assign_clients_random()
        elif strategy == 'network-centric':
            clusters = cluster_mgr.assign_clients_delay()
        else:
            self.logger.warning("Unknown clustering strategy. Using random assignment.")
            clusters = cluster_mgr.assign_clients_random()
        
        global_model = None
        for rnd in range(self.config['global']['num_rounds']):            
            global_model = dual_mode_aggregate(
                clusters,
                intra=self.config['aggregation'].get('intra_cluster', 'sync'),
                inter=self.config['aggregation'].get('inter_cluster', 'sync'),
                participation_ratio=self.config['aggregation'].get('participation_ratio', 0.6),
                compress_enabled=self.config['global'].get('model_compression', False)
            )

            for client in self.clients:
                client.model.load_state_dict(global_model)

            acc = accuracy(self.clients[0].model.to(self.device), self.test_loader, self.device)
            prec = precision(self.clients[0].model.to(self.device), self.test_loader, self.device)
            rec = recall(self.clients[0].model.to(self.device), self.test_loader, self.device)
            f1s = f1(self.clients[0].model.to(self.device), self.test_loader, self.device)

            self.logger.info(f"Round {rnd+1} - Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1s:.4f}")
