import torch
from core.model_compression import prune_model, compute_sparsity

def fedavg(models):
    if not models:
        return None
    avg_model = {}
    for k in models[0].keys():
        avg_model[k] = sum([model[k] for model in models]) / len(models)
    return avg_model

def fedmedian(models):
    if not models:
        return None
    stacked = {k: torch.stack([model[k] for model in models]) for k in models[0].keys()}
    median_model = {k: torch.median(v, dim=0).values for k, v in stacked.items()}
    return median_model

def fedprox(models, mu=0.01, global_model=None): # FedProx는 로컬 학습 시 사용하는 방식이지만, 여기선 global model과 proximity 반영
    if not models or global_model is None:
        return fedavg(models)

    prox_model = {}
    for k in models[0].keys():
        avg = sum([model[k] for model in models]) / len(models)
        prox_model[k] = avg - mu * (avg - global_model[k])
    return prox_model

def fednova(models, num_samples): # num_samples: list of # samples of each clients
    if not models:
        return None

    total_samples = sum(num_samples)
    weighted_model = {}
    for k in models[0].keys():
        weighted_model[k] = sum([
            model[k] * (n / total_samples) for model, n in zip(models, num_samples)
        ])
    return weighted_model

def dual_mode_aggregate(clusters, intra='async', inter='sync', participation_ratio=0.6, compress_enabled=False):
    cluster_models = []

    # Compute power of each cluster
    if compress_enabled:
        powers = {cid: sum(c.compute_power for c in clients)/len(clients) for cid, clients in clusters.items() if clients}
        max_power = max(powers.values())
        min_power = min(powers.values())

    # Intra-cluster aggregation
    for cluster_id, clients in clusters.items():
        local_models = []

        if intra == 'async':
            sorted_clients = sorted(
                clients,
                key=lambda c: c.get_network_profile().get('mean_delay', 1.0)
            ) # Sorting based on delay
            selected_clients = sorted_clients[:int(len(clients) * participation_ratio)] # Only fast 60% of clients are selected

        else:  # 'sync'
            selected_clients = clients

        for client in selected_clients:
            model = client.train()
            if model is not None:
                local_models.append(model)

        if local_models:
            cluster_model = fedavg(local_models)
            if cluster_model:
                if compress_enabled:
                    avg_power = powers[cluster_id]
                    norm_power = (avg_power - min_power) / (max_power - min_power + 1e-6)
                    prune_ratio = 1.0 - norm_power * 0.5
                    cluster_model = prune_model(cluster_model, pruning_rate=prune_ratio)

                    sparsity = compute_sparsity(cluster_model)
                    print(f"[Cluster {cluster_id}] avg_power={avg_power:.2f}, norm_power={norm_power:.2f}, "
                          f"prune_ratio={prune_ratio:.2f}, sparsity={sparsity:.2%}")

                cluster_models.append(cluster_model)

    if inter == 'sync':
        # Inter-cluster aggregation
        if cluster_models:
            global_model = fedavg(cluster_models)
        else:
            global_model = None
    else:
        raise NotImplementedError("Only sync inter-cluster supported for now")

    return global_model