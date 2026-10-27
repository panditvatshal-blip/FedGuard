"""Optional standalone Flower server for connecting separate demo clients.
Run: python flower_server.py. Uses synthetic data only.
"""
import os
import flwr as fl
from fedguard_ml import DiagnosticNet, state_to_arrays

HOST = os.environ.get('FLOWER_BIND', '0.0.0.0:8080')


def main():
    model = DiagnosticNet()
    initial = fl.common.ndarrays_to_parameters(state_to_arrays(model))
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=int(os.environ.get('FLOWER_MIN_CLIENTS', '2')),
        min_available_clients=int(os.environ.get('FLOWER_MIN_CLIENTS', '2')),
        initial_parameters=initial,
        on_fit_config_fn=lambda rnd: {'noise_multiplier': float(os.environ.get('DP_NOISE', '1.0'))},
    )
    print(f'Flower FedAvg server listening on {HOST}; synthetic demo data only.')
    fl.server.start_server(server_address=HOST, config=fl.server.ServerConfig(num_rounds=int(os.environ.get('FLOWER_ROUNDS', '3'))), strategy=strategy)


if __name__ == '__main__':
    main()
