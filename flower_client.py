"""Optional Flower hospital-node demo client using synthetic data only.
Usage: python flower_client.py --client-id 0 --server 127.0.0.1:8080
"""
import argparse
import flwr as fl
from fedguard_ml import DiagnosticNet, state_to_arrays, arrays_to_state, synthetic_client_data, _dp_local_train, _evaluate, LOCAL_EPOCHS


class FedGuardClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = DiagnosticNet()
        self.x, self.y = synthetic_client_data(client_id)

    def get_parameters(self, config):
        return state_to_arrays(self.model)

    def fit(self, parameters, config):
        arrays_to_state(self.model, parameters)
        noise = float(config.get('noise_multiplier', 1.0))
        steps = sum(_dp_local_train(self.model, self.x, self.y, noise) for _ in range(LOCAL_EPOCHS))
        return state_to_arrays(self.model), len(self.x), {'steps': steps, 'client_id': self.client_id}

    def evaluate(self, parameters, config):
        arrays_to_state(self.model, parameters)
        acc, loss = _evaluate(self.model, self.x, self.y)
        return float(loss), len(self.x), {'accuracy': float(acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-id', type=int, required=True, choices=range(0, 100))
    parser.add_argument('--server', default='127.0.0.1:8080')
    args = parser.parse_args()
    fl.client.start_numpy_client(server_address=args.server, client=FedGuardClient(args.client_id))


if __name__ == '__main__':
    main()
