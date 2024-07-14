import flwr as fl
from typing import Dict, List, Optional, Tuple
import torch
from model import SimpleCNN
import json
import os
from web3 import Web3
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ClientTracker:
    def __init__(self, db_file='client_database.json'):
        self.db_file = db_file
        self.client_data = {}
        self.load_database()

    def load_database(self):
        if not os.path.exists(self.db_file):
            with open(self.db_file, 'w') as f:
                json.dump({}, f)
        else:
            with open(self.db_file, 'r') as f:
                self.client_data = json.load(f)
        logging.info("Database loaded with data: {}".format(self.client_data))

    def save_database(self):
        with open(self.db_file, 'w') as f:
            json.dump(self.client_data, f, indent=2)
        logging.info("Database saved with data: {}".format(self.client_data))

    def update_client(self, model_id, eth_address, compute_time):
        if model_id not in self.client_data:
            self.client_data[model_id] = {}
        if eth_address not in self.client_data[model_id]:
            self.client_data[model_id][eth_address] = {"compute_time": 0, "rounds_participated": 0}
        self.client_data[model_id][eth_address]["compute_time"] += compute_time
        self.client_data[model_id][eth_address]["rounds_participated"] += 1
        self.save_database()

    def get_client_stats(self, model_id, eth_address):
        return self.client_data.get(model_id, {}).get(eth_address, {})

    def get_all_stats(self):
        return self.client_data

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = SimpleCNN()
        self.client_tracker = ClientTracker()

    def aggregate_fit(self, rnd, results, failures):
        aggregated_result = super().aggregate_fit(rnd, results, failures)
        if aggregated_result:
            aggregated_weights, aggregated_metrics = aggregated_result
            if aggregated_metrics:
                logging.info(f"Aggregated Metrics Round {rnd}: {aggregated_metrics}")
            for client, fit_res in results:
                model_id = fit_res.metrics.get("model_id")
                eth_address = fit_res.metrics.get("eth_address")
                compute_time = fit_res.metrics.get("compute_time", 0)
                if model_id and eth_address:
                    self.client_tracker.update_client(model_id, eth_address, compute_time)
            if aggregated_weights:
                weight_ndarrays = fl.common.parameters_to_ndarrays(aggregated_weights)
                state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), weight_ndarrays)}
                torch.save(state_dict, f"model_round_{rnd}.pt")
                logging.info(f"Saved model parameters for round {rnd} in PyTorch format")
        return aggregated_result

def distribute_rewards(client_data):
    with open('model_data.json', 'r') as f:
        model_data = json.load(f)

    sep_testnet = "https://rpc.sepolia.org"
    web3 = Web3(Web3.HTTPProvider(sep_testnet))
    if not web3.is_connected():
        logging.error("Failed to connect to Sepolia testnet")
        return

    private_key = "33a7c97a75a891872eec2f5cdcc6df3f9edc2184e9f678cdbc8a59bb8d7535c1"
    sender_address = web3.eth.account.from_key(private_key).address

    for model_id, data in client_data.items():
        total_compute_time = sum(client["compute_time"] for client in data.values())
        total_rounds = sum(client["rounds_participated"] for client in data.values())
        eth_amount = model_data[model_id]["eth_amount"]
        total_eth = web3.to_wei(eth_amount, 'ether')

        for eth_address, stats in data.items():
            weight = (stats["compute_time"] / total_compute_time + stats["rounds_participated"] / total_rounds) / 2
            amount = int(total_eth * weight)

            nonce = web3.eth.get_transaction_count(sender_address, 'pending')  # Get the pending nonce
            tx = {
                'nonce': nonce,
                'to': eth_address,
                'value': amount,
                'gas': 21000,
                'gasPrice': web3.eth.gas_price,  # Use the current network gas price
                'chainId': 11155111
            }
            signed_tx = web3.eth.account.sign_transaction(tx, private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            logging.info(f"Transaction hash: {web3.to_hex(tx_hash)}")
            receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)  # Wait for transaction to be mined
            logging.info(f"Transaction receipt: {receipt}")

def main():
    strategy = SaveModelStrategy(
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        on_fit_config_fn=lambda rnd: {"rnd": rnd},
        on_evaluate_config_fn=lambda rnd: {"rnd": rnd},
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy
    )
    # Call distribute_rewards after server completes all rounds
    distribute_rewards(ClientTracker().get_all_stats())

if __name__ == "__main__":
    main()