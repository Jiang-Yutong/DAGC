from MFL.client.SyncClient import SyncClient
from MFL.server.SyncServer import SyncServer

from MFL.datasets import get_data

from MFL.tools import jsonTool
import MFL.algo

import os
import argparse
import torch
import json
import numpy as np
import random
import copy

SYNC_BANDWIDTH = 1


def get_args():
    parser = argparse.ArgumentParser(description='federated learning')
    parser.add_argument('--model', type=str, default='CNN3', help='the type of model', choices=['CNN1', 'CNN3'])
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='set the dataset to be used')
    parser.add_argument('--n_clients', type=int, default=10, help='the number of clients participating in training')
    parser.add_argument('--epochs', type=int, default=10, help='the number of global epochs')
    parser.add_argument('--res_path', type=str, default="./results/test", help='The directory path to save result.')
    parser.add_argument('--log_freq', type=int, default=10, help='log frequency')

    parser.add_argument('--li', type=int, default=10, help='the number of local iteration')
    parser.add_argument('--bs', type=int, default=64, help='set the batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='set the learning rate')
    parser.add_argument('--cr', type=float, default=1, help='set the uplink compression ratio')
    parser.add_argument('--comp', type=str, default='none', help='set the uplink compression method')
    parser.add_argument('--err_feedback', action='store_true', help='use the error feedback')
    parser.add_argument('--device', type=str, default='cuda:0', help='set the uplink compression method')
    parser.add_argument('--seed', type=int, default=123, help='the random seed')
    parser.add_argument('--niid', type=int, default=-1, help='the niid split')
    parser.add_argument('--skew', type=int, default=-1, help='the skew of dataset')
    parser.add_argument('--DAGC', action='store_true', help='DAGC')
    parser.add_argument('--accordion', action='store_true', help='accordion')
    parser.add_argument('--acc_min', type=float, default=0.01, help='the minimum compression ratio of accordion')
    parser.add_argument('--acc_max', type=float, default=0.1, help='the maximum compression ratio of accordion')

    args = parser.parse_args()

    assert args.n_clients > 0, f"The number of clients must be greater than 0."
    assert args.epochs > 0, f"The number of global epochs must be greater than 0."
    assert args.li > 0, f"The number of local iteration must be greater than 0."
    assert args.bs > 0, f"The batch size must be greater than 0."
    assert args.lr > 0, f"The learning rate must be greater than 0."

    return args


args = get_args()
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# load config
# read config json file and generate config dict
config_file = jsonTool.get_config_file(mode="Sync")
config = jsonTool.generate_config(config_file)

# get global config
global_config = config["global"]
global_config["epoch"] = args.epochs
global_config["n_clients"] = args.n_clients
global_config["result_path"] = args.res_path
global_config['log_freq'] = args.log_freq
config["global"] = global_config

# get client's config
client_config = config["client"]
client_config["global epoch"] = args.epochs
client_config["model"] = global_config["model"] = args.model
client_config["dataset"] = global_config["dataset"] = args.dataset
client_config["local iteration"] = global_config["local iteration"] = args.li
client_config["loss function"] = global_config["loss function"]
client_config["batch_size"] = args.bs
client_config["optimizer"]["lr"] = args.lr

client_config["accordion"] = args.accordion
client_config["acc_max"] = args.acc_max
client_config["acc_min"] = args.acc_min
config["client"] = client_config

device = torch.device(args.device)

# gradient compression config
compressor_config = config["compressor"]
compressor_config["uplink"] = {"method": args.comp, "params": {"cr": args.cr, "error_feedback": args.err_feedback}}
config["compressor"] = compressor_config
compression_rates = [args.cr for i in range(args.n_clients)]

# data distribution config
data_distribution_config = config["data_distribution"]

# save config
path=config["global"]["result_path"]
config_file = os.path.join(path, 'config.txt')
if not os.path.exists(path):
    os.makedirs(path)
with open(config_file, 'w') as f:
    f.write(str(config) + '\n')
    f.write(str(args)+ '\n')

if args.niid > 0:
    data_distribution_config["iid"] = False
    data_distribution_config["customize"] = True
    data_distribution_config["cus_distribution"] = [args.niid] * args.n_clients

if args.skew > 1 or args.DAGC:
    ratio_matrix = np.random.dirichlet(np.repeat(1,10), size=args.n_clients)
    ratio_matrix = MFL.algo.get_skew_distribution(ratio_matrix,args.skew)

    ratio_matrix = sorted(ratio_matrix, key=lambda x: sum(x), reverse=True)
    ratio_matrix = [ratio_matrix[i] for i in range(args.n_clients)]

    data_distribution_config["iid"] = False
    data_distribution_config["customize"] = True
    data_distribution_config["cus_distribution"] = ratio_matrix

if args.DAGC:
    if args.comp=='topk':
        compression_rates = MFL.algo.DAGC_R(ratio_matrix,args.cr)
    elif args.comp=='ht':
        compression_rates = MFL.algo.DAGC_A(ratio_matrix,args.cr)


def create_clients_server(n_clients, split, test_set):
    clients = []
    for i in range(n_clients):
        client_comp_config = copy.deepcopy(compressor_config)
        client_comp_config["uplink"]['params']['cr'] = compression_rates[i]
        clients += [SyncClient(cid=i,
                            dataset=split[i],
                            client_config=client_config,
                            compression_config=client_comp_config,
                            bandwidth=SYNC_BANDWIDTH,
                            device=device)]
    server = SyncServer(global_config=global_config,
                    dataset=test_set,
                    compressor_config=compressor_config,
                    clients=clients,
                    device=device)
    return clients, server


if __name__ == "__main__":
    # print config
    jsonTool.print_config(config)

    n_clients = args.n_clients
    split, test_set = get_data(args.dataset, n_clients, data_distribution_config)
    clients, server = create_clients_server(n_clients, split, test_set)

    for client in clients:
        client.set_server(server)

    # start training
    server.start()
