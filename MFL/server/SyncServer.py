import torch
from ..model import get_model
import numpy as np

import threading
import queue
import copy
import schedule

from ..datasets import CustomerDataset, get_default_data_transforms

import MFL.server.ScheduleClass as sc

import MFL.tools.jsonTool as jsonTool
import MFL.tools.tensorTool as tl
import MFL.tools.resultTools as rt


def update_list(lst, num):
    if len(lst) == 0:
        lst.append(num)
    else:
        lst.append(lst[-1] + num)


class SyncServer:
    def __init__(self, global_config, dataset, compressor_config, clients, device):
        # global_config
        self.global_config = global_config
        self.schedule_config = global_config["schedule"]

        # device
        self.device = device

        # model
        self.model_name = global_config["model"]
        self.model = get_model(self.model_name).to(device)  # mechine learning model
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.dW_compress = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}

        self.A = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}

        # receive queue
        self.parameter_queue = queue.Queue()

        # dataset
        self.dataset_name = global_config["dataset"]
        self.dataset = dataset

        # global iteration
        self.current_epoch = 0  # indicate the version of global model
        self.total_epoch = global_config["epoch"]

        # loss function
        self.loss_fun_name = global_config["loss function"]  # loss function
        self.loss_func = self.init_loss_fun()

        self.compressor_config = compressor_config

        # results
        self.staleness_list = []
        self.loss_list = []
        self.accuracy_list = []
        self.time_list = []  # used time
        self.communication_list = []  # communication bandwith consumption

        # global manager
        self.global_manager = SyncGlobalManager(clients=clients,
                                                dataset=dataset,
                                                global_config=global_config)

    def start(self):  # start the whole training priod
        print("Start global training...\n")

        self.update()

        # Exit
        print("Global Updater Exit.\n")

    def update(self):
        for epoch in range(self.total_epoch):
            # select clients
            participating_clients = sc.random_schedule(self.global_manager.clients_list, self.schedule_config)
            for client in participating_clients:
                client.run()

            client_gradients = []  # save multi local_W
            data_nums = []
            self.current_epoch += 1

            # compute 
            max_time = -1
            communication_cost = 0
            while not self.parameter_queue.empty():
                transmit_dict = self.parameter_queue.get()  # get information from client,(cid, client_gradient, data_num, timestamp)
                cid = transmit_dict["cid"]  # cid
                client_gradient = transmit_dict["client_gradient"]  # client gradient
                data_num = transmit_dict["data_num"]  # number of data samples

                # total computation time
                computation_time_cid = transmit_dict["computation_consumption"]
                # communication traffic consumption
                communication_consumption_cid = transmit_dict["communication_consumption"]
                communication_cost += communication_consumption_cid
                # communication time
                communication_time_cid = transmit_dict["communication_time"]
                max_time = max(communication_time_cid + computation_time_cid, max_time)  # total time

                client_gradients.append(client_gradient)
                data_nums.append(data_num)


            update_list(self.time_list, max_time)
            update_list(self.communication_list, communication_cost)

            data_nums = torch.Tensor(data_nums)
            tl.weighted_average(target=self.dW,
                                sources=client_gradients,
                                weights=data_nums)  # global gradient
            tl.add(target=self.W, source=self.dW)

            if self.current_epoch % self.global_config['log_freq'] == 0:
                self.eval_model()

            # save results
            if self.current_epoch % self.global_config['log_freq'] == 0 or self.current_epoch == self.total_epoch:
                global_acc, global_loss = self.get_accuracy_and_loss_list()
                rt.save_results(self.global_config["result_path"],
                                global_loss=global_loss,
                                global_acc=global_acc
                                )

    def init_loss_fun(self):
        if self.loss_fun_name == 'CrossEntropy':
            return torch.nn.CrossEntropyLoss()
        elif self.loss_fun_name == 'MSE':
            return torch.nn.MSELoss()

    def scatter_init_model(self):
        for cid, client in self.global_manager.get_clients_dict().items():
            client.synchronize_with_server(self)
            model_timestamp = copy.deepcopy(self.current_epoch)["t"]
            client.model_timestamp = model_timestamp

    def schedule(self, clients, schedule_config, **kwargs):
        participating_clients = sc.schedule(clients, schedule_config)
        return participating_clients

    def select_clients(self, participating_clients):
        for client in participating_clients:
            client.set_selected_event(True)

    def receive(self, transmit_dict):
        self.parameter_queue.put(transmit_dict)

    def eval_model(self):
        self.model.eval()
        data_loader = self.global_manager.test_loader
        test_correct = 0.0
        test_loss = 0.0
        test_num = 0
        batch_num = 0
        for data in data_loader:
            features, labels = data
            features = features.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(features)  # predict
            _, id = torch.max(outputs.data, 1)
            test_loss += self.loss_func(outputs, labels).item()
            test_correct += torch.sum(id == labels.data).cpu().numpy()
            test_num += len(features)
            batch_num += 1
        accuracy = test_correct / test_num
        loss = test_loss / batch_num

        self.accuracy_list.append(accuracy)
        self.loss_list.append(loss)
        print("Server: Global Epoch {}, Test Accuracy: {} , Test Loss: {}".format(self.current_epoch, accuracy, loss))

    def get_accuracy_and_loss_list(self):
        return self.accuracy_list, self.loss_list

    def get_staleness_list(self):
        return self.staleness_list


class SyncGlobalManager:  # Manage clients and global information
    def __init__(self, clients, dataset, global_config):
        # clients
        self.clients_num = len(clients)
        self.clients_list = clients
        self.clients_dict = {}
        self.register_clients(clients)

        # global infromation
        self.global_epoch = global_config["epoch"]  # global epoch/iteration
        self.global_acc = []  # test accuracy
        self.global_loss = []  # training loss

        # global test dataset
        self.dataset_name = global_config["dataset"]
        self.dataset = dataset  # the test dataset of server, a list with 2 elements, the first is all data, the second is all label
        self.x_test = dataset[0]
        self.y_test = dataset[1]
        if type(self.x_test) == torch.Tensor:
            self.x_test, self.y_test = self.x_test.numpy(), self.y_test.numpy()
        elif type(self.y_test) == list:
            self.y_test = np.array(self.y_test)
        
        self.transforms_train, self.transforms_eval = get_default_data_transforms(self.dataset_name)
        self.test_loader = torch.utils.data.DataLoader(CustomerDataset(self.x_test, self.y_test, self.transforms_eval),
                                                       batch_size=100,
                                                       shuffle=False)

    def find_client_by_cid(self, cid):  # find client by cid
        for client in self.clients:
            if client.cid == cid:
                return client
        return None

    def get_clients_dict(self):
        return self.clients_dict

    def register_clients(self, clients):  # add multi-clients to server scheduler
        for client in clients:
            self.add_client(client)

    def add_client(self, client):  # add one client to server scheduler
        cid = client.cid
        if cid in self.clients_dict.keys():
            raise Exception("Client id conflict.")
        self.clients_dict[cid] = client

    def start_one_client(self, cid):
        clients_dict = self.get_clients_dict()  # get all clients
        for c in clients_dict.keys():
            if c == cid:
                clients_dict[c].start()
