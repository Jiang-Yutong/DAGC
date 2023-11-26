from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import torch
import numpy as np


class CIFAR10:
    def __init__(self):
        # 获取数据集
        train_datasets = datasets.CIFAR10(root='data/', train=True,
                                          transform=transforms.ToTensor(), download=True)
        test_datasets = datasets.CIFAR10(root='data/', train=False,
                                         transform=transforms.ToTensor(), download=True)
        train_datasets.data = train_datasets.data.transpose((0, 3, 1, 2))
        test_datasets.data = test_datasets.data.transpose((0, 3, 1, 2))
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets

    def get_test_dataset(self):
        return self.test_datasets

    def get_train_dataset(self):
        return self.train_datasets


class CustomerDataset(Dataset):
    '''
    A custom Dataset class for client
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    '''

    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        if inputs.dtype == 'uint8':
            self.data = torch.tensor(inputs)
        else:
            self.data = torch.tensor(inputs).float()
        self.targets = torch.Tensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return (img, label)

    def __len__(self):
        return self.data.shape[0]


def get_default_data_transforms(name, train=True, verbose=False):
    name = name.lower()
    transforms_train = {
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    }
    transforms_eval = {
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    }

    return (transforms_train[name], transforms_eval[name])


def print_split(clients_split):
    n_labels = 10
    sum = 0
    print("Data split:")
    for i, client in enumerate(clients_split):
        split = np.sum(client[1].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
        print(" - Client {}: {}, sum = {}".format(i, split,split.sum()))
        sum += split.sum()
    print(f'sum = {sum}')


# shuffle two list together, so that shuffling operation won't destroy the one-to-one map between data and label
def together_shuffle(x_data, y_data):
    assert len(x_data) == len(y_data)
    randomize = np.arange(len(x_data))
    np.random.shuffle(randomize)
    x_data = np.array(x_data)[randomize]
    y_data = np.array(y_data)[randomize]
    return x_data, y_data


# input: data distribution config, train set and test set
# output: each client train set and test set
def split_data(data_distribution_config, n_clients, train_set):
    if data_distribution_config["iid"] == True:  # iid
        split = iid_split(n_clients, train_set)
    elif data_distribution_config["customize"] == False:  # Non-IID and auto generate(non customize)
        split = niid_dirichlet_split(n_clients, 1.0, train_set)
    elif data_distribution_config["customize"] == True:  # Non-IID and customize
        split = niid_customize_split(n_clients, train_set, data_distribution_config["cus_distribution"])
    return split


def get_global_data(test_set):
    x_test = test_set.data  # get data of test samples
    y_test = test_set.targets
    test_set = (x_test, y_test)
    return test_set


# split data uniformly
def iid_split(n_clients, train_set):
    num_train = len(train_set)  # get number of training samples
    x_train = train_set.data  # get data of training samples
    y_train = train_set.targets  # get label of training samples

    clients_sample_num = int(num_train / n_clients)  # the number of client samples
    x_train, y_train = together_shuffle(x_train, y_train)  # shuffle

    split = []  # data split, each element is a tuple (x_data, y_data)

    for i in range(n_clients):
        client_x_data = x_train[clients_sample_num * i: clients_sample_num * (i + 1)]  # get a slice of data
        client_y_data = y_train[clients_sample_num * i: clients_sample_num * (i + 1)]
        # print(client_y_data.shape)
        split += [(client_x_data, client_y_data)]

    print_split(split)
    return split


def niid_dirichlet_split(n_clients, alpha, train_set):
    '''
    Dirichlet distribution with parameter alpha, dividing the data index into n_clients subsets
    '''
    # total classes num
    x_train = train_set.data  # get data of training samples
    y_train = train_set.targets  # get label of training samples
    try:
        n_classes = y_train.max() + 1
    except:
        n_classes = np.max(y_train) + 1

    # shuffle
    x_train, y_train = together_shuffle(x_train, y_train)

    # [alpha] * n_clients is as follows：
    # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # Record the ratio of each client to each category
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    # Record the sample subscript corresponding to each category
    class_idcs = [np.argwhere(y_train == y).flatten()
                  for y in range(n_classes)]

    # Define an empty list as the final return value
    client_idcs = [[] for _ in range(n_clients)]
    # Record the indexes of N clients corresponding to the sample collection
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split, According to the proportion, the samples of category k are divided into N subsets
        # for i, idcs In order to traverse the index of the sample set corresponding to the i-th client
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    split = []

    for i in range(n_clients):
        client_x_data = x_train[client_idcs[i]]  # get a slice of data
        client_y_data = y_train[client_idcs[i]]
        client_x_data, client_y_data = together_shuffle(client_x_data, client_y_data)
        split += [(client_x_data, client_y_data)]

    print_split(split)
    return split


def niid_class_split(n_clients, train_set, distribution):
    # total classes num
    x_train = train_set.data  # get data of training samples
    y_train = train_set.targets  # get label of training samples
    if not isinstance(x_train, torch.Tensor):
        x_train=torch.tensor(x_train)
    if not isinstance(y_train, torch.Tensor):
        y_train=torch.tensor(y_train)
    n_classes = y_train.max() + 1

    # generate ratio matrix
    ratio_matrix = []
    for i in range(n_clients):
        shift = i  # the first type of data
        class_num = distribution[i]  # the number of class of i-th client
        class_list = [0 for j in range(
            n_classes)]  # the class list of the i-the client, like:[1,1,0,0,0] i.e. i-th client has 0,1 two class
        for j in range(class_num):
            class_list[(shift + j) % n_classes] = 1
        ratio_matrix.append(class_list)

    # get split
    split = niid_matrix_split(train_set, n_clients, ratio_matrix)
    return split


def ratio_matrix_to_num_matrix(labels, ratio_matrix):
    """
    :param labels: Labels for all data in this dataset
    :param ratio_matrix: the scale matrix of the data distribution to obtain
    :return: The actual data matrix num_matrix of the data distribution to obtain
    """
    # Get the labels for each label
    mask = np.unique(labels)
    mask = sorted(mask)

    # Get the number of data for each label
    labels_num = []
    labels = labels.cpu().numpy()
    for v in mask:
        labels_num.append(np.sum(labels == v))

    # Get the total number of proportions, and the data volume of a proportion
    ratio_sum = np.sum(ratio_matrix, axis=0)  # Get the total number of proportions of each labeled data
    # one_ratio_num = labels_num / ratio_sum  # the data volume of a proportion
    one_ratio_num = labels_num / np.max(ratio_sum) # 假设每个label数量相同，保证client有的label对应的sample数相同

    # get data number matrix
    num_matrix = []
    for i in range(len(ratio_matrix)):  # for each client
        client_data_num = []  # Amount of data per client, ist
        for j in range(len(ratio_sum)):  # for each class
            # data_num = one_ratio_num[j] * ratio_matrix[i][j]  # Calculate the amount of data of the jth class of the i-th client
            data_num = one_ratio_num[j] * ratio_matrix[i][j]
            client_data_num.append(data_num)
        num_matrix.append(client_data_num)

    num_matrix=np.round(num_matrix).astype(int)
    return num_matrix


def niid_matrix_split(train_set, n_clients, ratio_matrix, shuffle=True):
    data = train_set.data
    labels = train_set.targets
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    num_matrix = ratio_matrix_to_num_matrix(labels, ratio_matrix)
    n_labels = len(num_matrix[0])

    # sort for labels
    data_idcs = [[] for i in range(n_labels)]
    for j, label in enumerate(labels):
        data_idcs[label] += [j]  # data_idcs[i] Represents all index numbers of the data of the i-th category
    if shuffle:
        for idcs in data_idcs:
            np.random.shuffle(idcs)  # shuffle

    clients_split = []
    for i in range(n_clients):  # for each client
        client_idcs = []  # Store all index numbers of client i's data
        client_data_num = num_matrix[i]  # Get the number of data of each type of client i, client_data_num[c] indicates the number of data of type c of client i
        for c in range(n_labels):  # for each class
            if client_data_num[c] == 0:  # If the class requires 0 data, continue looping
                continue
            take = int(client_data_num[c])
            client_idcs += data_idcs[c][:take]
            data_idcs[c] = data_idcs[c][take:]

        client_x_data = data[client_idcs]  # get a slice of data
        client_y_data = labels[client_idcs]
        client_x_data, client_y_data = together_shuffle(client_x_data, client_y_data)
        clients_split += [(client_x_data, client_y_data)]
    
    print_split(clients_split)
    return clients_split


def niid_customize_split(n_clients, train_set, distribution):
    if type(distribution[0]) is int:
        # if the distribution is the number of class of each client
        # like: [2,2,2,2,2,2,2,2,2,2]
        # i.e. each client has two class
        split = niid_class_split(n_clients, train_set, distribution)
    else:
        # if the distribution is the ratio matrix
        split = niid_matrix_split(train_set, n_clients, distribution)
    return split


def get_dataset(dataset_name):
    if dataset_name == 'CIFAR10':
        return CIFAR10()


def get_data(dataset, n_clients, data_config):
    # dataset
    # load whole dataset
    dataset = get_dataset(dataset)
    train_set = dataset.get_train_dataset()  # get global training set
    split = split_data(data_config, n_clients, train_set)
    test_set = dataset.get_test_dataset()
    test_set = get_global_data(test_set)

    return split, test_set
