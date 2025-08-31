import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import FashionMNIST, CIFAR10, MNIST
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.stats import dirichlet
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from torchvision.transforms import ToTensor
import copy
import torch
import random
from PIL import Image


class ClientDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        :param data: 客户端数据 (numpy array 或 tensor)
        :param labels: 客户端标签 (numpy array 或 tensor)
        :param transform: 数据增强或预处理函数
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


def partition_report(client_labels, save_path, n_class=10):
    """
    可视化客户端数据类分布
    :param client_labels: 客户端标签数据列表 (每个元素是一个 numpy array)
    """
    if len(client_labels) > 50:
        client_labels = client_labels[:50]
    n_clients = len(client_labels)
    col_names = [f"class{i}" for i in range(n_class)]
    report = []
    for label in client_labels:
        # 计算每个类别的样本数量
        category_counts = [len(label[label == i]) for i in range(n_class)]
        report.append(category_counts)
    report_pd = pd.DataFrame(np.array(report), columns=col_names)
    report_pd['client'] = ['client_' + str(i) for i in range(n_clients)]
    report_pd = report_pd.set_index('client')
    report_pd[col_names].plot.barh(stacked=True)
    plt.tight_layout()
    plt.xlabel('sample num')
    plt.savefig(save_path, dpi=2400)
    # plt.show()
    return report


def split_client_data(client_data, client_labels, transform, num_clients_per_group, batch_ratio=0.01, train_ratio=0.9):
    """
    将客户端数据划分为训练集和验证集
    :param client_data: 客户端数据列表 (每个元素是一个 numpy array 或 tensor)
    :param client_labels: 客户端标签列表 (每个元素是一个 numpy array 或 tensor)
    :param train_ratio: 训练集比例 (默认 90%)
    :return: 训练集和验证集的 Dataset 列表
    """
    train_loader = []
    val_loader = []
    id = 0
    for data, labels in zip(client_data, client_labels):
        # 构造P(x)异构
        # rot = id % num_clients_per_group % 4
        # id += 1
        # data = np.rot90(data, k=rot, axes=(1, 2))
        # 创建客户端 Dataset
        client_dataset = ClientDataset(copy.deepcopy(data), labels, transform=transform)

        # 计算训练集和验证集大小
        train_size = int(len(client_dataset) * train_ratio)
        val_size = len(client_dataset) - train_size

        # 随机划分数据集
        train_dataset, val_dataset = random_split(client_dataset, [train_size, val_size])
        # t_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        # v_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
        if 0 < batch_ratio < 1:
            batch_size = max(int(batch_ratio * len(train_dataset)), 8)
        else:
            batch_size = 64
        t_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=0, pin_memory=True)
        v_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=0, pin_memory=True)

        train_loader.append(t_loader)
        val_loader.append(v_loader)

    return train_loader, val_loader


def rotate_array(arr, times):
    rotated_arrays = []
    for _ in range(times):
        arr = [arr[-1]] + arr[:-1]  # 将末尾元素移到前面
        rotated_arrays.append(arr.copy())  # 保存当前结果
    return rotated_arrays


def create_label_map(num_classes, num_groups, overlap_ratio):
    """
    创建标签映射，允许部分随机修改
    :param num_classes: 类别数量
    :param num_groups: 分组数量
    :param overlap_ratio: 标签映射的重叠比例
    :return: 标签映射列表
    """
    nums = list(np.arange(num_classes))
    label_maps = [nums.copy()]


    for i in range(num_groups - 1):
        t_map = [(label + i + 1) % 10 for label in label_maps[0]]
        label_maps.append(t_map)

    # for _ in range(num_groups - 1):
    #     ori_map = np.arange(num_classes)
    #     shuffled_map = list(np.random.permutation(ori_map))
    #     label_maps.append(shuffled_map)

    if overlap_ratio:
        label_maps = [
            [1, 0, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 3, 2, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 5, 4, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 7, 6, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 9, 8],
        ]

    label_maps = np.array(label_maps)
    return label_maps


def split_dataset_by_concept_shift(train_data, train_labels, test_data, test_labels, label_maps):
    """
    根据概念偏移进行第一层分组
    :param data: 数据集
    :param labels: 标签
    :param label_maps: 标签映射列表
    :return: 分组后的数据集和标签
    """
    num_groups = len(label_maps)
    # 划分训练集
    grouped_train_data = []
    grouped_train_labels = []

    # 将数据分成不重复的组
    idxs = np.array_split(range(len(train_data)), num_groups)
    train_data_splits = [train_data[idx] for idx in idxs]
    train_label_splits = [train_labels[idx] for idx in idxs]

    # train_data_splits = [copy.deepcopy(train_data) for _ in range(num_groups)]
    # train_label_splits = [copy.deepcopy(train_labels) for _ in range(num_groups)]

    for i in range(num_groups):
        # 应用标签映射
        mapped_labels = np.array([label_maps[i][label] for label in train_label_splits[i]])
        grouped_train_data.append(train_data_splits[i])
        grouped_train_labels.append(mapped_labels)

    # 划分验证集
    grouped_test_data = []
    grouped_test_labels = []
    test_data_splits = [copy.deepcopy(test_data[:2000]) for _ in range(num_groups)]
    test_label_splits = [copy.deepcopy(test_labels[:2000]) for _ in range(num_groups)]

    for i in range(num_groups):
        # 应用标签映射
        mapped_labels = np.array([label_maps[i][label] for label in test_label_splits[i]])
        grouped_test_data.append(test_data_splits[i])
        grouped_test_labels.append(mapped_labels)

    return grouped_train_data, grouped_train_labels, grouped_test_data, grouped_test_labels


def split_dataset_by_label_distribution(data, labels, num_label_groups, num_clients_per_group, alpha, proportions=None):
    """
    根据类分布偏移进行第二层分组
    :param data: 数据集
    :param labels: 标签
    :param num_label_groups: 分割数量
    :param num_clients_per_group: 组内客户端数量
    :param alpha: 迪利克雷分布的参数迪利克雷分布
    :param proportions: 预定义的
    :return: 分组后的数据集和标签
    """
    num_classes = len(np.unique(labels))
    num_clients = num_label_groups * num_clients_per_group
    client_data = [[] for _ in range(num_clients)]
    client_labels = [[] for _ in range(num_clients)]

    # 为每个类别生成迪利克雷分布
    for class_idx in range(num_classes):
        if proportions is None:
            proportion = dirichlet([alpha] * num_label_groups).rvs(size=1).flatten()
        else:
            proportion = proportions[class_idx]
        class_indices = np.where(labels == class_idx)[0]
        splits = (proportion * len(class_indices)).astype(int)

        # 将数据分配给客户端
        g_start = 0
        for idx1, split in enumerate(splits):
            g_end = g_start + split
            # 组内分割
            for idx2 in range(num_clients_per_group):
                start = int(g_start + idx2 * (split/num_clients_per_group))
                end = min(int(start + (split/num_clients_per_group)), g_end)
                idx = idx1 * num_clients_per_group + idx2
                client_data[idx].extend(data[class_indices[start:end]])
                client_labels[idx].extend(labels[class_indices[start:end]])
            g_start = g_end

            # 将列表转换为 numpy 数组
    client_data = [np.array(client_data[i]) for i in range(num_clients)]
    client_labels = [np.array(client_labels[i]) for i in range(num_clients)]

    return client_data, client_labels


def split_dataset(dataset_name, num_concept_groups, num_label_groups, num_clients_per_group, alpha, overlap_ratio, same_py, batch_ratio):
    """
    双层分割数据集
    :param dataset_name: 数据集名称
    :param num_concept_groups: 第一层分组数量
    :param num_label_groups: 第二层分组数量
    :param num_clients_per_group: 组内客户端数量
    :param alpha: 迪利克雷分布的参数
    :param overlap_ratio: 标签映射的重叠比例
    :return: 分组后的数据集和标签
    """
    # 加载数据集
    if dataset_name == "Mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = MNIST(root="./data", train=True, download=True)
        test_dataset = MNIST(root="./data", train=False, download=True)
    elif dataset_name == "FashionMnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = FashionMNIST(root="./data", train=True, download=True)
        test_dataset = FashionMNIST(root="./data", train=False, download=True)
    elif dataset_name == "Cifar10":
        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root="./data", train=True, download=True)
        test_dataset = CIFAR10(root="./data", train=False, download=True)
    else:
        raise ValueError("Unsupported dataset")
    train_data, train_labels = train_dataset.data, np.array(train_dataset.targets)
    test_data, test_labels = test_dataset.data, np.array(test_dataset.targets)
    num_classes = len(np.unique(train_labels))
    # 一次分组
    label_maps = create_label_map(num_classes, num_concept_groups, overlap_ratio)
    grouped_train_data, grouped_train_labels, grouped_test_data, grouped_test_labels = split_dataset_by_concept_shift(
        train_data, train_labels, test_data, test_labels, label_maps)
    # 二次分组
    all_client_data = []
    all_client_labels = []
    # 令不同pyx组内的py异构一致
    if same_py:
        proportions = []
        for class_idx in range(num_classes):
            proportions.append(dirichlet([alpha] * num_label_groups).rvs(size=1).flatten())
    else:
        proportions = None

    for data_group, labels_group in zip(grouped_train_data, grouped_train_labels):
        client_data, client_labels = split_dataset_by_label_distribution(data_group, labels_group, num_label_groups,
                                                                         num_clients_per_group, alpha, proportions)
        all_client_data.extend(client_data)
        all_client_labels.extend(client_labels)

    for i in range(len(all_client_data)):
        print(f"Group {i}: Data shape={all_client_data[i].shape}, Labels shape={np.bincount(all_client_labels[i])}")
    save_path = ('./imgs/' + dataset_name +
                 '_g1' + '(' + str(num_concept_groups) + ')' +
                 '_g2' + '(' + str(num_label_groups) + ')' +
                 '_inner' + '(' + str(num_clients_per_group) + ')' + '.png')
    report = partition_report(all_client_labels, save_path, n_class=10)
    # 构建客户端训练集和服务端测试集
    train_loader, val_loader = split_client_data(all_client_data, all_client_labels, transform, num_clients_per_group,
                                                 batch_ratio, train_ratio=0.9)
    test_loader = []
    for data, label in zip(grouped_test_data, grouped_test_labels):
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        dataset = ClientDataset(data, label, transform=transform)
        # test_loader.append(DataLoader(dataset, shuffle=False, batch_size=batch_size))
        test_loader.append(DataLoader(dataset, shuffle=False, batch_size=64))

    return train_loader, val_loader, test_loader, label_maps, report, (all_client_data, all_client_labels, transform)
