import os
import json
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import dirichlet
from torchvision import transforms
from torchvision.datasets import FashionMNIST, CIFAR10, MNIST
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, RobertaTokenizer


def partition_report(client_labels, save_path, n_class):
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
    report_pd['client'] = [str(i) for i in range(n_clients)]
    report_pd = report_pd.set_index('client')
    report_pd[col_names].plot.barh(stacked=True)
    plt.tight_layout()
    plt.xlabel('sample num')
    plt.savefig(save_path, dpi=2400)
    return report


def create_label_map(num_classes, num_groups):
    """
    创建标签映射，允许部分随机修改
    :param num_classes: 类别数量
    :param num_groups: 分组数量
    :return: 标签映射列表
    """
    if num_classes < num_groups:
        raise ValueError('num_groups should be smaller than num_classes')

    nums = list(np.arange(num_classes))
    label_maps = [nums.copy()]

    for i in range(num_groups - 1):
        t_map = [(label + i + 1) % num_classes for label in label_maps[0]]
        label_maps.append(t_map)

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

    grouped_train_data = []
    grouped_train_labels = []
    grouped_test_data = []
    grouped_test_labels = []

    # 将数据分成不重复的组
    idxs = np.array_split(np.random.permutation(len(train_data) ), num_groups)
    grouped_train_data = [train_data[idx] for idx in idxs]
    train_label_splits = [train_labels[idx] for idx in idxs]
    for i in range(num_groups):
        mapped_train_labels = np.array([label_maps[i][label] for label in train_label_splits[i]])  # 应用标签映射
        grouped_train_labels.append(mapped_train_labels)

    # # 每个概念组的数据一致
    # for i in range(num_groups):
    #     # 组织训练集
    #     random_indices = torch.randperm(len(train_data))
    #
    #     shuffled_data = copy.deepcopy(train_data[random_indices])
    #     grouped_train_data.append(shuffled_data)
    #
    #     shuffled_labels = copy.deepcopy(train_labels[random_indices])
    #     mapped_train_labels = np.array([label_maps[i][label] for label in shuffled_labels])  # 应用标签映射
    #     grouped_train_labels.append(mapped_train_labels)

    for i in range(num_groups):
        # 组织测试集
        grouped_test_data.append(copy.deepcopy(test_data[:2000]))

        mapped_test_labels = np.array([label_maps[i][label] for label in copy.deepcopy(test_labels[:2000])])
        grouped_test_labels.append(mapped_test_labels)

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


def split_data(dataset, num_concept_groups, num_label_groups, num_clients_per_group, alpha, same_py, save_dir):
    """
    双层分割数据集，并保存分割结果和参数。
    :param dataset: 数据集名称 ("Mnist", "FashionMnist", "Cifar10")
    :param num_concept_groups: 第一层概念偏移分组的数量
    :param num_label_groups: 第二层标签分布分组的数量 (Dirichlet 维度)
    :param num_clients_per_group: 每个标签分布组内的客户端数量
    :param alpha: 迪利克雷分布的参数
    :param same_py: 布尔值，指示是否所有概念组使用相同的标签分布比例
    :return: 分组后的客户端训练数据、客户端训练标签、全局测试数据、全局测试标签和保存路径
    """
    # 加载数据集
    data_path = "./data"
    if dataset == "Mnist":
        num_classes = 10
        train_x = MNIST(root=data_path, train=True, download=True).data
        train_y = np.array(MNIST(root=data_path, train=True, download=True).targets)
        test_x = MNIST(root=data_path, train=False, download=True).data
        test_y = np.array(MNIST(root=data_path, train=False, download=True).targets)
    elif dataset == "FashionMnist":
        num_classes = 10
        train_x = FashionMNIST(root=data_path, train=True, download=True).data
        train_y = np.array(FashionMNIST(root=data_path, train=True, download=True).targets)
        test_x = FashionMNIST(root=data_path, train=False, download=True).data
        test_y = np.array(FashionMNIST(root=data_path, train=False, download=True).targets)
    elif dataset == "Cifar10":
        num_classes = 10
        train_x = CIFAR10(root=data_path, train=True, download=True).data
        train_y = np.array(CIFAR10(root=data_path, train=True, download=True).targets)
        test_x = CIFAR10(root=data_path, train=False, download=True).data
        test_y = np.array(CIFAR10(root=data_path, train=False, download=True).targets)
    elif dataset == "text-roberta" or dataset == "text-distill":
        num_classes = 10
        train_x = np.load(data_path + '/yahoo/yahoo_train_x_sample.npy')
        train_y = np.load(data_path + '/yahoo/yahoo_train_y_sample.npy')
        test_x = np.load(data_path + '/yahoo/yahoo_test_x_sample.npy')
        test_y = np.load(data_path + '/yahoo/yahoo_test_y_sample.npy')
    else:
        raise ValueError("Unsupported dataset")

    # 一次分组
    y_maps = create_label_map(num_classes, num_concept_groups)
    g_train_x, g_train_y, g_test_x, g_test_y = split_dataset_by_concept_shift(train_x, train_y, test_x, test_y, y_maps)

    # 二次分组
    all_client_x = []
    all_client_y = []
    if same_py:
        proportions = []
        for class_idx in range(num_classes):
            proportions.append(dirichlet([alpha] * num_label_groups).rvs(size=1).flatten())
    else:
        proportions = None

    for per_g_x, per_g_y in zip(g_train_x, g_train_y):
        client_data, client_labels = split_dataset_by_label_distribution(per_g_x, per_g_y, num_label_groups,
                                                                         num_clients_per_group, alpha, proportions)
        all_client_x.extend(client_data)
        all_client_y.extend(client_labels)

    # 打印每个客户端的数据形状和标签分布概览
    print("\n--- 客户端数据概览 ---")
    for i in range(len(all_client_x)):
        print(f"Group {i}: Labels shape={np.bincount(all_client_y[i])}")

    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n数据集将保存到: {os.path.abspath(save_dir)}")

    # 保存客户端训练数据和标签
    client_data_dir = os.path.join(save_dir, 'client_train_data')
    os.makedirs(client_data_dir, exist_ok=True)
    print(f"正在保存客户端训练数据到: {client_data_dir}")
    for i, (data, labels) in enumerate(zip(all_client_x, all_client_y)):
        np.save(os.path.join(client_data_dir, f'client_{i}_data.npy'), data)
        np.save(os.path.join(client_data_dir, f'client_{i}_labels.npy'), labels)

    # 保存全局测试数据和标签 (每个概念组对应一个测试集)
    global_test_dir = os.path.join(save_dir, 'global_test_data')
    os.makedirs(global_test_dir, exist_ok=True)
    print(f"正在保存全局测试数据到: {global_test_dir}")
    for i, (data, labels) in enumerate(zip(g_test_x, g_test_y)):
        np.save(os.path.join(global_test_dir, f'global_test_group_{i}_data.npy'), data)
        np.save(os.path.join(global_test_dir, f'global_test_group_{i}_labels.npy'), labels)

    # 将分割参数保存到 JSON 文件
    partition_params = {
        "dataset_name": dataset,
        "num_concept_groups": num_concept_groups,
        "num_label_groups": num_label_groups,
        "num_clients_per_group": num_clients_per_group,
        "alpha": alpha,
        "same_py": same_py,
        "num_classes": num_classes
    }
    params_json_path = os.path.join(save_dir, 'partition_params.json')
    with open(params_json_path, 'w') as f:
        json.dump(partition_params, f, indent=4)
    print(f"分割参数已保存到: {params_json_path}")

    # 生成并保存客户端标签分布报告图
    report_image_path = os.path.join(save_dir, 'label_distribution.png')
    report = partition_report(all_client_y, report_image_path, n_class=num_classes)
    print(f"客户端标签分布报告图已保存到: {report_image_path}")

    return all_client_x, all_client_y, g_test_x, g_test_y


def _check_data_integrity(save_dir):
    """
    检查指定目录下的数据文件完整性。
    验证是否存在参数文件、客户端训练数据和全局测试数据，
    并确认文件数量与参数文件中记录的期望值一致。

    :param save_dir: 数据保存的根目录路径。
    :return: (is_valid, params, error_message)
             is_valid: 布尔值，表示数据是否完整。
             params: 如果完整，返回加载的参数字典；否则为 None。
             error_message: 如果不完整，返回错误信息；否则为“数据完整性检查通过。”
    """
    if not os.path.exists(save_dir):
        return False, None, f"错误：保存目录 '{save_dir}' 不存在。"

    params_json_path = os.path.join(save_dir, 'partition_params.json')
    if not os.path.exists(params_json_path):
        return False, None, f"错误：参数文件 '{params_json_path}' 不存在，无法验证数据结构。请确保该文件已保存。"

    try:
        with open(params_json_path, 'r') as f:
            params = json.load(f)
    except json.JSONDecodeError:
        return False, None, f"错误：参数文件 '{params_json_path}' 格式不正确，无法解析。"

    # 验证关键参数是否存在
    required_params = ["num_concept_groups", "num_label_groups", "num_clients_per_group", "num_classes"]
    for p in required_params:
        if p not in params:
            return False, None, f"错误：参数文件 '{params_json_path}' 缺少关键参数 '{p}'。"

    num_concept_groups = params["num_concept_groups"]
    num_label_groups = params["num_label_groups"]
    num_clients_per_group = params["num_clients_per_group"]
    expected_total_clients = num_concept_groups * num_label_groups * num_clients_per_group

    print(f"参数文件加载成功。预期客户端总数: {expected_total_clients}, 概念组数: {num_concept_groups}")

    # 检查客户端训练数据完整性
    client_train_data_dir = os.path.join(save_dir, 'client_train_data')
    if not os.path.exists(client_train_data_dir):
        return False, None, f"错误：客户端训练数据目录 '{client_train_data_dir}' 不存在。"

    for i in range(expected_total_clients):
        data_path = os.path.join(client_train_data_dir, f'client_{i}_data.npy')
        labels_path = os.path.join(client_train_data_dir, f'client_{i}_labels.npy')
        if not os.path.exists(data_path):
            return False, None, f"错误：客户端 {i} 的训练数据文件 '{data_path}' 不存在。"
        if not os.path.exists(labels_path):
            return False, None, f"错误：客户端 {i} 的训练标签文件 '{labels_path}' 不存在。"
    print(f"客户端训练数据完整性检查通过，预期并找到 {expected_total_clients} 个客户端的数据和标签文件。")


    # 检查全局测试数据完整性
    global_test_data_dir = os.path.join(save_dir, 'global_test_data')
    if not os.path.exists(global_test_data_dir):
        return False, None, f"错误：全局测试数据目录 '{global_test_data_dir}' 不存在。"

    for i in range(num_concept_groups): # 每个概念组有一个全局测试集
        data_path = os.path.join(global_test_data_dir, f'global_test_group_{i}_data.npy')
        labels_path = os.path.join(global_test_data_dir, f'global_test_group_{i}_labels.npy')
        if not os.path.exists(data_path):
            return False, None, f"错误：全局测试组 {i} 的数据文件 '{data_path}' 不存在。"
        if not os.path.exists(labels_path):
            return False, None, f"错误：全局测试组 {i} 的标签文件 '{labels_path}' 不存在。"
    print(f"全局测试数据完整性检查通过，预期并找到 {num_concept_groups} 个概念组的测试集数据和标签文件。")

    return True, params, "数据完整性检查通过。"


def get_data(save_dir):
    """
    从指定目录加载所有客户端训练集和全局测试集数据。
    在加载之前会进行数据完整性检查。

    :param save_dir: 数据保存的根目录路径。
    :return: (all_client_train_x, all_client_train_y, all_global_test_x, all_global_test_y, loaded_params)
             如果完整性检查失败，则返回 ([], [], [], [], None)。
    """
    print(f"开始从 '{save_dir}' 目录加载数据...")
    is_valid, params, message = _check_data_integrity(save_dir)
    print(message)

    if not is_valid:
        print("数据完整性检查失败，停止加载。")
        return [], [], [], [], None

    loaded_params = params
    num_concept_groups = loaded_params["num_concept_groups"]
    expected_total_clients = num_concept_groups * loaded_params["num_label_groups"] * loaded_params["num_clients_per_group"]


    all_client_train_x = []
    all_client_train_y = []
    client_train_data_dir = os.path.join(save_dir, 'client_train_data')

    print("\n--- 正在加载客户端训练数据 ---")
    for i in range(expected_total_clients):
        data_path = os.path.join(client_train_data_dir, f'client_{i}_data.npy')
        labels_path = os.path.join(client_train_data_dir, f'client_{i}_labels.npy')
        client_data = np.load(data_path, allow_pickle=True)
        client_labels = np.load(labels_path, allow_pickle=True)
        all_client_train_x.append(client_data)
        all_client_train_y.append(client_labels)
        # print(f"已加载客户端 {i}: 数据形状={client_data.shape}, 标签形状={client_labels.shape}") # 可以在调试时取消注释
    print(f"成功加载 {len(all_client_train_x)} 个客户端的训练数据。")

    all_global_test_x = []
    all_global_test_y = []
    global_test_data_dir = os.path.join(save_dir, 'global_test_data')

    print("\n--- 正在加载全局测试数据 ---")
    for i in range(num_concept_groups):
        data_path = os.path.join(global_test_data_dir, f'global_test_group_{i}_data.npy')
        labels_path = os.path.join(global_test_data_dir, f'global_test_group_{i}_labels.npy')
        test_data = np.load(data_path, allow_pickle=True)
        test_labels = np.load(labels_path, allow_pickle=True)
        all_global_test_x.append(test_data)
        all_global_test_y.append(test_labels)
        # print(f"已加载全局测试组 {i}: 数据形状={test_data.shape}, 标签形状={test_labels.shape}") # 可以在调试时取消注释
    print(f"成功加载 {len(all_global_test_x)} 个全局测试集。")

    print("\n所有数据加载完成。")
    return all_client_train_x, all_client_train_y, all_global_test_x, all_global_test_y


class ImgTransform:
    """
    对每个样本生成 K 个随机增强视图
    不消耗额外隐私预算
    """
    def __init__(self, mean, std, device):
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, img):
        return self.transform(img).to(self.device)


class ImgDataset(Dataset):
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


class TextDataset(Dataset):
    """
    用于处理联邦学习分割后的文本数据集
    """
    def __init__(self, texts, labels, tokenizer, device, max_length=256):
        # texts 和 labels 是来自 split_data 的 numpy 数组
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 取出单个文本和标签 (强制转换 numpy string 为 python str)
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # 动态 Tokenization
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        

        sample = {
            'input_ids': encoding['input_ids'].flatten().to(self.device),
            'attention_mask': encoding['attention_mask'].flatten().to(self.device)
        }
        label = torch.tensor(label, dtype=torch.long)
        # 返回与之前训练代码兼容的字典格式
        return sample, label


def get_dataloader(dataset, groups1, groups2, clients_per_group, alpha, same_py, batch_ratio, device):
    # 1. prepare data
    data_save_dir = os.path.join(
        './saved_datasets',
        f"{dataset}_g1({groups1})_g2({groups2})_inner({clients_per_group})_alpha({alpha})"
    )

    if not os.path.exists(data_save_dir):
        client_x, client_y, g_test_x, g_test_y = split_data(dataset=dataset, num_concept_groups=groups1,
                                                            num_label_groups=groups2,
                                                            num_clients_per_group=clients_per_group,
                                                            alpha=alpha, same_py=same_py,
                                                            save_dir=data_save_dir)
    else:
        client_x, client_y, g_test_x, g_test_y = get_data(data_save_dir)

    # 2. prepare data loader
    client_train_loader = []
    group_test_loader = []

    is_text = False
    train_transform = None
    test_transform = None
    tokenizer = None

    if dataset == 'Cifar10':
        mean, std, size = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 32
        train_transform = ImgTransform(mean=mean, std=std, device=device)
        test_transform = ImgTransform(mean=mean, std=std, device=device)

    elif dataset == 'FashionMnist':
        mean, std, size = (0.5,), (0.5,), 28
        train_transform = ImgTransform(mean=mean, std=std, device=device)
        test_transform = ImgTransform(mean=mean, std=std, device=device)

    elif dataset == 'Mnist':
        mean, std, size = (0.5,), (0.5,), 28
        train_transform = ImgTransform(mean=mean, std=std, device=device)
        test_transform = ImgTransform(mean=mean, std=std, device=device)

    elif dataset == 'text-roberta':
        is_text = True
        tokenizer = RobertaTokenizer.from_pretrained('./pretrained/RobertaModel')

    elif dataset == 'text-distill':
        is_text = True
        tokenizer = DistilBertTokenizer.from_pretrained('./pretrained/DistilBertModel')

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    def build_dataset(data, label, is_train=True):
        """根据配置动态构建对应的 Dataset，消除内部的 if-else"""
        if is_text:
            return TextDataset(data, label, tokenizer=tokenizer, device=device, max_length=256)

        # 图像数据集处理
        if not is_train and isinstance(data, torch.Tensor):
            data = data.numpy()

        transform = train_transform if is_train else test_transform
        return ImgDataset(data, label, transform=transform)

    for data, label in zip(client_x, client_y):
        client_dataset = build_dataset(data, label, is_train=True)

        # 动态计算 Batch Size
        if 0 < batch_ratio < 1:
            batch_size = max(int(batch_ratio * len(client_dataset)), 16)
            print(f'batch_size: {batch_size}')
        else:
            batch_size = 32

        client_train_loader.append(DataLoader(client_dataset, shuffle=True, batch_size=batch_size, drop_last=True))

    # 生成组测试集 Loader
    for data, label in zip(g_test_x, g_test_y):
        test_dataset = build_dataset(data, label, is_train=False)

        group_test_loader.append(DataLoader(test_dataset, shuffle=False, batch_size=64))

    return client_train_loader, group_test_loader
