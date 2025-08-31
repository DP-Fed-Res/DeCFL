# fedrc_parallel.py

import os
import copy
import time
import abc
import shutil
import random
import tempfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import default_rng
from scipy.stats import dirichlet
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import FashionMNIST, CIFAR10, MNIST
from torchvision import transforms
import matplotlib.pyplot as plt

from customopacus import PrivacyEngine
from options import args_parser


# =========================
# 通用工具
# =========================

def set_random(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================
# 模型定义
# =========================

class Cifar10Cnn(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Cifar10Cnn, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(5, 5), padding=2),
            nn.GroupNorm(4, 16),
            nn.ReLU()
        )
        self.pool1 = nn.AvgPool2d(2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            nn.GroupNorm(4, 32),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.GroupNorm(4, 64),
            nn.ReLU()
        )
        self.pool2 = nn.AvgPool2d(2)
        self.fc = nn.Linear(6 * 6 * 64, num_classes)

    def forward(self, x):
        out = self.pool1(self.layer1(x))
        out = self.pool2(self.layer3(self.layer2(out)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def create_keras_model(dataset):
    if dataset == 'Cifar10':
        model = Cifar10Cnn(in_channels=3, num_classes=10)
    else:
        raise NotImplementedError(f'Unsupported dataset: {dataset}')
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)
    return model


# =========================
# 数据集与划分
# =========================

class ClientDataset(Dataset):
    def __init__(self, data, labels, transform=None):
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
    if len(client_labels) > 50:
        client_labels = client_labels[:50]
    n_clients = len(client_labels)
    col_names = [f"class{i}" for i in range(n_class)]
    report = []
    for label in client_labels:
        category_counts = [len(label[label == i]) for i in range(n_class)]
        report.append(category_counts)
    report_pd = pd.DataFrame(np.array(report), columns=col_names)
    report_pd['client'] = ['client_' + str(i) for i in range(n_clients)]
    report_pd = report_pd.set_index('client')
    report_pd[col_names].plot.barh(stacked=True)
    plt.tight_layout()
    plt.xlabel('sample num')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600)
    plt.close()
    return report


def split_client_data(client_data, client_labels, transform, num_clients_per_group, batch_ratio=0.01, train_ratio=0.9):
    train_loader = []
    val_loader = []
    for data, labels in zip(client_data, client_labels):
        client_dataset = ClientDataset(copy.deepcopy(data), labels, transform=transform)
        train_size = int(len(client_dataset) * train_ratio)
        val_size = len(client_dataset) - train_size
        train_dataset, val_dataset = random_split(client_dataset, [train_size, val_size])
        if 0 < batch_ratio < 1:
            batch_size = max(int(batch_ratio * len(train_dataset)), 8)
        else:
            batch_size = 64
        # 为并行稳定性，起步 num_workers=0
        t_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=0, pin_memory=True)
        v_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=0, pin_memory=True)
        train_loader.append(t_loader)
        val_loader.append(v_loader)
    return train_loader, val_loader


def create_label_map(num_classes, num_groups, overlap_ratio):
    nums = list(np.arange(num_classes))
    label_maps = [nums.copy()]
    if num_groups <= 10:
        for i in range(num_groups - 1):
            t_map = [(label + i + 1) % 10 for label in label_maps[0]]
            label_maps.append(t_map)
    else:
        for _ in range(num_groups - 1):
            ori_map = np.arange(num_classes)
            shuffled_map = list(np.random.permutation(ori_map))
            label_maps.append(shuffled_map)
    label_maps = np.array(label_maps)
    return label_maps


def split_dataset_by_concept_shift(train_data, train_labels, test_data, test_labels, label_maps):
    num_groups = len(label_maps)
    grouped_train_data = []
    grouped_train_labels = []

    idxs = np.array_split(range(len(train_data)), num_groups)
    train_data_splits = [train_data[idx] for idx in idxs]
    train_label_splits = [train_labels[idx] for idx in idxs]

    for i in range(num_groups):
        mapped_labels = np.array([label_maps[i][label] for label in train_label_splits[i]])
        grouped_train_data.append(train_data_splits[i])
        grouped_train_labels.append(mapped_labels)

    grouped_test_data = []
    grouped_test_labels = []
    test_data_splits = [copy.deepcopy(test_data[:2000]) for _ in range(num_groups)]
    test_label_splits = [copy.deepcopy(test_labels[:2000]) for _ in range(num_groups)]

    for i in range(num_groups):
        mapped_labels = np.array([label_maps[i][label] for label in test_label_splits[i]])
        grouped_test_data.append(test_data_splits[i])
        grouped_test_labels.append(mapped_labels)

    return grouped_train_data, grouped_train_labels, grouped_test_data, grouped_test_labels


def split_dataset_by_label_distribution(data, labels, num_label_groups, num_clients_per_group, alpha, proportions=None):
    num_classes = len(np.unique(labels))
    num_clients = num_label_groups * num_clients_per_group
    client_data = [[] for _ in range(num_clients)]
    client_labels = [[] for _ in range(num_clients)]

    for class_idx in range(num_classes):
        if proportions is None:
            proportion = dirichlet([alpha] * num_label_groups).rvs(size=1).flatten()
        else:
            proportion = proportions[class_idx]
        class_indices = np.where(labels == class_idx)[0]
        splits = (proportion * len(class_indices)).astype(int)

        g_start = 0
        for idx1, split in enumerate(splits):
            g_end = g_start + split
            for idx2 in range(num_clients_per_group):
                start = int(g_start + idx2 * (split/num_clients_per_group))
                end = min(int(start + (split/num_clients_per_group)), g_end)
                idx = idx1 * num_clients_per_group + idx2
                client_data[idx].extend(data[class_indices[start:end]])
                client_labels[idx].extend(labels[class_indices[start:end]])
            g_start = g_end

    client_data = [np.array(client_data[i]) for i in range(num_clients)]
    client_labels = [np.array(client_labels[i]) for i in range(num_clients)]
    return client_data, client_labels


def split_dataset(dataset_name, num_concept_groups, num_label_groups, num_clients_per_group, alpha, overlap_ratio, same_py, batch_ratio):
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

    label_maps = create_label_map(num_classes, num_concept_groups, overlap_ratio)
    grouped_train_data, grouped_train_labels, grouped_test_data, grouped_test_labels = split_dataset_by_concept_shift(
        train_data, train_labels, test_data, test_labels, label_maps)

    all_client_data = []
    all_client_labels = []
    if same_py:
        proportions = []
        for class_idx in range(num_classes):
            proportions.append(dirichlet([alpha] * num_label_groups).rvs(size=1).flatten())
    else:
        proportions = None

    for data_group, labels_group in zip(grouped_train_data, grouped_train_labels):
        client_data, client_labels = split_dataset_by_label_distribution(
            data_group, labels_group, num_label_groups, num_clients_per_group, alpha, proportions)
        all_client_data.extend(client_data)
        all_client_labels.extend(client_labels)

    for i in range(len(all_client_data)):
        binc = np.bincount(all_client_labels[i], minlength=10)
        print(f"Group {i}: Data shape={all_client_data[i].shape}, Labels shape={binc}")

    save_path = ('./imgs/' + dataset_name +
                 '_g1' + '(' + str(num_concept_groups) + ')' +
                 '_g2' + '(' + str(num_label_groups) + ')' +
                 '_inner' + '(' + str(num_clients_per_group) + ')' + '.png')
    report = partition_report(all_client_labels, save_path, n_class=10)

    # 注意：此函数后续并不直接把 DataLoader 传给子进程，而是用于主进程评估/调试
    train_loader, val_loader = split_client_data(all_client_data, all_client_labels, transform, num_clients_per_group,
                                                 batch_ratio, train_ratio=0.9)

    # test loaders（服务器评估用，主进程）
    test_loader = []
    for data, label in zip(grouped_test_data, grouped_test_labels):
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        dataset = ClientDataset(data, label, transform=transform)
        test_loader.append(DataLoader(dataset, shuffle=False, batch_size=64, num_workers=0, pin_memory=True))

    return train_loader, val_loader, test_loader, label_maps, report, (all_client_data, all_client_labels, transform)


# =========================
# 训练配置与FL配置
# =========================

def get_train_config(data):
    config = {
        'Mnist': {
            'lr_mode': 'SGD',
            'local_steps': 20,
            'batch_ratio': 0.02,
            'lr': 0.1,
            'lr_decay': 0.998,
            'noise': 1.0,
            'noise_decay': 1.0,
            'clip': 1.0,
        },
        'FashionMnist': {
            'lr_mode': 'SGD',
            'local_steps': 20,
            'batch_ratio': 0.02,
            'lr': 0.1,
            'lr_decay': 0.998,
            'noise': 1.0,
            'noise_decay': 1.0,
            'clip': 1.0,
        },
        'Cifar10': {
            'lr_mode': 'SGD',
            'local_steps': 20,
            'batch_ratio': 0.02,
            'lr': 0.1,
            'lr_decay': 0.998,
            'noise': 1.0,
            'noise_decay': 1.0,
            'clip': 1.0,
        }
    }
    return config.get(data, None)


def get_fl_config(dataset):
    args = args_parser()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.dataset = dataset
    args.rounds = 1
    args.k = 400
    args.groups1 = 5
    args.groups2 = 5
    args.clients_per_group = 1
    args.K = args.groups1
    args.n_clients = args.groups1 * args.groups2 * args.clients_per_group
    args.alpha = 1.0
    args.overlap_ratio = 0
    args.same_py = True
    args.prior_cls = [i // int(args.n_clients/args.K) for i in range(args.n_clients)]
    print(args.prior_cls)
    args.primary_aggregation = True
    args.train_config = get_train_config(args.dataset)

    args.ckpt_dir = ('./ckpt/group/client' + str(args.n_clients) +
                     '_g1' + '(' + str(args.groups1) + ')' +
                     '_g2' + '(' + str(args.groups2) + ')' +
                     '_split' + '(' + args.split + str(args.alpha) + ')' +
                     '_noise' + '(' + str(args.train_config.get('noise')) + ')' +
                     args.dataset)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    else:
        shutil.rmtree(args.ckpt_dir)
        os.makedirs(args.ckpt_dir)

    # 其他参数兜底
    if not hasattr(args, 'seed'):
        args.seed = 2024
    if not hasattr(args, 'sampling_rate'):
        args.sampling_rate = 1.0

    return args


# =========================
# 客户端与服务器
# =========================

# 续写自上文：fedrc_parallel.py

import warnings

class FedRCClientLite(object):
    """
    子进程内使用的精简客户端，避免pickle主进程的复杂对象。
    仅依赖：train_loader / val_loader / device / train_config
    """
    def __init__(self, train_loader, val_loader, device, train_config):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_steps = train_config.get('local_steps')
        self.lr = train_config.get('lr')
        self.grad_clip = train_config.get('clip')
        self.noise_multiplier = train_config.get('noise')
        self.privacy_engine = PrivacyEngine(accountant='prv')
        self.criterion = nn.CrossEntropyLoss()

    def len(self):
        return len(self.train_loader.dataset)

    @torch.no_grad()
    def estimate_gamma_omega_for_client(self, models: List[nn.Module], num_classes: int = 10, max_val_batches: int = 10):
        device = self.device
        K = len(models)
        omega = torch.full((K,), 1.0 / K, device=device)
        C_yk = torch.ones(K, num_classes, device=device)
        gamma_sums = torch.zeros(K, device=device)
        sample_count = 0

        gammas_batches: List[torch.Tensor] = []
        batches = 0
        for batch in self.val_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            B = x.size(0)
            logits_list = [models[k](x) for k in range(K)]
            losses_per_k = [F.cross_entropy(logits_list[k], y, reduction='none') for k in range(K)]
            losses = torch.stack(losses_per_k, dim=1)  # [B,K]
            exp_neg_loss = torch.exp(-losses)
            C_per_sample = torch.stack([C_yk[:, y[b]] for b in range(B)], dim=0)
            q = omega.view(1, K) * exp_neg_loss / C_per_sample
            gamma = q / (q.sum(dim=1, keepdim=True) + 1e-12)
            gammas_batches.append(gamma.detach())
            for b in range(B):
                C_yk[:, y[b]] += gamma[b]
            gamma_sums += gamma.sum(dim=0)
            sample_count += B
            batches += 1
            if max_val_batches is not None and batches >= max_val_batches:
                break

        if sample_count > 0:
            omega = gamma_sums / sample_count
            omega = omega / (omega.sum() + 1e-12)

        return gammas_batches, omega

    def state_dict_sub(self, a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[
        str, torch.Tensor]:
        res = {}
        for k in a.keys():
            if self.noise_multiplier > 0:
                group_k = k.split('_module.')[-1]
            else:
                group_k = k
            res[group_k] = a[k] - b[group_k]
        return res

    def local_weighted_update(self, global_models: List[nn.Module], gammas_batches: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        train_loader = self.train_loader
        device = self.device
        local_epochs = self.local_steps
        lr = self.lr
        momentum = 0.9
        weight_decay = 0
        grad_clip = self.grad_clip
        noise_multiplier = self.noise_multiplier
        privacy_engine = self.privacy_engine

        K = len(global_models)
        local_models = [copy.deepcopy(global_models[k]).to(device).train() for k in range(K)]
        optimizers = [
            torch.optim.SGD(local_models[k].parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            for k in range(K)
        ]

        # 仅包装一次
        if noise_multiplier > 0:
            for k in range(K):
                local_models[k], optimizers[k], train_loader = privacy_engine.make_private(
                    Vk=None,
                    module=local_models[k],
                    optimizer=optimizers[k],
                    data_loader=train_loader,
                    noise_multiplier=noise_multiplier,
                    max_grad_norm=grad_clip,
                    poisson_sampling=False,
                )

        for epoch in range(local_epochs):
            batch_idx = 0
            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                gamma = gammas_batches[min(batch_idx, len(gammas_batches)-1)].to(device)
                for k in range(K):
                    optimizers[k].zero_grad(set_to_none=True)
                    logits = local_models[k](x)
                    ce = F.cross_entropy(logits, y, reduction='none')
                    weights = gamma[:, k]
                    denom = weights.sum() + 1e-12
                    loss = torch.sum(weights * ce) / denom
                    loss.backward()
                    optimizers[k].step()
                batch_idx += 1

        deltas = []
        for k in range(K):
            new_sd = local_models[k].state_dict()
            old_sd = global_models[k].state_dict()
            # 参数名一致
            delta = self.state_dict_sub(new_sd, old_sd)
            # delta = {kk: (new_sd[kk] - old_sd[kk]).detach().cpu() for kk in new_sd.keys()}
            deltas.append(delta)
        return deltas


class ClientSampler(abc.ABC):
    @abc.abstractmethod
    def sample(self) -> int:
        """sample the index of a client"""


class FixedSampler(ClientSampler):
    def __init__(self, n_clients, p, seed=1234):
        self.n_clients = n_clients
        self.p = p
        self._rng = default_rng(seed)

    def sample(self) -> List[int]:
        client_indices = np.arange(self.n_clients)
        mask = np.random.choice(client_indices, int(self.n_clients * self.p), replace=False)
        return client_indices[mask]


class FedRCServer:
    def __init__(self, cluster_models, num_clients, test_loader, K: int = 3, device: Optional[torch.device] = None):
        self.name = 'FedRC'
        self.K = K
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cluster_models = cluster_models
        self.test_loader = test_loader
        self.n_clients = num_clients
        self.client_groups = {i: 0 for i in range(num_clients)}
        self.sample_account = np.zeros(num_clients)

    def select_clients(self, sampling_rate):
        client_sampler = FixedSampler(self.n_clients, p=sampling_rate)
        return client_sampler.sample()

    @torch.no_grad()
    def aggregate(self, deltas_list: List[List[Dict[str, torch.Tensor]]], client_sizes: List[int], lr_g: float = 1.0):
        num_clients = len(deltas_list)
        total_size = sum(client_sizes) + 1e-12
        ind = self.cluster_models.keys()
        for k in ind:
            avg_delta = None
            for i in range(num_clients):
                w = client_sizes[i] / total_size
                delta = deltas_list[i][k]
                if avg_delta is None:
                    avg_delta = {key: w * val for key, val in delta.items()}
                else:
                    for key in avg_delta.keys():
                        avg_delta[key] += w * delta[key]
            model_sd = self.cluster_models[k].state_dict()
            for key in model_sd.keys():
                model_sd[key] += lr_g * avg_delta[key].to(model_sd[key].device)
            self.cluster_models[k].load_state_dict(model_sd)

    def get_model_copies(self) -> List[nn.Module]:
        return [copy.deepcopy(self.cluster_models[i]).to(self.device) for i in self.cluster_models.keys()]


# =========================
# 评估
# =========================

def evaluate_cluster_models(models, test_loader):
    test_acc, class_accuracy = 0, 0
    for model in models.values():
        t_acc, t_class_acc = label_test(model, test_loader)
        if t_acc > test_acc:
            test_acc, class_accuracy = t_acc, t_class_acc
    return test_acc, class_accuracy


def label_test(model, test_loader):
    device = next(model.parameters()).device
    num_classes = 10
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    class_loss = [0.0] * num_classes
    criterion = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.long().to(device)
            outputs = model(images)
            losses = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(labels)):
                label_i = labels[i].item()
                pred_i = predicted[i].item()
                class_loss[label_i] += losses[i].item()
                if pred_i == label_i:
                    class_correct[label_i] += 1
                class_total[label_i] += 1
    class_accuracy = []
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i] * 100
        else:
            acc = 0
        class_accuracy.append(acc)
    return np.sum(class_correct)/np.sum(class_total), class_accuracy


def fed_eval(server, clients):
    test_loader = server.test_loader
    test_acc = []
    for i, loader in enumerate(test_loader):
        t_acc, _ = evaluate_cluster_models(server.cluster_models, loader)
        test_acc.append(t_acc)
    return 0, np.mean(test_acc)


# =========================
# Transform 可重建配置
# =========================

def build_transform_from_spec(spec: dict):
    dataset = spec['dataset']
    if dataset in ['Mnist', 'FashionMnist']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif dataset == 'Cifar10':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    else:
        raise ValueError(f'Unsupported dataset in transform spec: {dataset}')


# =========================
# 子进程任务：重建 DataLoader + 模型，结果写临时文件
# =========================

def client_worker_task_to_file(
    client_id: int,
    model_sd_list_cpu: List[Dict[str, torch.Tensor]],
    pack: dict,
    estimate_batches: int,
) -> dict:
    """
    子进程执行：
    - 选择 device（支持多卡轮询）
    - 从 pack['client_arrays'] 重建 train/val DataLoader
    - 重建 K 个模型并加载 CPU state_dict
    - E步 + 本地训练
    - 将 deltas 写入临时 .pt 文件，返回 meta（路径、data_size、omega 也写文件）
    """
    # 子进程独立随机种子
    set_random(pack['base_seed'] + 10000 + client_id)

    # 设备选择
    if torch.cuda.is_available() and pack.get('use_cuda', True):
        num_gpus = torch.cuda.device_count()
        gpu_id = client_id % max(1, num_gpus)
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')

    dataset_name = pack['dataset']
    K = pack['K']
    train_config = pack['train_config']
    transform = build_transform_from_spec({'dataset': dataset_name})

    # 重建 DataLoader（子进程内）
    data_np, labels_np = pack['client_arrays'][client_id]
    # 为确定性/一致性，这里用与父进程相同的划分比例与 batch_ratio（可从 train_config 推导）
    batch_ratio = train_config.get('batch_ratio', 0.02)
    train_ratio = 0.9

    client_dataset = ClientDataset(copy.deepcopy(data_np), labels_np, transform=transform)
    train_size = int(len(client_dataset) * train_ratio)
    val_size = len(client_dataset) - train_size
    train_dataset, val_dataset = random_split(client_dataset, [train_size, val_size])
    if 0 < batch_ratio < 1:
        batch_size = max(int(batch_ratio * len(train_dataset)), 8)
    else:
        batch_size = 64

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, drop_last=True, num_workers=0, pin_memory=True)

    client = FedRCClientLite(train_loader, val_loader, device, train_config)

    # 重建 K 个模型
    global_models = []
    for k in range(K):
        model = create_keras_model(dataset_name).to(device)
        model.load_state_dict(model_sd_list_cpu[k])
        model.eval()
        global_models.append(model)

    # E 步
    gammas_batches, omega_i = client.estimate_gamma_omega_for_client(
        global_models, num_classes=10, max_val_batches=estimate_batches
    )

    # 切换训练
    for k in range(K):
        global_models[k].train()

    # 本地更新
    deltas_i = client.local_weighted_update(
        global_models=global_models,
        gammas_batches=gammas_batches
    )

    # 将结果写入临时文件，返回路径
    tmp_dir = pack['tmp_dir']
    os.makedirs(tmp_dir, exist_ok=True)
    # 为减少单轮 FD 峰值，将每个客户端的 deltas 放到独立 .pt 文件
    delta_path = os.path.join(tmp_dir, f"client_{client_id}_deltas.pt")
    omega_path = os.path.join(tmp_dir, f"client_{client_id}_omega.pt")

    torch.save(deltas_i, delta_path)
    torch.save({'omega': omega_i.detach().cpu(), 'data_size': client.len()}, omega_path)

    # 返回轻量信息（仅路径）
    return {
        'client_id': client_id,
        'delta_path': delta_path,
        'omega_path': omega_path,
    }


# =========================
# 主训练循环（并行）
# =========================

def fedrc_experiment(global_model, train_loader, val_loader, test_loader, EM_num, args, client_arrays_pack):
    estimate_batches = 5
    alg = args.experiment
    M = args.groups1 * args.groups2 * args.clients_per_group
    K = args.K
    device = args.device
    num_rounds = args.rounds
    frac = args.sampling_rate
    client_indices = list(range(M))

    # 服务器簇模型
    cluster_models = {0: global_model}
    for i in range(1, K):
        cluster_models[i] = create_keras_model(args.dataset).to(device=device)
    server = FedRCServer(cluster_models=cluster_models, num_clients=M, K=K, device=device, test_loader=test_loader)

    index_names = []
    column_names = (['sigma'] + ['eps'] + ['sr'] +
                    [f"test_acc{i+1}" for i in range(num_rounds//EM_num)])
    res = pd.DataFrame(index=index_names, columns=column_names)
    res.loc[alg, 'sigma'] = 0
    res.loc[alg, 'eps'] = 0
    res.loc[alg, 'sr'] = frac

    omega_clients = torch.full((M, K), 1.0 / K, device=device)

    # 序列化包：仅传轻量信息 + 原始数组
    tmp_dir = tempfile.mkdtemp(prefix="fedrc_tmp_")
    serialized_pack = {
        'use_cuda': True,
        'dataset': args.dataset,
        'K': K,
        'base_seed': args.seed if hasattr(args, 'seed') else 2024,
        'client_arrays': client_arrays_pack['client_arrays'],  # [(np.array, np.array), ...]
        'train_config': args.train_config,
        'tmp_dir': tmp_dir,
    }

    max_workers = 5
    pool = ProcessPoolExecutor(max_workers=max_workers)
    try:
        for rd in range(num_rounds):
            start_time = time.time()

            m = max(1, int(frac * M))
            selected = random.sample(client_indices, m)
            server.sample_account[selected] += 1

            # 下发：CPU state_dict
            global_models = server.get_model_copies()
            model_sd_list_cpu = []
            for m_k in global_models:
                sd_cpu = {k: v.detach().cpu() for k, v in m_k.state_dict().items()}
                model_sd_list_cpu.append(sd_cpu)

            # 并行执行：返回轻量路径
            futures = []
            for cid in selected:
                futures.append(pool.submit(
                    client_worker_task_to_file,
                    cid,
                    model_sd_list_cpu,
                    serialized_pack,
                    estimate_batches
                ))

            deltas_all_clients = []
            client_sizes = []
            omega_updates = {}

            # 逐个读取文件并删除，避免FD/文件堆积
            for fut in as_completed(futures):
                out = fut.result()
                cid = out['client_id']
                delta_path = out['delta_path']
                omega_path = out['omega_path']

                # 读取
                deltas_i = torch.load(delta_path, map_location='cpu')
                omega_info = torch.load(omega_path, map_location='cpu')
                omega_i = omega_info['omega']
                data_size_i = omega_info['data_size']

                deltas_all_clients.append(deltas_i)
                client_sizes.append(int(data_size_i))
                omega_updates[cid] = omega_i

                # 及时删除临时文件，避免FD/磁盘堆积
                try:
                    os.remove(delta_path)
                except Exception as e:
                    warnings.warn(f"Failed to remove temp delta file: {delta_path}, err={e}")
                try:
                    os.remove(omega_path)
                except Exception as e:
                    warnings.warn(f"Failed to remove temp omega file: {omega_path}, err={e}")

            # 更新 omega
            for cid, omega_i in omega_updates.items():
                omega_clients[cid] = omega_i.to(device)

            # 服务器聚合
            server.aggregate(deltas_all_clients, client_sizes=client_sizes, lr_g=1.0)

            # 评估
            if (rd + 1) % EM_num == 0:
                _, test_acc = fed_eval(server, None)
                res.loc[alg, f"test_acc{(rd+1) // EM_num}"] = test_acc
                print(f"Round {(rd+1)} eval acc: {test_acc:.4f}")

            end_time = time.time()
            print(f"Round {rd+1} time: {end_time - start_time:.2f}s")

        print(f'{alg}_final_test_acc: {test_acc:.4f}')

        # 如果用了DP，可以计算 epsilon（这里保留你的接口；若无必要可忽略）
        if args.train_config.get('noise') > 0:
            # 你的 get_eps 依赖 clients 和 privacy_engine，这里没有持有原 clients。
            # 可按需要自定义全局 epsilon 估计；或将 per-client 统计写入并回传。
            res.loc[alg, 'eps'] = 0  # 占位
    finally:
        # 关闭进程池
        pool.shutdown(wait=True, cancel_futures=True)
        # 清理临时目录
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as e:
            warnings.warn(f"Failed to remove temp dir: {tmp_dir}, err={e}")

    return res


def main(args):
    # 数据划分，注意返回 client_arrays_pack 供子进程重建 DataLoader
    train, val, test, label_maps, report, (all_client_data, all_client_labels, transform) = split_dataset(
        dataset_name=args.dataset,
        num_concept_groups=args.groups1,
        num_label_groups=args.groups2,
        num_clients_per_group=args.clients_per_group,
        alpha=args.alpha,
        overlap_ratio=args.overlap_ratio,
        same_py=args.same_py,
        batch_ratio=args.train_config.get('batch_ratio')
    )
    print(label_maps)
    print('dataset: ', args.dataset)

    # 服务器初始化全局模型（主进程用于评估，可放在 CUDA）
    global_model = create_keras_model(args.dataset).to(device=args.device)
    total_params = sum(p.numel() for p in global_model.parameters())
    args.params_num = total_params
    print('Total params: {}'.format(total_params))

    # 训练配置
    args.train_config['noise'] = 1.0
    args.sampling_rate = 1.0

    # 你的 FedRC 原逻辑里有 EM_num 的轮次展开；这里沿用
    column_names = (['sigma'] + ['eps'] + ['sr'] +
                    [f"test_acc{i+1}" for i in range(args.rounds)])
    result = pd.DataFrame(columns=column_names)
    ori_rounds = args.rounds
    ori_local_steps = args.train_config.get('local_steps')
    EM_num = ori_local_steps

    # args.rounds = int((ori_rounds - 1) * EM_num) + 200 // EM_num
    args.rounds = ori_local_steps

    args.train_config['local_steps'] = max(1, ori_local_steps // EM_num)

    args.experiment = 'FedRC'
    print(args.experiment)

    # 将 per-client 原始数组打包传给子进程，避免直接 pickle DataLoader/FedRCClient
    client_arrays_pack = {
        'client_arrays': [(all_client_data[i], all_client_labels[i]) for i in range(len(all_client_data))],
    }

    start_time = time.time()
    res = fedrc_experiment(copy.deepcopy(global_model), train, val, test, EM_num, args, client_arrays_pack)
    result = pd.concat([result, res])

    noise = args.train_config['noise']
    result.to_csv(f'./res/{args.dataset}'
                  f'_g1{args.groups1}'
                  f'_g2{args.groups2}'
                  f'_pergroup{args.clients_per_group}'
                  f'_alpha{args.alpha}'
                  f'_noise{noise}'
                  f'_rounds{args.rounds}'
                  f'_cmp_experiment.csv')

    # 恢复配置
    args.rounds = ori_rounds
    args.train_config['local_steps'] = ori_local_steps

    print('finish, total time: {:.2f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    # 重要：使用 spawn，修复 CUDA + 多进程
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    os.environ["WANDB_API_KEY"] = 'e5dc3f1f4d367ec3a412d359ae4cfc222deacfa2'
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=UserWarning)

    # 可选：限制可见 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 如需提升系统文件描述符：在 shell 中先执行 `ulimit -n 65535`
    for dataset in ['Cifar10', 'FashionMnist', 'Mnist']:
        args = get_fl_config(dataset)
        if not hasattr(args, 'seed'):
            args.seed = 209
        set_random(args.seed)
        main(args)