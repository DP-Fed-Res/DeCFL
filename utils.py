import abc
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

from typing import List
from numpy.random import default_rng
from model import MnistCnn, FashionMnistCnn, CifarResNet, Cifar10Cnn
from account import MomentsAccountant
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
from cluster import compute_pairwise_dis

import os
from sklearn.metrics import normalized_mutual_info_score

from collections import Counter


def nmi_cal(cluster, pre_cls):
    # mask = np.zeros([args.n_clients, args.groups1])
    # for i in range(args.groups1):
    #     for j in cluster[i]:
    #         mask[j][i] = 1
    #
    # u = best_u * mask
    # u = u / np.sum(u, axis=1, keepdims=True)
    # pyx_nmi = fuzzy_normalized_mutual_info(u, args.prior_cls)

    n = len(pre_cls)
    m = len(Counter(pre_cls))
    u = np.zeros(n)
    for i in range(n):
        for j in range(m):
            if i in cluster[j]:
                u[i] = j
    pyx_nmi = normalized_mutual_info_score(pre_cls, u)

    return pyx_nmi


def evaluate_cluster_models(models, test_loader):
    """评估所有聚类模型，返回最佳测试准确率和对应类别准确率"""
    test_acc, class_accuracy = 0, 0
    for model in models.values():
        t_acc, t_class_acc = label_test(model, test_loader)
        if t_acc > test_acc:
            test_acc, class_accuracy = t_acc, t_class_acc
    return test_acc, class_accuracy


def evaluate_clients(clients, cluster_models, client_groups):
    """评估客户端验证集性能"""
    return [
        clients[i].evaluate(cluster_models[client_groups[i]])[1]
        for i in range(len(clients))
    ]


def fed_eval(server, clients):
    test_loader = server.test_loader
    # 测试集评估
    test_acc = []
    for i, loader in enumerate(test_loader):
        t_acc, class_accuracy = evaluate_cluster_models(server.cluster_models, loader)
        # draw_bar(num_classes=10, data=class_accuracy, name=f'{alg}_cluster{i}_acc')
        test_acc.append(t_acc)
        # print(f'{alg}_test_acc_{i}: {np.max(t_acc)}')

    # 验证集评估
    # val_acc = evaluate_clients(clients, server.cluster_models, server.client_groups)
    # print(f'{alg}_val_acc: {np.mean(val_acc)}')
    return 0, np.mean(test_acc)


def client_grad_tnse(dws):
    # 梯度聚集性可视化
    label = list(range(len(dws)))
    dis_w = compute_pairwise_dis(dws, metric='cosine')
    t_sne(dis_w, label)


def t_sne(data, labels):
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, perplexity=5)
    data_embedded = tsne.fit_transform(data)

    # 可视化结果
    label_size = 14
    plt.figure(figsize=(8, 8))

    # 使用不同的颜色表示不同的类别
    scatter = plt.scatter(data_embedded[:, 0], data_embedded[:, 1], s=120, c=labels, cmap='rainbow', marker='o')

    # 给每个点标注文本（这里直接用 label 值）
    for i, label in enumerate(labels):
        x = data_embedded[i, 0]
        y = data_embedded[i, 1]
        # 可适当加一个偏移量，避免文字和散点重合
        plt.text(
            x + 0.3,
            y + 0.3,
            str(label),
            fontsize=label_size,
            ha='left',
            va='bottom'
        )

    handles, _ = scatter.legend_elements(prop='colors', alpha=0.6, num=n_classes)
    legend_labels = [f'Client {lbl}' for lbl in unique_labels]

    # 添加图例
    # plt.legend(handles, legend_labels,
    #            loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=label_size, labelspacing=1.0)

    # plt.title('t-SNE Visualization of Client Model Gradient')
    plt.xlabel('t-SNE Component 1', fontsize=label_size)
    plt.ylabel('t-SNE Component 2', fontsize=label_size)
    plt.xticks(fontsize=label_size)  # x轴刻度数字大小
    plt.yticks(fontsize=label_size)  # y轴刻度数字大小
    # plt.grid()
    plt.tight_layout()
    plt.savefig('t-SNE.png')
    return 0


def draw_bar(num_classes, data, name):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(np.arange(num_classes), data, color='skyblue', alpha=0.8)
    plt.xlabel('class', fontsize=12)
    plt.ylabel('acc(%)', fontsize=12)
    plt.title(name, fontsize=14)
    plt.xticks(np.arange(num_classes), [f"Class {i}" for i in range(num_classes)])
    # 在柱状图顶部显示数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height,
                 f"{height:.1f}", ha='center', va='bottom', fontsize=10)
    plt.ylim([0, 110])
    plt.show()
    plt.savefig("./imgs/" + name + ".png")


def label_test(model, test_loader):
    num_classes = 10
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    class_loss = [0.0] * num_classes  # 用于累加每个类别的损失
    criterion = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.long().cuda()
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
    # 计算并打印各类准确率和损失
    class_accuracy = []
    class_avg_loss = []
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i] * 100
            avg_loss = class_loss[i] / class_total[i]
        else:
            acc = 0
            avg_loss = 0
        class_accuracy.append(acc)
        class_avg_loss.append(avg_loss)

    return np.sum(class_correct)/np.sum(class_total), class_accuracy


def softmax(x):
    # 为了数值稳定性，减去最大值
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def min_max_normalize(arr, target_min=0, target_max=1):
    """
    将数组线性规范化到指定区间 [target_min, target_max]
    """
    arr = np.array(arr, dtype=float)
    current_min = np.min(arr)
    current_max = np.max(arr)

    # 避免除以零（当所有元素相同时）
    if current_max == current_min:
        return np.full_like(arr, (target_min + target_max) / 2)

    # 线性变换公式
    normalized = (arr - current_min) * (target_max - target_min) / (current_max - current_min) + target_min
    return normalized


def visualize_forward_propagation(activations):
    num_Clients = len(activations)
    plt.figure(figsize=(80, 10))

    for i in range(num_Clients):
        layer = activations[i].cpu()
        plt.subplot(1, num_Clients, i + 1)
        plt.imshow(layer, aspect='auto', cmap='viridis')
        plt.title(f'Client {i}')
        plt.colorbar()
        plt.xlabel('Neurons')
        plt.ylabel('Samples')

    plt.tight_layout()
    plt.show()


def get_eps(clients, smaple_account, args):
    if np.min(args.train_config.get('noise')) > 0.1:
        epsilon = []
        for i in range(args.n_clients):
            if smaple_account[i] > 0:
                epsilon.append(clients[i].privacy_engine.get_epsilon(delta=args.delta))
            else:
                epsilon.append(0)
    else:
        epsilon = np.zeros(args.n_clients) + 1e3
    return epsilon


def get_Vk(args, dataset, model):
    T = int(args.k * 1.5)
    device = args.device
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
    step = 0
    gradients = []
    while step < T:
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()  # 优化器更新语句后模型梯度才会被更新
            # 记录梯度
            grad = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
            gradients.append(grad)
            step += 1
            if step >= T:
                break
    X = torch.stack(gradients)
    _, S, V = torch.svd(X)  # 进行SVD分解
    V_k = V[:, :args.k]
    print('Dimensionality reduction ratio:', torch.sum(S[:args.k]**2) / torch.sum(S**2))
    return V_k


def eval_op(model, loader):
    model.eval()
    samples, correct = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()

            output = model(x)

            pred = output.argmax(dim=1, keepdim=True)  # 获取概率最高的预测结果
            correct += pred.eq(y.view_as(pred)).sum().item()  # 计算正确预测的数量
            samples += y.shape[0]
    model.train()
    return correct / samples


def test_model(model, test_loader, criterion):
    model.eval()  # 将模型设置为评估模式
    test_loss = 0
    correct = 0

    with torch.no_grad():  # 在这个块内部，所有计算都不会计算梯度，减少内存消耗
        for data, target in test_loader:
            data = data.cuda()
            target = target.cuda()
            output = model(data)
            test_loss += criterion(output, target.long()).item()  # 累加测试集上的损失
            pred = output.argmax(dim=1, keepdim=True)  # 获取概率最高的预测结果
            correct += pred.eq(target.view_as(pred)).sum().item()  # 计算正确预测的数量

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    model.train()  # 将模型设置为训练模式
    return test_loss, test_accuracy


def set_random(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 如果有CUDA设备
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_keras_model(dataset):
    if dataset == 'Cifar10':
        # model = CifarResNet(in_channels=3, num_classes=10)  # VGG11(), ResNet9
        model = Cifar10Cnn(in_channels=3, num_classes=10)
    elif dataset == 'FashionMnist':
        model = FashionMnistCnn(in_channels=1, num_classes=10)
    elif dataset == 'Mnist':
        model = MnistCnn(in_channels=1, num_classes=10)
    else:
        raise NotImplementedError
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


def compute_epsilon(steps, sampling_probability, args):
    """Computes epsilon value for given hyperparameters."""
    delta = args.delta
    accountant = MomentsAccountant()
    epsilon = accountant.get_privacy_spent(args.noise_multiplier, sampling_probability, steps, args.delta)
    print('delta: %f ,epsilon: %f' % (delta, epsilon))
    return delta, epsilon


def print_model_parameters(model):
    print("模型各层的参数量:")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()  # 参数数量
        total_params += param
        print(f"{name}: {param}")
    print(f"总参数量: {total_params}")


def partition_report(clients, args):
    # 报告前50个客户端数据分布
    if len(clients) > 50:
        clients = clients[:50]
    col_names = [f"class{i}" for i in range(args.n_class)]
    report = []
    for client in clients:
        d = client.train_dataset.y
        # 计算每个类别的样本数量
        category_counts = [len(d[d == i]) for i in range(args.n_class)]
        report.append(category_counts)
    report_pd = pd.DataFrame(np.array(report), columns=col_names)
    report_pd['client'] = ['client_' + str(i) for i in range(args.n_clients)]
    report_pd = report_pd.set_index('client')
    report_pd[col_names].plot.barh(stacked=True)
    plt.tight_layout()
    plt.xlabel('sample num')
    plt.savefig(args.ckpt_dir + "/cifar10" +
                '_clients' + str(args.n_clients) +
                '_g1' + '(' + str(args.groups1) + ')' +
                '_g2' + '(' + str(args.groups2) + ')' +
                '.png', dpi=1600)
    plt.show()
    return report


def tensor_dict_to_vector(tensor_dict):
    """将包含多个Tensor的字典转换为一个一维向量"""
    vector = torch.cat([t.view(-1) for t in tensor_dict.values()])
    return vector


class ClientSampler(abc.ABC):
    @abc.abstractmethod
    def sample(self) -> int:
        """sample the index of a client"""


class BinomialSampler(ClientSampler):
    def __init__(self, n_clients, p, seed=1234):
        self.n_clients = n_clients
        self.p = p
        self._rng = default_rng(seed)

    def sample(self) -> List[int]:
        # 按概率采样，采样数目不固定，更真实
        client_indices = np.arange(self.n_clients)
        while True:
            mask = self._rng.uniform(size=self.n_clients) < self.p
            if len(client_indices[mask]) > 1:
                break
        return client_indices[mask]


class FixedSampler(ClientSampler):
    def __init__(self, n_clients, p, seed=1234):
        self.n_clients = n_clients
        self.p = p
        self._rng = default_rng(seed)

    def sample(self) -> List[int]:
        # 采固定比例的客户端
        client_indices = np.arange(self.n_clients)
        mask = np.random.choice(client_indices, int(self.n_clients * self.p), replace=False)
        return client_indices[mask]


def cmp_entropy(probabilities):
    """
    计算离散概率分布的熵。
    参数:
    probabilities (list or numpy array): 概率分布列表或数组。
    返回:
    float: 熵值。
    """
    # 将输入转换为 numpy 数组
    probabilities = np.array(probabilities)

    # 检查概率分布是否有效
    if not np.isclose(np.sum(probabilities), b=1.0, rtol=1e-3):
        raise ValueError("概率分布之和必须为 1")
    if np.any(probabilities < 0):
        raise ValueError("概率值不能为负数")

        # 计算熵
    entropy_value = -np.sum(probabilities * np.log2(probabilities, where=probabilities > 0))
    return entropy_value


def fuzzy_normalized_mutual_info(U, true_labels):
    """
    计算模糊归一化互信息（FNMI）。

    参数:
        U (np.ndarray): 模糊聚类隶属度矩阵，形状为 (n_samples, n_clusters)。
                       每行表示一个样本对各簇的隶属度，且行和为1。
        true_labels (np.ndarray): 真实标签，形状为 (n_samples,)。

    返回:
        float: FNMI值，范围 [0, 1]，值越大表示聚类与真实标签一致性越高。
    """
    n_samples, n_clusters = U.shape
    unique_labels = np.unique(true_labels)
    n_classes = len(unique_labels)

    # 1. 计算联合分布 P(U, V)
    joint = np.zeros((n_clusters, n_classes))
    for k in range(n_clusters):
        for c in range(n_classes):
            mask = (true_labels == unique_labels[c])
            joint[k, c] = np.sum(U[mask, k]) / n_samples  # 模糊联合概率

    # 2. 计算边际分布 P(U) 和 P(V)
    p_U = np.sum(joint, axis=1)  # P(U): 聚类边际分布
    p_V = np.sum(joint, axis=0)  # P(V): 真实标签边际分布

    # 3. 计算互信息 I(U; V)
    mi = 0.0
    for k in range(n_clusters):
        for c in range(n_classes):
            if joint[k, c] > 0:
                mi += joint[k, c] * np.log(joint[k, c] / (p_U[k] * p_V[c]))

    # 4. 计算熵 H(U) 和 H(V)
    h_U = -np.sum(p_U * np.log(p_U + 1e-10))  # 避免log(0)
    h_V = -np.sum(p_V * np.log(p_V + 1e-10))

    # 5. 归一化互信息
    if h_U == 0 or h_V == 0:
        return 0.0  # 避免除以0
    else:
        return mi / np.sqrt(h_U * h_V)