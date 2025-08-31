import copy
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Subset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from torch.utils.data import ConcatDataset

from cluster import adaptive_kmeans, compute_pairwise_dis, fuzz_cluster
from utils import FixedSampler, tensor_dict_to_vector, min_max_normalize
from utils import label_test, cmp_entropy
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Iterable, Optional
import torch.nn as nn


class DeCflServer(object):
    def __init__(self, model, test_loader, name, n_clients, device, beta):
        self.name = name
        self.n_clients = n_clients
        self.device = device
        self.sample_account = np.zeros(n_clients)
        self.sampled_clients = None
        self.test_loader = test_loader
        self.cluster_models = {0: model}  # 组标号 -> 类簇模型
        self.client_groups = {}   # 客户端ID -> 组标号
        self.beta = beta
        self.gamma = 1.0
        for i in range(n_clients):
            self.client_groups[i] = 0

    def select_clients(self, sampling_rate):
        client_sampler = FixedSampler(self.n_clients, p=sampling_rate)
        return client_sampler.sample()

    def reset_dW(self, clients):
        for client in clients:
            client.dW = None

    def train_clients(self, clients, sampling_rate=1.0, vk=None, r=0):
        # 客户端训练
        self.reset_dW(clients)
        if sampling_rate < 1.0:
            sampled_clients = self.select_clients(sampling_rate)
        else:
            sampled_clients = np.array(list(range(self.n_clients)))
        self.sample_account[sampled_clients] += 1
        self.sampled_clients = sampled_clients
        grads = []
        clients_pre_py = []
        py_js = []
        for i in range(len(sampled_clients)):  # 采样客户端训练循环
            # 客户端训练
            client = clients[sampled_clients[i]]
            cls = self.client_groups[sampled_clients[i]]
            local_model = copy.deepcopy(self.cluster_models[cls]).to(device=self.device)
            if r == 0:
                pre_py, js = client.train_py_aware(local_model, vk)
                clients_pre_py.append(pre_py)
                py_js.append(js)
            else:
                client.train(local_model, vk)
            grads.append(client.dW.cpu().detach().numpy())
        grads = np.array(grads)
        if r == 0:
            clients_pre_py = np.array(clients_pre_py)
            mean_py_js = np.mean(py_js)
        else:
            clients_pre_py = None
            mean_py_js = None
        return grads, clients_pre_py, mean_py_js

    def eval_clients_all(self, clients, r, draw=False):
        cluster_acc = []
        cluster_loss = []
        for i in range(len(self.cluster_models)):
            # name = 'round' + str(r) + '_cluster' + str(i) if draw else None
            model = self.cluster_models[i]
            cluster_idc = [key for key, value in self.client_groups.items() if value == i]
            val_dataset = ConcatDataset([clients[j].val_loader.dataset for j in cluster_idc])
            acc, loss = label_test(model, val_dataset)
            cluster_acc.append(acc)
            cluster_loss.append(loss)
        return cluster_acc, cluster_loss

    def cls_py(self, dis, cluster_pyx):
        cluster_ind = []
        for c in cluster_pyx:
            tmp = []
            if len(c) > 2:
                t_dis = dis[c][:, c]
                best_labels, best_k, best_score = adaptive_kmeans(t_dis, max_clusters=len(t_dis)-1)
                if best_score > 0.5:
                    c_ind = [np.where(best_labels == i)[0] for i in set(best_labels)]
                    for ind in c_ind:
                        tmp.append(c[ind])
                else:
                    tmp.append(c)
            else:
                tmp.append(c)
            cluster_ind.append(tmp)
        return cluster_ind

    def cls_pyx(self, dis_fc, dis_py, k):
        # 基于修正梯度聚类
        mask = np.triu_indices_from(dis_fc, k=1)
        x1 = dis_py[mask]
        x1 = min_max_normalize(x1, target_min=0.1, target_max=1.0)
        y = dis_fc[mask]
        x1 = x1.reshape(-1, 1)
        ploy = PolynomialFeatures(degree=3)  # 多项式回归
        x_ploy = ploy.fit_transform(x1.reshape(-1, 1))  # 标签分布带来的梯度距离

        # 回归模型构建
        model = Ridge(alpha=0.01)  # 岭回归（L2正则化）
        # model = LinearRegression()  # 线性回归

        # 回归预测
        model.fit(x_ploy, y)
        d_py_effect = model.predict(x_ploy, )
        residuals = y - d_py_effect  # 残差反映P(y|x)影响
        residuals = min_max_normalize(residuals, target_min=0.1, target_max=1.0)
        dis_residual = np.zeros_like(dis_fc)
        dis_residual[mask] = residuals
        dis_residual += dis_residual.T

        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_pyx = kmeans.fit_predict(dis_residual)
        cluster_pyx = [np.where(cluster_pyx == i)[0] for i in set(cluster_pyx)]

        # cluster_pyx2 = kmeans.fit_predict(dis_grad)
        # cluster_pyx2 = [np.where(cluster_pyx2 == i)[0] for i in set(cluster_pyx2)]
        # cluster_pyx2, best_u = fuzz_cluster(dis_residual, k)

        return dis_residual, cluster_pyx

    def client_grouping(self, grads, grad_split, clients_pre_py, k, grad_correction=True):
        """
        :param grads: numpy数组
        :param clients_pre_py: numpy数组
        """
        backbone_grads = np.array(grads[:, :grad_split[0]])
        fcw_grads = np.array(grads[:, grad_split[0]:grad_split[1]])
        fcb_grads = np.array(grads[:, grad_split[1]:])
        clients_pre_py = np.array(clients_pre_py)

        dis_backbone = compute_pairwise_dis(backbone_grads, metric='cosine')
        dis_fcw = compute_pairwise_dis(fcw_grads, metric='cosine')
        dis_fcb = compute_pairwise_dis(fcb_grads, metric='cosine')
        dis_py = compute_pairwise_dis(clients_pre_py, metric='js')

        # fcw_grads2 = fcw_grads.reshape(len(fcw_grads), 10, 1600)
        # dis_fcw2 = compute_pairwise_dis(fcw_grads2, metric='fcw')
        # 面向pyx聚类
        if grad_correction:
            dis_residual, cluster_pyx = self.cls_pyx(dis_fcw, dis_py, k)
            dis_residual2, cluster_pyx2 = self.cls_pyx(dis_fcb, dis_py, k)
            dis_residual3, cluster_pyx3 = self.cls_pyx(dis_backbone, dis_py, k)
        else:
            dis_grad = compute_pairwise_dis(grads, metric='cosine')
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_pyx = kmeans.fit_predict(dis_grad)
            cluster_pyx = [np.where(cluster_pyx == i)[0] for i in set(cluster_pyx)]
            # cluster_pyx, best_u = fuzz_cluster(dis_grad, k)
        # 面向py细分类簇
        dis_py = compute_pairwise_dis(clients_pre_py, metric='js')
        cluster_py = self.cls_py(dis_py, cluster_pyx)
        return cluster_pyx, cluster_py

    def avg_aggregate(self, group_clients):
        # 初始化聚合模型
        group_model = copy.deepcopy(self.cluster_models[0])
        group_model_params = group_model.state_dict()

        # 加权聚合
        for key in group_model_params.keys():
            total_weight = 0  # 计算归一化因子
            group_model_params[key] = torch.zeros_like(group_model_params[key])
            for client in group_clients:
                if client.noise_multiplier:
                    client_key = '_module.' + key
                else:
                    client_key = key
                group_model_params[key] += client.model_params[client_key] * client.data_size
                total_weight += client.data_size

            if total_weight > 0:
                group_model_params[key] /= total_weight
        # 更新组内模型
        group_model.load_state_dict(group_model_params)
        return group_model


    # def hierarchical_aggregation(self, group_clients, performance=None):
    #     '''
    #     :param group_clients:
    #     :param init_model:
    #     '''
    #     group_model = copy.deepcopy(self.cluster_models[0])
    #     group_model_params = group_model.state_dict()
    #
    #     M = performance.shape[1]
    #
    #     # 加权聚合
    #     for key in group_model_params.keys():
    #         total_weight = 0  # 计算归一化因子
    #
    #         group_model_params[key] = torch.zeros_like(group_model_params[key])
    #         for idx, client in group_clients.items():
    #             d = jensenshannon(performance[idx], np.zeros(M) + 1 / M)
    #             w = client.data_size * 1 / (1 + 50 * d)
    #             group_model_params[key] += client.model_params[key] * w
    #             total_weight += w
    #
    #         if total_weight > 0:
    #             group_model_params[key] /= total_weight
    #
    #     group_model.load_state_dict(group_model_params)
    #     return group_model


    def hierarchical_aggregation(self, group_clients, performance=None):
        '''
        :param group_clients:
        :param init_model:
        '''
        group_model = copy.deepcopy(self.cluster_models[0])
        group_model_params = group_model.state_dict()

        # 加权聚合
        for key in group_model_params.keys():
            group_model_params[key] = torch.zeros_like(group_model_params[key])

            if 'fc' in key:
                m = len(group_model_params[key])
                total_weight = np.zeros(m)  # 计算归一化因子
                b_km = np.abs(performance - 1/m * 2)
                beta = self.beta
                for idx, client in group_clients.items():
                    if client.noise_multiplier:
                        client_key = '_module.' + key
                    else:
                        client_key = key
                    for i in range(m):
                        bias = np.mean(b_km[idx]) + (m-1)/m * b_km[idx][i]
                        # w = 1 / (beta/client.data_size + self.gamma * bias**2)
                        w = client.data_size * performance[idx][i]
                        group_model_params[key][i] += client.model_params[client_key][i] * w
                        total_weight[i] += w
                for i in range(m):
                    if total_weight[i] > 0:
                        group_model_params[key][i] /= total_weight[i]
            else:
                total_weight = 0  # 计算归一化因子
                for client in group_clients.values():
                    if client.noise_multiplier:
                        client_key = '_module.' + key
                    else:
                        client_key = key
                    group_model_params[key] += client.model_params[client_key] * client.data_size
                    total_weight += client.data_size

                if total_weight > 0:
                    group_model_params[key] /= total_weight

        group_model.load_state_dict(group_model_params)
        self.gamma *= 0.9
        return group_model


    # def hierarchical_aggregation(self, group_clients, performance):
    #     '''
    #     :param group_clients:
    #     :param init_model:
    #     :param performance: 形如(n_clients, n_classes)的numpy数组
    #     '''
    #     noise_multiplier = list(group_clients.values())[0].noise_multiplier > 0
    #     # 分离共享参数和个性化参数，并初始化
    #     aggregated_shared = {}
    #     aggregated_personal = {}
    #
    #     n = len(group_clients)
    #     group_model = copy.deepcopy(self.cluster_models[0])
    #
    #     # 假设分类层是最后一层全连接层（根据实际模型结构调整）
    #     for key, param in group_model.named_parameters():
    #         if 'fc' in key:  # 分类层
    #             aggregated_personal[key] = torch.zeros_like(param)
    #         else:  # 特征提取层
    #             aggregated_shared[key] = torch.zeros_like(param)
    #
    #     # Step 1: 对所有客户端聚合特征提取层（FedAvg）
    #     total_data_size = sum(client.data_size for client in group_clients.values())
    #     for key in aggregated_shared.keys():
    #         for client in group_clients.values():
    #             if noise_multiplier:
    #                 aggregated_shared[key] += client.model_params['_module.' + key] * client.data_size
    #             else:
    #                 aggregated_shared[key] += client.model_params[key] * client.data_size
    #         if total_data_size > 0:
    #             aggregated_shared[key] /= total_data_size
    #
    #     # Step 2: 对所有客户端聚合分类层(类分布加权法)
    #     for key in aggregated_personal.keys():
    #         m = len(aggregated_personal[key])
    #         total_perf_weight = np.zeros(m)
    #         for idx, client in group_clients.items():
    #             if noise_multiplier:
    #                 tmp = client.model_params['_module.' + key]
    #             else:
    #                 tmp = client.model_params[key]
    #
    #             w = client.data_size
    #             aggregated_personal[key] += tmp * w
    #
    #             for i in range(m):
    #                 # w = client.data_size
    #                 # aggregated_personal[key][i] += tmp[i] * w
    #                 total_perf_weight[i] += w
    #
    #         for i in range(m):
    #             if total_perf_weight[i] > 0:
    #                 aggregated_personal[key][i] /= total_perf_weight[i]
    #
    #     # b_km = np.abs(performance - 1/performance.shape[1])
    #     # beta = self.beta
    #     #
    #     # for name in aggregated_personal.keys():
    #     #     m = len(aggregated_personal[name])
    #     #     total_perf_weight = np.zeros(m)
    #     #     for idx, client in group_clients.items():
    #     #         if noise_multiplier:
    #     #             tmp = client.model_params['_module.' + name]
    #     #         else:
    #     #             tmp = client.model_params[name]
    #     #
    #     #         for i in range(m):
    #     #             bias = np.mean(b_km[idx]) + (m-1)/m * b_km[idx][i]
    #     #             w = client.data_size / (beta + 0 * client.data_size * bias**2)
    #     #             aggregated_personal[name][i] += tmp[i] * w
    #     #             total_perf_weight[i] += w
    #     #
    #     #     for i in range(m):
    #     #         if total_perf_weight[i] > 0:
    #     #             aggregated_personal[name][i] /= total_perf_weight[i]
    #
    #     group_model_dict = {**aggregated_shared, **aggregated_personal}
    #     group_model.load_state_dict(group_model_dict)
    #
    #     return group_model


class FedAvgServer(object):
    def __init__(self, model, test_loader, name, n_clients, device, train_config):
        self.name = name
        self.n_clients = n_clients
        self.device = device
        self.sample_account = np.zeros(n_clients)
        self.sampled_clients = None
        self.test_loader = test_loader
        self.cluster_models = {0: model}
        self.client_groups = {}  # 客户端ID -> 组标号

        self.grad_clip = train_config.get('clip')
        self.noise_multiplier = train_config.get('noise')
        for i in range(n_clients):
            self.client_groups[i] = 0

    def select_clients(self, sampling_rate):
        client_sampler = FixedSampler(self.n_clients, p=sampling_rate)
        return client_sampler.sample()

    def reset_dW(self, clients):
        for client in clients:
            client.dW = None

    def train_clients(self, clients, sampling_rate=1.0, Vk=None):
        # 客户端训练
        self.reset_dW(clients)
        if sampling_rate < 1.0:
            sampled_clients = self.select_clients(sampling_rate)
        else:
            sampled_clients = np.array(list(range(self.n_clients)))
        self.sample_account[sampled_clients] += 1
        self.sampled_clients = sampled_clients
        for i in range(len(sampled_clients)):  # 采样客户端训练循环
            # 客户端训练
            client = clients[sampled_clients[i]]
            cls = self.client_groups[sampled_clients[i]]
            local_model = copy.deepcopy(self.cluster_models[cls]).to(device=self.device)
            client.train(local_model, Vk)

    def avg_aggregate(self, group_clients):
        # 初始化聚合模型
        group_model = copy.deepcopy(self.cluster_models[0])
        group_model_params = group_model.state_dict()

        # 计算总数据量
        total_data_size = sum(client.data_size for client in group_clients)

        # 加权聚合
        for key in group_model_params.keys():
            group_model_params[key] = torch.zeros_like(group_model_params[key])
            if self.noise_multiplier > 0:
                client_key = '_module.' + key
            else:
                client_key = key
            for client in group_clients:
                group_model_params[key] += client.model_params[client_key]*(client.data_size/total_data_size)

        # 更新组内模型
        group_model.load_state_dict(group_model_params)
        return group_model


class FedProxServer(object):
    def __init__(self, model, test_loader, name, n_clients, device, train_config):
        self.name = name
        self.n_clients = n_clients
        self.device = device
        self.sample_account = np.zeros(n_clients)
        self.sampled_clients = None
        self.test_loader = test_loader
        self.cluster_models = {0: model}
        self.client_groups = {}  # 客户端ID -> 组标号
        self.grad_clip = train_config.get('clip')
        self.noise_multiplier = train_config.get('noise')
        for i in range(n_clients):
            self.client_groups[i] = 0

    def select_clients(self, sampling_rate):
        client_sampler = FixedSampler(self.n_clients, p=sampling_rate)
        return client_sampler.sample()

    def reset_dW(self, clients):
        for client in clients:
            client.dW = None

    def train_clients(self, clients, sampling_rate=1.0):
        # 客户端训练
        self.reset_dW(clients)
        if sampling_rate < 1.0:
            sampled_clients = self.select_clients(sampling_rate)
        else:
            sampled_clients = np.array(list(range(self.n_clients)))
        self.sample_account[sampled_clients] += 1
        self.sampled_clients = sampled_clients
        for i in range(len(sampled_clients)):  # 采样客户端训练循环
            # 客户端训练
            client = clients[sampled_clients[i]]
            cls = self.client_groups[sampled_clients[i]]
            local_model = copy.deepcopy(self.cluster_models[cls]).to(device=self.device)
            client.train(local_model)

    def avg_aggregate(self, group_clients):
        # 初始化聚合模型
        group_model = copy.deepcopy(self.cluster_models[0])
        group_model_params = group_model.state_dict()

        # 计算总数据量
        total_data_size = sum(client.data_size for client in group_clients)

        # 加权聚合
        for key in group_model_params.keys():
            group_model_params[key] = torch.zeros_like(group_model_params[key])
            if self.noise_multiplier > 0:
                client_key = '_module.' + key
            else:
                client_key = key
            for client in group_clients:
                group_model_params[key] += client.model_params[client_key]*(client.data_size/total_data_size)

        # 更新组内模型
        group_model.load_state_dict(group_model_params)
        return group_model


class ScaffoldServer(object):
    def __init__(self, model, test_loader, name, n_clients, device, train_config):
        self.name = name
        self.n_clients = n_clients
        self.device = device
        self.sample_account = np.zeros(n_clients)
        self.sampled_clients = None
        self.test_loader = test_loader
        self.cluster_models = {0: model}
        self.client_groups = {}  # 客户端ID -> 组标号
        self.grad_clip = train_config.get('clip')
        self.noise_multiplier = train_config.get('noise')
        for i in range(n_clients):
            self.client_groups[i] = 0

        self.server_control = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    def select_clients(self, sampling_rate):
        client_sampler = FixedSampler(self.n_clients, p=sampling_rate)
        return client_sampler.sample()

    def reset_dW(self, clients):
        for client in clients:
            client.dW = None

    def train_clients(self, clients, sampling_rate=1.0):
        # 客户端训练
        self.reset_dW(clients)
        if sampling_rate < 1.0:
            sampled_clients = self.select_clients(sampling_rate)
        else:
            sampled_clients = np.array(list(range(self.n_clients)))
        self.sample_account[sampled_clients] += 1
        self.sampled_clients = sampled_clients
        for i in range(len(sampled_clients)):  # 采样客户端训练循环
            # 客户端训练
            client = clients[sampled_clients[i]]
            cls = self.client_groups[sampled_clients[i]]
            local_model = copy.deepcopy(self.cluster_models[cls]).to(device=self.device)
            client.train(local_model, self.server_control)

    def avg_aggregate(self, group_clients):
        # 初始化聚合模型
        group_model = copy.deepcopy(self.cluster_models[0])
        group_model_params = group_model.state_dict()

        # 计算总数据量
        total_data_size = sum(client.data_size for client in group_clients)

        # 加权聚合
        for key in group_model_params.keys():
            group_model_params[key] = torch.zeros_like(group_model_params[key])
            if self.noise_multiplier > 0:
                client_key = '_module.' + key
            else:
                client_key = key
            for client in group_clients:
                group_model_params[key] += client.model_params[client_key]*(client.data_size/total_data_size)

        # 更新全局控制变量
        # Update global control variate
        for name in self.server_control:
            self.server_control[name] += torch.mean(torch.stack([client.delta_control[name]
                                                                 for client in group_clients]), dim=0)
        # 更新组内模型
        group_model.load_state_dict(group_model_params)
        return group_model


class FlexCflServer(object):
    def __init__(self, model, test_loader, name, n_clients, device, train_config):
        self.name = name
        self.n_clients = n_clients
        self.device = device
        self.sample_account = np.zeros(n_clients)
        self.sampled_clients = None
        self.test_loader = test_loader
        self.cluster_models = {0: model}
        self.client_groups = {}  # 客户端ID -> 组标号
        self.grad_clip = train_config.get('clip')
        self.noise_multiplier = train_config.get('noise')
        for i in range(n_clients):
            self.client_groups[i] = 0
        self.eta_g = 0.1

    def select_clients(self, sampling_rate):
        client_sampler = FixedSampler(self.n_clients, p=sampling_rate)
        return client_sampler.sample()

    def reset_dW(self, clients):
        for client in clients:
            client.dW = None

    def group_cold_start(self, w, k):
        for i in range(len(w)):
            w[i] = w[i].detach().cpu().numpy()
        delta_w = np.array(w)  # shape=(n_clients, n_params)
        svd = TruncatedSVD(n_components=k)
        decomp_updates = svd.fit_transform(delta_w.T)  # shape=(n_params, n_groups)
        decomposed_cossim_matrix = cosine_similarity(delta_w, decomp_updates.T)  # shape=(n_clients, n_clients)
        affinity_matrix = decomposed_cossim_matrix
        result = KMeans(k, max_iter=100).fit(affinity_matrix)
        cluster = result.labels_
        cluster = [np.where(cluster == i)[0] for i in set(cluster)]
        return cluster

    def train_clients(self, clients, sampling_rate=1.0):
        # 客户端训练
        dw = []
        self.reset_dW(clients)
        if sampling_rate < 1.0:
            sampled_clients = self.select_clients(sampling_rate)
        else:
            sampled_clients = np.array(list(range(self.n_clients)))
        self.sample_account[sampled_clients] += 1
        self.sampled_clients = sampled_clients
        for i in range(len(sampled_clients)):  # 采样客户端训练循环
            # 客户端训练
            client = clients[sampled_clients[i]]

            cls = self.client_groups[sampled_clients[i]]
            local_model = copy.deepcopy(self.cluster_models[cls]).to(device=self.device)

            if i == 0:
                ori_local_steps = client.local_steps
                client.local_steps = 200
                client.train(local_model)
                client.local_steps = ori_local_steps
            else:
                client.train(local_model)
            dw.append(client.dW)

        return dw

    def IntraGroupUpdate(self, group_clients):
        # 初始化聚合模型
        group_model = copy.deepcopy(self.cluster_models[0])
        group_model_params = group_model.state_dict()

        # 计算总数据量
        total_data_size = sum(client.data_size for client in group_clients)

        # 加权聚合
        for key in group_model_params.keys():
            group_model_params[key] = torch.zeros_like(group_model_params[key])
            if self.noise_multiplier > 0:
                client_key = '_module.' + key
            else:
                client_key = key
            for client in group_clients:
                group_model_params[key] += client.model_params[client_key]*(client.data_size/total_data_size)

        # 更新组内模型
        group_model.load_state_dict(group_model_params)
        return group_model

    def InterGroupAggregation(self):
        param = {}
        for key in self.cluster_models.keys():
            dw = copy.deepcopy(tensor_dict_to_vector(self.cluster_models[key].state_dict()))
            param[key] = torch.norm(dw, p=2).item()

        m = copy.deepcopy(self.cluster_models[0].state_dict())
        for key in self.cluster_models.keys():
            for k in m.keys():
                m[k] = m[k] * 0.0
                for key_ in self.cluster_models.keys():
                    if key_ == key:
                        continue
                    m[k] = m[k] + torch.div(self.eta_g * self.cluster_models[key_].state_dict()[k], param[key_])

                m[k] = self.cluster_models[key].state_dict()[k] + m[k]

            self.cluster_models[key].load_state_dict(m)


class FeSemServer(object):
    def __init__(self, model, test_loader, name, n_clients, device, train_config):
        self.name = name
        self.n_clients = n_clients
        self.device = device
        self.sample_account = np.zeros(n_clients)
        self.sampled_clients = None
        self.test_loader = test_loader

        self.cluster_models = {0: model}  # 组标号 -> 类簇模型

        self.client_groups = {}  # 客户端ID -> 组标号
        for i in range(n_clients):
            self.client_groups[i] = 0

        self.grad_clip = train_config.get('clip')
        self.noise_multiplier = train_config.get('noise')

    def select_clients(self, sampling_rate):
        client_sampler = FixedSampler(self.n_clients, p=sampling_rate)
        return client_sampler.sample()

    def reset_dW(self, clients):
        for client in clients:
            client.dW = None

    def train_clients(self, clients, sampling_rate=1.0):
        # 客户端训练
        client_params = []
        self.reset_dW(clients)
        if sampling_rate < 1.0:
            sampled_clients = self.select_clients(sampling_rate)
        else:
            sampled_clients = np.array(list(range(self.n_clients)))
        self.sample_account[sampled_clients] += 1
        self.sampled_clients = sampled_clients
        for i in range(len(sampled_clients)):  # 采样客户端训练循环
            # 客户端训练
            client = clients[sampled_clients[i]]

            cls = self.client_groups[sampled_clients[i]]
            local_model = copy.deepcopy(self.cluster_models[cls]).to(device=self.device)

            client.train(local_model)

            client_params.append(tensor_dict_to_vector(client.model_params))

        return client_params

    def avg_aggregate(self, group_clients):
        # 初始化聚合模型
        group_model = copy.deepcopy(self.cluster_models[0])
        group_model_params = group_model.state_dict()

        # 计算总数据量
        total_data_size = sum(client.data_size for client in group_clients)

        # 加权聚合
        for key in group_model_params.keys():
            group_model_params[key] = torch.zeros_like(group_model_params[key])
            if self.noise_multiplier > 0:
                client_key = '_module.' + key
            else:
                client_key = key
            for client in group_clients:
                group_model_params[key] += client.model_params[client_key]*(client.data_size/total_data_size)
                # group_model_params[key] += client.model_params[key] * (1 / len(group_clients))

        # 更新组内模型
        group_model.load_state_dict(group_model_params)
        return group_model


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
