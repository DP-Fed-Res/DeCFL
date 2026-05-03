import gc
import copy
import torch
import numpy as np
from numpy import ndarray, dtype, signedinteger
from sklearn.preprocessing import PolynomialFeatures

from cluster import adaptive_kmeans, compute_pairwise_dis, fuzz_cluster
from utils import tensor_dict_to_vector, min_max_normalize
from utils import label_test, cmp_entropy, create_keras_model
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from numpy.random import default_rng
from typing import List, Any


class FixedSampler(object):
    def __init__(self, n_clients, p, seed=1234):
        self.n_clients = n_clients
        self.p = p
        self._rng = default_rng(seed)

    def sample(self) -> ndarray[Any, dtype[signedinteger[Any]]]:
        # 采固定比例的客户端
        client_indices = np.arange(self.n_clients)
        mask = np.random.choice(client_indices, int(self.n_clients * self.p), replace=False)
        return client_indices[mask]


class BaseServer(object):
    def __init__(self, model, test_loader, name, n_clients, device):
        self.name = name
        self.n_clients = n_clients
        self.device = device
        self.test_loader = test_loader

        self.sample_account = np.zeros(n_clients)
        self.sampled_clients = None

        self.cluster_models = {0: model}  # 组标号 -> 类簇模型
        self.client_groups = {i: 0 for i in range(n_clients)}  # 客户端ID -> 组标号

    def select_clients(self, sampling_rate):
        client_sampler = FixedSampler(self.n_clients, p=sampling_rate)
        return client_sampler.sample()

    def train_clients(self, clients, r, sampling_rate=1.0, vk=None):
        # 客户端训练
        dw = []
        if sampling_rate < 1.0:
            sampled_clients = self.select_clients(sampling_rate)
        else:
            sampled_clients = np.arange(self.n_clients)

        self.sample_account[sampled_clients] += 1
        self.sampled_clients = sampled_clients

        for cid in sampled_clients:
            client = clients[cid]
            cls = self.client_groups[cid]

            local_model = copy.deepcopy(self.cluster_models[cls]).to(device=self.device)
            client.train(local_model, r=r, vk=vk)

            init_params = {k.replace('_module.', ''): v.cpu() for k, v in self.cluster_models[cls].state_dict().items()}
            client_params = {k.replace('_module.', ''): v.cpu() for k, v in client.model_params.items()}

            update_parts = []
            for k, init_val in init_params.items():
                # 自动处理原始键名差异（支持_module.前缀）
                update_val = client_params[k]
                # 计算差值并立即展平（避免中间副本）
                diff = (update_val - init_val).view(-1)
                update_parts.append(diff)
            delta_w = torch.cat(update_parts) if update_parts else torch.tensor([])

            dw.append(delta_w)

            del local_model

        return dw


    def avg_aggregate(self, group_clients):
        first_client = group_clients[0]
        group_id = self.client_groups[first_client.id]

        global_model = self.cluster_models[group_id]
        global_dict = global_model.state_dict()

        total_data_size = sum(client.data_size for client in group_clients)

        avg_dict = {}
        for key in global_dict.keys():
            avg_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32, device='cpu')

            for client in group_clients:
                if getattr(client, 'noise_mul', 0) > 0:
                    client_key = '_module.' + key
                else:
                    client_key = key

                client_param = client.model_params[client_key].float().cpu()

                weight = client.data_size / total_data_size
                avg_dict[key] += client_param * weight

        global_model.load_state_dict(avg_dict)

        global_model.to(self.device)

        return global_model


class FedProxServer(BaseServer):
    """
    FedProx 服务器
    """
    def __init__(self, model, test_loader, name, n_clients, device):
        super().__init__(model, test_loader, name, n_clients, device)


class ScaffoldServer(BaseServer):
    """
    Scaffold 服务器
    """
    def __init__(self, model, test_loader, name, n_clients, device):
        super().__init__(model, test_loader, name, n_clients, device)
        self.server_control = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    def train_clients(self, clients, sampling_rate=1.0, vk=None, r=0):
        # 客户端训练
        if sampling_rate < 1.0:
            sampled_clients = self.select_clients(sampling_rate)
        else:
            sampled_clients = np.arange(self.n_clients)

        self.sample_account[sampled_clients] += 1
        self.sampled_clients = sampled_clients

        for cid in sampled_clients:
            client = clients[cid]
            cls = self.client_groups[cid]

            local_model = copy.deepcopy(self.cluster_models[cls]).to(device=self.device)
            client.train(local_model, r=r, vk=vk)

            del local_model


    def avg_aggregate(self, group_clients):
        first_client = group_clients[0]
        group_id = self.client_groups[first_client.id]

        global_model = self.cluster_models[group_id]
        global_dict = global_model.state_dict()

        avg_dict = {}
        for key in global_dict.keys():
            avg_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32, device='cpu')

            for client in group_clients:
                if getattr(client, 'noise_mul', 0) > 0:
                    client_key = '_module.' + key
                else:
                    client_key = key

                client_param = client.model_params[client_key].float().cpu()

                weight = 1.0 / len(group_clients)
                avg_dict[key] += client_param * weight

        # Update global control variate
        sampling_ratio = len(group_clients) / self.n_clients

        for name in self.server_control:
            stacked_deltas = torch.stack([client.delta_control[name].to(self.device) for client in group_clients])

            delta_c_mean = torch.mean(stacked_deltas, dim=0)

            # 核心修正：c_t = c_t-1 + l * \Delta c_t
            self.server_control[name] += sampling_ratio * delta_c_mean

        global_model.load_state_dict(avg_dict)

        global_model.to(self.device)

        return global_model


class FlexCflServer(BaseServer):
    """
    FlexCfl 服务器
    """
    def __init__(self, model, test_loader, name, n_clients, device):
        super().__init__(model, test_loader, name, n_clients, device)
        self.eta_g = 0.1

    def group_cold_start(self, w, k):
        for i in range(len(w)):
            if not isinstance(w[i] , np.ndarray):
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


class FeSemServer(BaseServer):
    def __init__(self, model, test_loader, name, n_clients, device):
        super().__init__(model, test_loader, name, n_clients, device)

    def train_clients(self, clients, sampling_rate=1.0, vk=None, r=0):
        client_params = []
        # 客户端训练
        if sampling_rate < 1.0:
            sampled_clients = self.select_clients(sampling_rate)
        else:
            sampled_clients = np.arange(self.n_clients)

        self.sample_account[sampled_clients] += 1
        self.sampled_clients = sampled_clients

        for cid in sampled_clients:
            client = clients[cid]
            cls = self.client_groups[cid]

            local_model = copy.deepcopy(self.cluster_models[cls]).to(device=self.device)
            client.train(local_model, r=r, vk=vk)

            client_params.append(tensor_dict_to_vector(client.model_params))

            del local_model

        return client_params


from typing import List, Tuple, Dict, Iterable, Optional
class FedRCServer(BaseServer):
    def __init__(self, model, test_loader, name, n_clients, device, dataset, K):
        super().__init__(model, test_loader, name, n_clients, device)
        self.num_clusters = K
        for i in range(1, K):
            self.cluster_models[i] = create_keras_model(dataset).to(device=device)

    def get_global_models(self):
        """
        直接返回 K 个模型实例，供实验主循环下发给客户端
        """
        return self.cluster_models

    def aggregate(self, client_updates):
        """
        加权聚合逻辑不变，使用客户端计算的 pi_weights 聚合 updated_state_dicts
        :param client_updates: 列表, 元素为 (K个更新后的state_dict, 权重数组pi)
        """
        new_cluster_states = [
            {name: torch.zeros_like(param) for name, param in self.cluster_models[k].state_dict().items()}
            for k in range(self.num_clusters)
        ]

        cluster_total_weights = np.zeros(self.num_clusters)

        for client_idx, (c_state_dicts, c_pi) in enumerate(client_updates):
            for k in range(self.num_clusters):
                weight = c_pi[k]
                cluster_total_weights[k] += weight

                for name in new_cluster_states[k].keys():
                    new_cluster_states[k][name] += weight * c_state_dicts[k][name].to(self.device)

        # 归一化并更新到服务端的全局模型实例中
        for k in range(self.num_clusters):
            if cluster_total_weights[k] > 0:
                for name in new_cluster_states[k].keys():
                    new_cluster_states[k][name] /= cluster_total_weights[k]
            else:
                new_cluster_states[k] = self.cluster_models[k].state_dict()

            self.cluster_models[k].load_state_dict(new_cluster_states[k])


class DeCflServer(BaseServer):
    """
    DeCfl 自定义服务器，包含聚类和层次聚合逻辑
    """
    def __init__(self, model, test_loader, name, n_clients, device, beta):
        super().__init__(model, test_loader, name, n_clients, device)
        self.beta = beta
        self.gamma = 1.0

    def train_clients(self, clients, r, sampling_rate=1.0, vk=None):
        # 客户端训练
        if sampling_rate < 1.0:
            sampled_clients = self.select_clients(sampling_rate)
        else:
            sampled_clients = np.arange(self.n_clients)

        self.sample_account[sampled_clients] += 1
        self.sampled_clients = sampled_clients

        cluster_init_states = {}
        for cls, model in self.cluster_models.items():
            # 安全地提取到 CPU 并剥离计算图
            cluster_init_states[cls] = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        grads, clients_pre_py, py_js = [], [], []
        for cid in sampled_clients:
            client = clients[cid]
            cls = self.client_groups[cid]

            local_model = copy.deepcopy(self.cluster_models[cls]).to(device=self.device)
            if r == 0:
                pre_py, js = client.train(local_model, r=r, vk=vk)
                clients_pre_py.append(pre_py)
                py_js.append(js)
            else:
                client.train(local_model, r=r, vk=vk)
            del local_model

            init_state = cluster_init_states[cls]
            update_state = client.model_params  # 也是客户端模型在cpu上的state_dict

            client_grad_list = []
            for k in init_state.keys():
                if getattr(client, 'noise_mul', 0) > 0:
                    client_key = '_module.' + k
                else:
                    client_key = k
                delta = init_state[k] - update_state[client_key]
                client_grad_list.append(delta.view(-1))

            flat_grad = torch.cat(client_grad_list)
            grads.append(flat_grad.numpy())

        grads = np.stack(grads) if len(grads) > 0 else np.array([])
        if r == 0:
            clients_pre_py = np.array(clients_pre_py)
            mean_py_js = np.mean(py_js) if len(py_js) > 0 else 0.0
            return grads, clients_pre_py, mean_py_js
        else:
            return grads

    def cls_pyx(self, dis_fc, dis_py, k):
        # 基于修正梯度聚类
        mask = np.triu_indices_from(dis_fc, k=1)
        x1 = dis_py[mask]
        x1 = min_max_normalize(x1, target_min=1e-5, target_max=1.0)
        y = dis_fc[mask]
        y = min_max_normalize(y, target_min=1e-5, target_max=1.0)

        x1 = x1.reshape(-1, 1)
        ploy = PolynomialFeatures(degree=3)  # 多项式回归
        x_ploy = ploy.fit_transform(x1.reshape(-1, 1))  # 标签分布带来的梯度距离

        # 回归模型构建
        model = Ridge(alpha=0.0001)  # 岭回归（L2正则化）

        # 回归预测
        model.fit(x_ploy, y)
        d_py_effect = model.predict(x_ploy)
        residuals = y - d_py_effect  # 残差反映P(y|x)影响
        residuals = min_max_normalize(residuals, target_min=0.1, target_max=1.0)
        dis_residual = np.zeros_like(dis_fc)
        dis_residual[mask] = residuals
        dis_residual += dis_residual.T

        # kmeans = KMeans(n_clusters=k, random_state=42)
        # cluster_pyx = kmeans.fit_predict(dis_residual)
        # cluster_pyx = [np.where(cluster_pyx == i)[0] for i in set(cluster_pyx)]

        cluster_pyx, best_u = fuzz_cluster(dis_residual, k)

        return dis_residual, cluster_pyx

    def client_grouping(self, grads, grad_split, clients_pre_py, k, grad_correction=True):
        """
        :param grads: numpy数组
        :param clients_pre_py: numpy数组
        """
        backbone_grads = np.array(grads[:, :grad_split[0]])
        fcb_grads = np.array(grads[:, grad_split[1]:])
        fcw_grads = np.array(grads[:, grad_split[0]:grad_split[1]])
        clients_pre_py = np.array(clients_pre_py)

        dis_fcw = compute_pairwise_dis(fcw_grads, metric='cosine')
        dis_py = compute_pairwise_dis(clients_pre_py, metric='js')

        # 面向pyx聚类
        if grad_correction:
            dis, cluster_pyx = self.cls_pyx(dis_fcw, dis_py, k)
        else:
            dis = compute_pairwise_dis(grads, metric='cosine')
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_pyx = kmeans.fit_predict(dis)
            cluster_pyx = [np.where(cluster_pyx == i)[0] for i in set(cluster_pyx)]

        return dis, cluster_pyx

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
                for idx, client in group_clients.items():
                    if getattr(client, 'noise_mul', 0):
                        client_key = '_module.' + key
                    else:
                        client_key = key
                    for i in range(m):
                        w = client.data_size * performance[idx][i]
                        group_model_params[key][i] += client.model_params[client_key][i].to(self.device) * w
                        total_weight[i] += w
                for i in range(m):
                    if total_weight[i] > 0:
                        group_model_params[key][i] /= total_weight[i]
            else:
                total_weight = 0  # 计算归一化因子
                for client in group_clients.values():
                    if getattr(client, 'noise_mul', 0):
                        client_key = '_module.' + key
                    else:
                        client_key = key
                    group_model_params[key] += client.model_params[client_key].to(self.device) * client.data_size
                    total_weight += client.data_size

                if total_weight > 0:
                    group_model_params[key] /= total_weight

        group_model.load_state_dict(group_model_params)
        self.gamma *= 0.9
        return group_model