from cluster import *
from utils import get_Vk, get_eps, fed_eval, tensor_dict_to_vector, fuzzy_normalized_mutual_info, nmi_cal
from torch.utils.data import Subset
import wandb
from datetime import datetime
import pandas as pd
from client import DeCflClient, FedAvgClient, FedProxClient, ScaffoldClient, FlexCflClient, FeSemClient, FedRCClientLite
from server import DeCflServer, FedAvgServer, FedProxServer, ScaffoldServer, FlexCflServer, FeSemServer, FedRCServer
import time
import random
from sklearn.metrics import normalized_mutual_info_score
from utils import set_random, create_keras_model, t_sne
from customopacus import PrivacyEngine
from typing import List, Tuple, Dict, Iterable, Optional
import os
from torchvision import transforms
from dataset import ClientDataset
from torch.utils.data import Dataset, DataLoader, random_split


def abl_experiment(train, val, test, global_model, args):
    # 记录实验结果
    column_names = (['sigma'] + ['eps'] + ['sr'] +
                    [f"test_acc{i+1}" for i in range(args.rounds)])
    result = pd.DataFrame(columns=column_names)

    args.pre_cluster = None
    args.primary_aggregation = True

    # 所提方法(DeCFl_Pri.)
    # args.primary_aggregation = True
    args.gc = True
    args.ha = True
    args.experiment = 'DeCFl_Pri.'
    print(args.experiment)
    start_time = time.time()

    res = de_cfl_experiment(copy.deepcopy(global_model), train, val, test, args)  # 形成的分组会在后续实验中共享
    result = pd.concat([result, res])

    end_time = time.time()
    print(f"{args.experiment}_experiment运行时间：{end_time - start_time}秒")

    # 所提方法(DeCFl_no_ha.) 与DeCFl_Pri.共享分组结构
    args.primary_aggregation = True
    args.gc = True
    args.ha = False
    args.experiment = 'DeCFl_no_ha.'
    print(args.experiment)
    start_time = time.time()
    res = de_cfl_experiment(copy.deepcopy(global_model), train, val, test, args)  # 形成的分组会在后续实验中共享
    result = pd.concat([result, res])
    end_time = time.time()
    print(f"{args.experiment}_experiment运行时间：{end_time - start_time}秒")

    # 所提方法(DeCFl_no_gc.) 基于梯度相似性形成分组结构
    args.primary_aggregation = True
    args.gc = False
    args.ha = True
    args.experiment = 'DeCFl_no_gc.'
    print(args.experiment)
    start_time = time.time()
    res = de_cfl_experiment(copy.deepcopy(global_model), train, val, test, args)  # 形成的分组会在后续实验中共享
    result = pd.concat([result, res])
    end_time = time.time()
    print(f"{args.experiment}_experiment运行时间：{end_time - start_time}秒")

    noise = args.train_config['noise']
    result.to_csv(f'./res/{args.dataset}'
                  f'_g1{args.groups1}'
                  f'_g2{args.groups2}'
                  f'_pergroup{args.clients_per_group}'
                  f'_overlap{args.overlap_ratio}'
                  f'_alpha{args.alpha}'
                  f'_noise{noise}'
                  f'_rounds{args.rounds}'
                  f'abl_experiment.csv')

def cmp_experiment(train, val, test, global_model, args, client_arrays_pack):
    # 记录实验结果
    column_names = (['sigma'] + ['eps'] + ['sr'] +
                    [f"test_acc{i+1}" for i in range(args.rounds)])
    result = pd.DataFrame(columns=column_names)

    # 对比实验(FedAvg)
    args.experiment = 'FedAvg'
    print(args.experiment)
    start_time = time.time()
    res = fed_avg_experiment(copy.deepcopy(global_model), train, val, test, args)
    result = pd.concat([result, res])
    end_time = time.time()
    print(f"{args.experiment}_experiment运行时间：{end_time - start_time}秒")

    # 对比实验(FedProx)
    args.experiment = 'FedProx'
    print(args.experiment)
    start_time = time.time()
    res = fed_prox_experiment(copy.deepcopy(global_model), train, val, test, args)
    result = pd.concat([result, res])
    end_time = time.time()
    print(f"{args.experiment}_experiment运行时间：{end_time - start_time}秒")

    # 对比实验(scaffold)
    args.experiment = 'Scaffold'
    print(args.experiment)
    start_time = time.time()
    res = scaffold_experiment(copy.deepcopy(global_model), train, val, test, args)
    result = pd.concat([result, res])
    end_time = time.time()
    print(f"{args.experiment}_experiment运行时间：{end_time - start_time}秒")

    # 对比实验(FeSem)
    args.experiment = 'FeSem'
    print(args.experiment)
    # FeSem参数修正
    ori_rounds = args.rounds
    ori_local_steps = args.train_config.get('local_steps')
    EM_num = ori_local_steps  # 1 or ori_local_steps
    args.rounds = 200 + int((ori_rounds - 1) * EM_num)
    args.train_config['local_steps'] = ori_local_steps // EM_num
    start_time = time.time()
    res = fesem_experiment(copy.deepcopy(global_model), train, val, test, EM_num, args)
    result = pd.concat([result, res])
    end_time = time.time()
    # FeSem参数修正恢复
    args.rounds = ori_rounds
    args.train_config['local_steps'] = ori_local_steps
    print(f"{args.experiment}_experiment运行时间：{end_time - start_time}秒")

    # 对比实验(FlexCfl)
    args.experiment = 'FlexCfl'
    print(args.experiment)
    start_time = time.time()
    res = flex_cfl_experiment(copy.deepcopy(global_model), train, val, test, args)
    result = pd.concat([result, res])
    end_time = time.time()
    print(f"{args.experiment}_experiment运行时间：{end_time - start_time}秒")

    # 对比实验(FedRC)
    # FedRC参数修正
    ori_rounds = args.rounds
    ori_local_steps = args.train_config.get('local_steps')
    EM_num = ori_local_steps  # 1 or ori_local_steps
    args.train_config['local_steps'] = ori_local_steps // EM_num
    args.rounds = 200 // args.train_config['local_steps'] + int((ori_rounds - 1) * EM_num)

    args.experiment = 'FedRC'
    print(args.experiment)
    start_time = time.time()
    res = fedrc_experiment(copy.deepcopy(global_model), test, EM_num, args, client_arrays_pack)
    result = pd.concat([result, res])

    # FedRC参数修正恢复
    args.rounds = ori_rounds
    args.train_config['local_steps'] = ori_local_steps

    # 对比实验(FedPCDP)
    if args.train_config.get('noise') > 0:
        args.experiment = 'FedPCDP'
        print(args.experiment)
        start_time = time.time()
        res = fed_PCDP_experiment(copy.deepcopy(global_model), train, val, test, args)
        result = pd.concat([result, res])
        end_time = time.time()
        print(f"{args.experiment}_experiment运行时间：{end_time - start_time}秒")

    end_time = time.time()
    print(f"{args.experiment}_experiment运行时间：{end_time - start_time}秒")

    noise = args.train_config['noise']
    result.to_csv(f'./res/{args.dataset}'
                  f'_g1{args.groups1}'
                  f'_g2{args.groups2}'
                  f'_pergroup{args.clients_per_group}'
                  f'_overlap{args.overlap_ratio}'
                  f'_alpha{args.alpha}'
                  f'_noise{noise}'
                  f'_rounds{args.rounds}'
                  f'_cmp_experiment.csv')


def noise_experiment(train, val, test, global_model, args):
    column_names = (['sigma'] + ['eps'] + ['sr'] + ['pyx_nmi'] + ['py_js'] +
                    [f"test_acc{i+1}" for i in range(args.rounds)])
    result = pd.DataFrame(columns=column_names)
    experiment = 'DP_noise'
    print(f"{experiment}_experiment")
    start_time = time.time()
    args.pre_cluster = None
    args.primary_aggregation = True
    args.gc = True
    args.ha = True
    args.experiment = 'DeCFl_Pri.'
    args.sampling_rate = 1.0
    for noise in [0.5, 1.0, 1.5, 2.0]:
        args.pre_cluster = None  # 聚类结构不共享，每次实验更新
        args.train_config['noise'] = noise
        res = de_cfl_experiment(copy.deepcopy(global_model), train, val, test, args)
        result = pd.concat([result, res])
    end_time = time.time()
    print(f"{experiment}_experiment运行时间：{end_time - start_time}秒")
    result.to_csv(f'./res/{args.dataset}'
                  f'_g1{args.groups1}'
                  f'_g2{args.groups2}'
                  f'_pergroup{args.clients_per_group}'
                  f'_alpha{args.alpha}'
                  f'_rounds{args.rounds}'
                  f'_DP_experiment.csv')

def sampling_experiment(train, val, test, global_model, args):
    column_names = (['sigma'] + ['eps'] + ['sr'] + ['pyx_nmi'] + ['py_js'] +
                    [f"test_acc{i+1}" for i in range(args.rounds)])
    result = pd.DataFrame(columns=column_names)
    experiment = 'sample_rate'
    print(f"{experiment}_experiment")
    start_time = time.time()
    args.pre_cluster = None
    args.primary_aggregation = True
    args.gc = True
    args.ha = True
    args.experiment = 'DeCFl_Pri.'
    args.train_config['noise'] = 0
    for sr in [1.0, 0.6, 0.3, 0.1]:  # 聚类结构共享
        args.sampling_rate = sr
        res = de_cfl_experiment(copy.deepcopy(global_model), train, val, test, args)
        result = pd.concat([result, res])
    end_time = time.time()
    print(f"{experiment}_experiment运行时间：{end_time - start_time}秒")

    result.to_csv(f'./res/{args.dataset}'
                  f'_g1{args.groups1}'
                  f'_g2{args.groups2}'
                  f'_pergroup{args.clients_per_group}'
                  f'_alpha{args.alpha}'
                  f'_rounds{args.rounds}'
                  f'_sr_experiment.csv')

def de_cfl(server, clients, args, indicate_dataset):
    eps = [0]
    res = np.zeros(args.rounds)
    init_model = copy.deepcopy(server.cluster_models[0])
    vk = get_Vk(args, indicate_dataset, init_model) if args.k > 0 else None
    noise = args.train_config.get('noise')
    tmp = args.ha
    for r in range(args.rounds):
        # if r < 1:
        #     args.ha = True
        # else:
        #     args.ha = tmp
        if r == 0:
            # one-shot cluster, clients_pre_py已进行概率归一化
            grads, clients_pre_py, py_js = server.train_clients(clients, sampling_rate=1.0, vk=vk, r=r)
            # # t-sne
            # t_sne(grads, [0,3,6,1,4,7,2,5,8])
            print('py_js: ', py_js)
            if args.dataset == 'FashionMnist':
                grad_split = [-16010, -10]
            elif args.dataset == 'Mnist':
                grad_split = [-8010, -10]
            elif args.dataset == 'Cifar10':
                grad_split = [-(6 * 6 * 64 * 10) - 10, -10]
            else:
                raise NotImplementedError
            if not args.gc:
                cluster_pyx, cluster_py = server.client_grouping(grads, grad_split, clients_pre_py, k=args.K, grad_correction=args.gc)
            elif args.pre_cluster == None:
                cluster_pyx, cluster_py = server.client_grouping(grads, grad_split, clients_pre_py, k=args.K, grad_correction=args.gc)
                args.pre_cluster = [cluster_pyx, cluster_py]
            else:
                cluster_pyx = args.pre_cluster[0]
                cluster_py = args.pre_cluster[1]

            # 选择组内聚合层级
            if args.primary_aggregation:
                cluster = cluster_pyx
            else:
                cluster = [ll for l in cluster_py for ll in l]

            sample_clients_idx = server.sampled_clients
            group_clients_idx = [[c for c in cls if c in sample_clients_idx] for cls in cluster]
            for i in range(len(cluster)):
                idx = group_clients_idx[i]
                group_clients = clients[idx]
                if args.ha:
                    group_model = server.hierarchical_aggregation(dict(zip(idx, group_clients)), clients_pre_py)
                else:
                    group_model = server.avg_aggregate(group_clients)
                server.cluster_models[i] = group_model
                for k in cluster[i]:
                    server.client_groups[k] = i
        else:
            server.train_clients(clients, sampling_rate=args.sampling_rate, vk=None, r=r)
            sample_clients_idx = server.sampled_clients
            group_clients_idx = [[c for c in cls if c in sample_clients_idx] for cls in cluster]
            for i in range(len(cluster)):
                idx = group_clients_idx[i]
                if len(idx) > 0:
                    group_clients = clients[idx]
                    if args.ha:
                        group_model = server.hierarchical_aggregation(dict(zip(idx, group_clients)), clients_pre_py)
                    else:
                        group_model = server.avg_aggregate(group_clients)
                    server.cluster_models[i] = group_model
        if (r+1) % 1 == 0:
            val_acc, test_acc = fed_eval(server, clients)
            res[r] = test_acc
            print(f'{server.name}_noise{noise}_sr{args.sampling_rate}_test_acc{r}: {res[r]}')

    print('cluster: ', cluster)
    print(f'{server.name}_noise{noise}_sr{args.sampling_rate}_test_acc{r}: {res[r]}')

    if args.train_config.get('noise') > 0:
        eps = get_eps(clients, server.sample_account, args)

    return np.max(eps), res, cluster, py_js


def de_cfl_experiment(global_model, train_loader, val_loader, test_loader, args):
    alg = args.experiment
    # 构造小规模探测数据集
    indicate_dataset = Subset(test_loader[0].dataset, torch.randperm(len(test_loader[0].dataset))[:500])
    # 构造客户端
    clients = [DeCflClient(id_num=k,
                           train_d=train_loader[k],
                           val_d=val_loader[k],
                           device=args.device,
                           train_config=args.train_config
                           ) for k in range(args.groups1 * args.groups2 * args.clients_per_group)]
    clients = np.array(clients)
    beta = 0.001 * np.mean([client.data_size for client in clients])
    server = DeCflServer(copy.deepcopy(global_model), test_loader, alg, args.n_clients, args.device, beta)

    index_names = [alg]
    column_names = (['sigma'] + ['eps'] + ['sr'] + ['pyx_nmi'] + ['py_js'] +
                    [f"test_acc{i+1}" for i in range(args.rounds)])
    res = pd.DataFrame(index=index_names, columns=column_names)
    eps, np_res, cluster, py_js = de_cfl(server, copy.deepcopy(clients), args, indicate_dataset)

    pyx_nmi = nmi_cal(cluster, args.prior_cls)

    res.loc[alg, 'sigma'] = args.train_config.get('noise')
    res.loc[alg, 'sr'] = args.sampling_rate
    res.loc[alg, 'eps'] = eps
    for i in range(args.rounds):
        res.loc[alg, f'test_acc{i+1}'] = np_res[i]
    res.loc[alg, 'pyx_nmi'] = pyx_nmi
    res.loc[alg, 'py_js'] = py_js
    print('pyx_nmi: ', pyx_nmi)
    return res


def flex_cfl_experiment(global_model, train_loader, val_loader, test_loader, args):
    alg = args.experiment
    # 构造客户端
    clients = [FlexCflClient(id_num=k,
                             train_d=train_loader[k],
                             val_d=val_loader[k],
                             device=args.device,
                             train_config=args.train_config
                             ) for k in range(args.groups1 * args.groups2 * args.clients_per_group)]
    clients = np.array(clients)
    server = FlexCflServer(copy.deepcopy(global_model), test_loader, alg, args.n_clients, args.device, args.train_config)

    index_names = [alg]
    column_names = (['sigma'] + ['eps'] + ['sr'] +
                    [f"test_acc{i+1}" for i in range(args.rounds)])
    res = pd.DataFrame(index=index_names, columns=column_names)
    res.loc[alg, 'sigma'] = 0
    res.loc[alg, 'eps'] = 0
    res.loc[alg, 'sr'] = args.sampling_rate
    for r in range(args.rounds):
        if r == 0:
            ori_local_steps = clients[0].local_steps
            for i in range(len(clients)):
                clients[i].local_steps = 200

            grads = server.train_clients(clients, sampling_rate=1.0)

            for i in range(len(clients)):
                clients[i].local_steps = ori_local_steps
        else:
            grads = server.train_clients(clients, sampling_rate=1.0)

        if r == 0:
            cluster = server.group_cold_start(grads, k=args.K)

        sample_clients_idx = server.sampled_clients
        group_clients_idx = [[c for c in cls if c in sample_clients_idx] for cls in cluster]
        for i in range(len(cluster)):
            idx = group_clients_idx[i]
            if len(idx) > 0:
                group_clients = clients[idx]
                group_model = server.IntraGroupUpdate(group_clients)  # 组内聚合
                server.cluster_models[i] = group_model

        server.InterGroupAggregation()  # 组间聚合
        for i in range(len(cluster)):
            for k in cluster[i]:
                server.client_groups[k] = i

        val_acc, test_acc = fed_eval(server, clients)
        res.loc[alg, f"test_acc{r+1}"] = test_acc

    print('cluster: ', cluster)
    print(f'{alg}_test_acc_{r}: {test_acc}')

    pyx_nmi = nmi_cal(cluster, args.prior_cls)
    if args.train_config.get('noise') > 0:
        eps = get_eps(clients, server.sample_account, args)
        res.loc[alg, 'eps'] = np.max(eps)
    res.loc[alg, 'pyx_nmi'] = pyx_nmi
    res.loc[alg, 'py_js'] = None
    print('pyx_nmi: ', pyx_nmi)
    return res


def fesem_experiment(global_model, train_loader, val_loader, test_loader, EM_num, args):
    alg = args.experiment
    # 构造客户端
    clients = [FeSemClient(id_num=k,
                           train_d=train_loader[k],
                           val_d=val_loader[k],
                           device=args.device,
                           train_config=args.train_config
                           ) for k in range(args.groups1 * args.groups2 * args.clients_per_group)]
    clients = np.array(clients)
    server = FeSemServer(copy.deepcopy(global_model), test_loader, alg, args.n_clients, args.device, args.train_config)

    index_names = [alg]
    column_names = (['sigma'] + ['eps'] + ['sr'] +
                    [f"test_acc{i+1}" for i in range(args.rounds//EM_num)])
    res = pd.DataFrame(index=index_names, columns=column_names)
    res.loc[alg, 'sigma'] = 0
    res.loc[alg, 'eps'] = 0
    res.loc[alg, 'sr'] = args.sampling_rate
    N = args.n_clients
    K = args.K
    for r in range(args.rounds):
        # E-step
        if r == 0:
            dws = np.zeros([N, K])
            client_params = server.train_clients(clients, sampling_rate=1.0)
            cls_states = [client_params[i] for i in list(range(0, N, N//K))]
            for i in range(N):
                client_state = client_params[i]
                for j in range(K):
                    cls_state = cls_states[j]
                    dws[i][j] = torch.norm(client_state - cls_state, p=2).item()
            min_k = np.argmin(dws, axis=1)
            cluster = [np.where(min_k == i)[0] for i in set(min_k)]

        else:
            dws = np.zeros([N, K])
            client_params = server.train_clients(clients, sampling_rate=1.0)
            for i in range(args.n_clients):
                client_state = client_params[i]
                for j in range(args.K):
                    cls_state = tensor_dict_to_vector(server.cluster_models[j].state_dict())
                    dws[i][j] = torch.norm(client_state - cls_state, p=2).item()
            min_k = np.argmin(dws, axis=1)
            cluster = [np.where(min_k == i)[0] for i in set(min_k)]

        # M-step
        sample_clients_idx = server.sampled_clients
        group_clients_idx = [[c for c in cls if c in sample_clients_idx] for cls in cluster]
        for i in range(len(cluster)):
            idx = group_clients_idx[i]
            if len(idx) > 0:
                group_clients = clients[idx]
                group_model = server.avg_aggregate(group_clients)  # 组内聚合
                server.cluster_models[i] = group_model

        for i in range(len(cluster)):
            for k in cluster[i]:
                server.client_groups[k] = i

        if (r+1) % EM_num == 0:
            val_acc, test_acc = fed_eval(server, clients)

            res.loc[alg, f"test_acc{(r+1) // EM_num}"] = test_acc

    print('cluster: ', cluster)
    print(f'{alg}_test_acc_{(r+1) % EM_num}: {test_acc}')
    while len(cluster) < K:
        cluster.append([])
    pyx_nmi = nmi_cal(cluster, args.prior_cls)
    if args.train_config.get('noise') > 0:
        eps = get_eps(clients, server.sample_account, args)
        res.loc[alg, 'eps'] = np.max(eps)

    res.loc[alg, 'pyx_nmi'] = pyx_nmi
    res.loc[alg, 'py_js'] = None
    print('pyx_nmi: ', pyx_nmi)
    return res


def fed_PCDP_experiment(global_model, train_loader, val_loader, test_loader, args):
    alg = args.experiment
    indicate_dataset = Subset(test_loader[0].dataset, torch.randperm(len(test_loader[0].dataset))[:500])
    # 构造客户端
    clients = [FedAvgClient(id_num=k,
                            train_d=train_loader[k],
                            val_d=val_loader[k],
                            device=args.device,
                            train_config=args.train_config
                            ) for k in range(args.groups1 * args.groups2 * args.clients_per_group)]
    clients = np.array(clients)
    server = FedAvgServer(copy.deepcopy(global_model), test_loader, alg, args.n_clients, args.device, args.train_config)

    index_names = [alg]
    column_names = (['sigma'] + ['eps'] + ['sr'] +
                    [f"test_acc{i+1}" for i in range(args.rounds)])
    res = pd.DataFrame(index=index_names, columns=column_names)
    res.loc[alg, 'sigma'] = 0
    res.loc[alg, 'eps'] = 0
    res.loc[alg, 'sr'] = args.sampling_rate

    if args.k > 0 and args.train_config.get('noise') > 0:
        Vk = get_Vk(args, indicate_dataset, copy.deepcopy(global_model))
    else:
        Vk = None

    for r in range(args.rounds):
        if r == 0:
            ori_local_steps = clients[0].local_steps
            for i in range(len(clients)):
                clients[i].local_steps = 200

            server.train_clients(clients, sampling_rate=args.sampling_rate, Vk=Vk)

            for i in range(len(clients)):
                clients[i].local_steps = ori_local_steps
        else:
            server.train_clients(clients, sampling_rate=args.sampling_rate, Vk=Vk)

        sample_clients_idx = server.sampled_clients
        group_clients = clients[sample_clients_idx]
        group_model = server.avg_aggregate(group_clients)
        server.cluster_models[0] = group_model
        val_acc, test_acc = fed_eval(server, clients)

        if args.k > 0 and args.train_config.get('noise') > 0:
            Vk = get_Vk(args, indicate_dataset, copy.deepcopy(group_model))

        res.loc[alg, f"test_acc{r+1}"] = test_acc

    print(f'{alg}_test_acc_{r}: {test_acc}')
    if args.train_config.get('noise') > 0:
        eps = get_eps(clients, server.sample_account, args)
        res.loc[alg, 'eps'] = np.max(eps)
    res.loc[alg, 'pyx_nmi'] = normalized_mutual_info_score(np.zeros(args.n_clients).astype('int'), args.prior_cls)
    res.loc[alg, 'py_js'] = None

    return res


def fed_avg_experiment(global_model, train_loader, val_loader, test_loader, args):
    alg = args.experiment
    # 构造客户端
    clients = [FedAvgClient(id_num=k,
                            train_d=train_loader[k],
                            val_d=val_loader[k],
                            device=args.device,
                            train_config=args.train_config
                            ) for k in range(args.groups1 * args.groups2 * args.clients_per_group)]
    clients = np.array(clients)
    server = FedAvgServer(copy.deepcopy(global_model), test_loader, alg, args.n_clients, args.device, args.train_config)

    index_names = [alg]
    column_names = (['sigma'] + ['eps'] + ['sr'] +
                    [f"test_acc{i+1}" for i in range(args.rounds)])
    res = pd.DataFrame(index=index_names, columns=column_names)
    res.loc[alg, 'sigma'] = 0
    res.loc[alg, 'eps'] = 0
    res.loc[alg, 'sr'] = args.sampling_rate
    for r in range(args.rounds):
        if r == 0:
            ori_local_steps = clients[0].local_steps
            for i in range(len(clients)):
                clients[i].local_steps = 200

            server.train_clients(clients, sampling_rate=args.sampling_rate)

            for i in range(len(clients)):
                clients[i].local_steps = ori_local_steps
        else:
            server.train_clients(clients, sampling_rate=args.sampling_rate)
        sample_clients_idx = server.sampled_clients
        group_clients = clients[sample_clients_idx]
        group_model = server.avg_aggregate(group_clients)
        server.cluster_models[0] = group_model
        val_acc, test_acc = fed_eval(server, clients)

        res.loc[alg, f"test_acc{r+1}"] = test_acc

    print(f'{alg}_test_acc_{r}: {test_acc}')
    if args.train_config.get('noise') > 0:
        eps = get_eps(clients, server.sample_account, args)
        res.loc[alg, 'eps'] = np.max(eps)
    res.loc[alg, 'pyx_nmi'] = normalized_mutual_info_score(np.zeros(args.n_clients).astype('int'), args.prior_cls)
    res.loc[alg, 'py_js'] = None

    return res


def fed_prox_experiment(global_model, train_loader, val_loader, test_loader, args):
    alg = args.experiment
    # 构造客户端
    clients = [FedProxClient(id_num=k,
                             train_d=train_loader[k],
                             val_d=val_loader[k],
                             device=args.device,
                             train_config=args.train_config,
                             mu=0.05) for k in range(args.groups1 * args.groups2 * args.clients_per_group)]
    clients = np.array(clients)
    server = FedProxServer(copy.deepcopy(global_model), test_loader, alg, args.n_clients, args.device, args.train_config)

    index_names = [alg]
    column_names = (['sigma'] + ['eps'] + ['sr'] + ['pyx_nmi'] + ['py_js'] +
                    [f"test_acc{i+1}" for i in range(args.rounds)])
    res = pd.DataFrame(index=index_names, columns=column_names)
    res.loc[alg, 'sigma'] = 0
    res.loc[alg, 'eps'] = 0
    res.loc[alg, 'sr'] = args.sampling_rate
    for r in range(args.rounds):
        if r == 0:
            ori_local_steps = clients[0].local_steps
            for i in range(len(clients)):
                clients[i].local_steps = 200

            server.train_clients(clients, sampling_rate=args.sampling_rate)

            for i in range(len(clients)):
                clients[i].local_steps = ori_local_steps
        else:
            server.train_clients(clients, sampling_rate=args.sampling_rate)
        sample_clients_idx = server.sampled_clients
        group_clients = clients[sample_clients_idx]
        group_model = server.avg_aggregate(group_clients)
        server.cluster_models[0] = group_model
        val_acc, test_acc = fed_eval(server, clients)

        res.loc[alg, f"test_acc{r+1}"] = test_acc

    print(f'{alg}_test_acc_{r}: {test_acc}')
    if args.train_config.get('noise') > 0:
        eps = get_eps(clients, server.sample_account, args)
        res.loc[alg, 'eps'] = np.max(eps)
    res.loc[alg, 'pyx_nmi'] = normalized_mutual_info_score(np.zeros(args.n_clients).astype('int'), args.prior_cls)
    res.loc[alg, 'py_js'] = None
    return res


def scaffold_experiment(global_model, train_loader, val_loader, test_loader, args):
    alg = args.experiment
    # 构造客户端
    clients = [ScaffoldClient(id_num=k,
                              train_d=train_loader[k],
                              val_d=val_loader[k],
                              device=args.device,
                              train_config=args.train_config,
                              model=global_model
                              ) for k in range(args.groups1 * args.groups2 * args.clients_per_group)]
    clients = np.array(clients)
    server = ScaffoldServer(copy.deepcopy(global_model), test_loader, alg, args.n_clients, args.device, args.train_config)

    index_names = [alg]
    column_names = (['sigma'] + ['eps'] + ['sr'] + ['pyx_nmi'] + ['py_js'] +
                    [f"test_acc{i+1}" for i in range(args.rounds)])
    res = pd.DataFrame(index=index_names, columns=column_names)
    res.loc[alg, 'sigma'] = 0
    res.loc[alg, 'eps'] = 0
    res.loc[alg, 'sr'] = args.sampling_rate
    for r in range(args.rounds):
        if r == 0:
            ori_local_steps = clients[0].local_steps
            for i in range(len(clients)):
                clients[i].local_steps = 200

            server.train_clients(clients, sampling_rate=args.sampling_rate)

            for i in range(len(clients)):
                clients[i].local_steps = ori_local_steps
        else:
            server.train_clients(clients, sampling_rate=args.sampling_rate)
        sample_clients_idx = server.sampled_clients
        group_clients = clients[sample_clients_idx]
        group_model = server.avg_aggregate(group_clients)
        server.cluster_models[0] = group_model
        val_acc, test_acc = fed_eval(server, clients)

        res.loc[alg, f"test_acc{r+1}"] = test_acc

    print(f'{alg}_test_acc_{r}: {test_acc}')
    if args.train_config.get('noise') > 0:
        eps = get_eps(clients, server.sample_account, args)
        res.loc[alg, 'eps'] = np.max(eps)
    res.loc[alg, 'pyx_nmi'] = normalized_mutual_info_score(np.zeros(args.n_clients).astype('int'), args.prior_cls)
    res.loc[alg, 'py_js'] = None
    return res


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
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import warnings
import shutil
import multiprocessing as mp

def fedrc_experiment(global_model, test_loader, EM_num, args, client_arrays_pack):
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    estimate_batches = 1
    alg = args.experiment
    M = args.groups1 * args.groups2 * args.clients_per_group
    K = args.K
    max_workers = 5 # !!!
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
            if rd > 198 and (rd + 1) % EM_num == 0:
                _, test_acc = fed_eval(server, None)
                res.loc[alg, f"test_acc{(rd+1) // EM_num - 10 + 1}"] = test_acc
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
    mp.set_start_method(None, force=True)
    return res


# def fedrc_experiment(global_model, train_loader, val_loader, test_loader, EM_num, args):
#     estimate_batches = 5
#     alg = args.experiment
#     M = args.groups1 * args.groups2 * args.clients_per_group
#     K = args.K
#     device = args.device
#     num_rounds = args.rounds
#     frac = args.sampling_rate
#     client_indices = list(range(M))
#
#     clients = [FedRCClient(id_num=k,
#                            train_d=train_loader[k],
#                            val_d=val_loader[k],
#                            device=args.device,
#                            train_config=args.train_config
#                            ) for k in range(M)]
#
#     cluster_models = {}  # 组标号 -> 类簇模型
#     cluster_models[0] = global_model
#     for i in range(1, args.K):
#         cluster_models[i] = create_keras_model(args.dataset).to(device=device)
#     server = FedRCServer(cluster_models=cluster_models, num_clients=M, K=K, device=device, test_loader=test_loader)
#
#     index_names = []
#     column_names = (['sigma'] + ['eps'] + ['sr'] +
#                     [f"test_acc{i+1}" for i in range(num_rounds//EM_num)])
#     res = pd.DataFrame(index=index_names, columns=column_names)
#     res.loc[alg, 'sigma'] = 0
#     res.loc[alg, 'eps'] = 0
#     res.loc[alg, 'sr'] = frac
#
#     # 保存每个客户端的 ω（可初始化均匀）
#     omega_clients = torch.full((M, K), 1.0 / K, device=args.device)
#
#     for rd in range(num_rounds):
#         start_time = time.time()
#         # 抽样参与客户端
#         m = max(1, int(frac * M))
#         selected = random.sample(client_indices, m)
#         server.sample_account[selected] += 1
#
#         # 1) 服务器下发模型拷贝
#         global_models = server.get_model_copies()
#
#         # 2) 客户端：E 步 + 局部加权训练 -> ∆θ
#         deltas_all_clients = []
#         client_sizes = []
#         for i in selected:
#             # E 步：估计 γ 与 ω（按 batch 序列）
#             gammas_batches, omega_i, _ = clients[i].estimate_gamma_omega_for_client(
#                 global_models, num_classes=10, max_val_batches=estimate_batches
#             )
#             omega_clients[i] = omega_i.detach()
#
#             # 近似做法：把 E 步得到的 gammas_batches 直接用于 local update
#             deltas_i = clients[i].local_weighted_update(
#                 global_models=global_models,
#                 gammas_batches=gammas_batches
#             )
#             deltas_all_clients.append(deltas_i)
#
#             # 记录客户端样本量（用于加权）
#             client_sizes.append(clients[i].len())
#
#         # 3) 服务器聚合（M 步）
#         server.aggregate(deltas_all_clients, client_sizes=client_sizes, lr_g=1.0)
#
#         # 4) 简要评估
#         if (rd + 1) % EM_num == 0:
#             _, test_acc = fed_eval(server, clients)
#             res.loc[alg, f"test_acc{(rd+1) // EM_num}"] = test_acc
#             print((rd+1) // EM_num, test_acc)
#
#         end_time = time.time()
#         print(end_time - start_time)
#     print(f'{alg}_test_acc_{num_rounds}: {test_acc}')
#
#     if args.train_config.get('noise') > 0:
#         eps = get_eps(clients, server.sample_account, args)
#         res.loc[alg, 'eps'] = np.max(eps)
#     return res
