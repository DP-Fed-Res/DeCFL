import gc
import copy
import time
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Subset
from datetime import datetime

from client import *
from server import *
from utils import get_Vk, get_eps, fed_eval, cluster_eval
from customopacus.accountants.utils import get_noise_multiplier
from peft import LoraConfig, get_peft_model, TaskType


def setup_experiment_config(args):
    """初始化 CUDA 内存统计并计算 DP 噪声乘数，生成训练配置"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    steps = int((args.tg - 1) * args.tc + args.tc_1st)
    if args.alg == 'fedrc':
        steps *= args.g1
    if args.eps > 0:
        args.noise = get_noise_multiplier(
            target_epsilon=args.eps,
            target_delta=args.delta,
            sample_rate=args.br,
            steps=steps,
            accountant='prv'
        )
    else:
        args.noise = 0
    print(f'client DPSGD noise multiplier: {args.noise}')

    train_config = {
        'lr_mode': args.lr_mode, 'freeze_1st': args.freeze_1st, 'tc': args.tc,
        'tc_1st': args.tc_1st, 'lr': args.lr, 'clip': args.clip,
        'noise': args.noise, 'device': args.device
    }
    return train_config


def init_results_dataframe(args):
    """初始化用于存储结果的 DataFrame"""
    acc_columns = [f"test_acc{i + 1}" for i in range(args.tg)]
    column_names = (
            ['alg', 'dataset', 'client', 'sigma', 'eps', 'clip', 'sr',
             'Tc_1st', 'Tc', 'pyx_ari', 'py_js',
             'time_cost', 'peak_memory'] + acc_columns
    )
    res = pd.DataFrame(columns=column_names)
    res.loc[0, ['pyx_ari', 'py_js']] = -1
    return res


def finalize_and_save_experiment(res, clients, server, args, start_time):
    """计算最终指标（时间、内存、隐私预算），更新 DataFrame 并保存为 CSV"""
    end_time = time.time()

    # 计算峰值显存
    peak_memory_mb = 0
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated(device=args.device) / (1024 * 1024)
        print(f'peak_memory_mb: {peak_memory_mb}')

    # 验证与获取实际隐私预算
    privacy_budget = ['inf']
    if args.noise is not None and args.noise > 0:
        try:
            privacy_budget = get_eps(clients, server.sample_account, args.delta)
            privacy_budget = max(privacy_budget) if isinstance(privacy_budget, list) else privacy_budget
            eps_error = np.abs(privacy_budget - args.eps)
            if eps_error > 0.01:
                raise ValueError(f'eps_error: {eps_error}, privacy_budget does not match the set privacy budget')
        except Exception as e:
            print(f"Warning: Failed to calculate epsilon: {e}")

    final_eps = max(privacy_budget) if isinstance(privacy_budget, list) else privacy_budget

    # 汇总结果
    res.loc[0].update({
        'dataset': args.dataset,
        'client': args.n_clients,
        'alg': args.alg,
        'sigma': args.noise,
        'eps': final_eps,
        'clip': args.clip,
        'sr': args.sampling_rate,
        'Tc_1st': args.tc_1st,
        'Tc': args.tc,
        'time_cost': end_time - start_time,
        'peak_memory': peak_memory_mb
    })

    # 文件保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_type = f'{args.dataset}_g1-{args.g1}_g2-{args.g2}_per{args.per_group}_alpha{args.alpha}'
    saved_dir = f'./res/{data_type}'
    os.makedirs(saved_dir, exist_ok=True)
    file_name = f'{args.alg}_eps{args.eps}_rounds{args.tg}_{data_type}-{timestamp}.csv'
    res.to_csv(os.path.join(saved_dir, file_name), index=False)

    return res


def fedavg_exp(global_model, train_loader, test_loader, args):
    train_config = setup_experiment_config(args)
    train_config['freeze_1st'] = 0
    res = init_results_dataframe(args)

    clients = np.array([BaseClient(id_num=k, train_d=train_loader[k], train_config=train_config)
                        for k in range(args.n_clients)])
    server = BaseServer(copy.deepcopy(global_model), test_loader, args.alg, args.n_clients, args.device)

    if not server.cluster_models:
        raise ValueError("server.cluster_models cannot be empty.")

    st = time.time()

    # 核心训练循环
    for r in range(args.tg):
        server.train_clients(clients, r=r)

        group_clients = clients[server.sampled_clients]
        group_model = server.avg_aggregate(group_clients)
        server.cluster_models[0] = group_model

        if (r + 1) % 1 == 0:
            test_acc = fed_eval(server, args.device)
            res.loc[0, f'test_acc{r + 1}'] = test_acc
            print(f'[{r + 1}/{args.tg}] {server.name} | Acc: {100 * test_acc:.2f}%')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = finalize_and_save_experiment(res, clients, server, args, st)

    return result


def fedprox_exp(global_model, train_loader, test_loader, args):
    train_config = setup_experiment_config(args)
    train_config['freeze_1st'] = 0
    res = init_results_dataframe(args)

    clients = np.array([FedProxClient(id_num=k, train_d=train_loader[k], train_config=train_config, mu=0.1)
                        for k in range(args.n_clients)])
    server = FedProxServer(copy.deepcopy(global_model), test_loader, args.alg, args.n_clients, args.device)

    if not server.cluster_models:
        raise ValueError("server.cluster_models cannot be empty.")

    st = time.time()

    # 核心训练循环
    for r in range(args.tg):
        server.train_clients(clients, r=r)

        group_clients = clients[server.sampled_clients]
        group_model = server.avg_aggregate(group_clients)
        server.cluster_models[0] = group_model

        if (r + 1) % 1 == 0:
            test_acc = fed_eval(server, args.device)
            res.loc[0, f'test_acc{r + 1}'] = test_acc
            print(f'[{r + 1}/{args.tg}] {server.name} | Acc: {100 * test_acc:.2f}%')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = finalize_and_save_experiment(res, clients, server, args, st)

    return result


def scaffold_exp(global_model, train_loader, test_loader, args):
    train_config = setup_experiment_config(args)
    train_config['freeze_1st'] = 0
    res = init_results_dataframe(args)

    clients = np.array([ScaffoldClient(id_num=k, train_d=train_loader[k], train_config=train_config, model=copy.deepcopy(global_model))
                        for k in range(args.n_clients)])
    server = ScaffoldServer(copy.deepcopy(global_model), test_loader, args.alg, args.n_clients, args.device)

    if not server.cluster_models:
        raise ValueError("server.cluster_models cannot be empty.")

    st = time.time()

    # 核心训练循环
    for r in range(args.tg):
        server.train_clients(clients, r=r)

        group_clients = clients[server.sampled_clients]
        group_model = server.avg_aggregate(group_clients)
        server.cluster_models[0] = group_model

        if (r + 1) % 1 == 0:
            test_acc = fed_eval(server, args.device)
            res.loc[0, f'test_acc{r + 1}'] = test_acc
            print(f'[{r + 1}/{args.tg}] {server.name} | Acc: {100 * test_acc:.2f}%')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = finalize_and_save_experiment(res, clients, server, args, st)

    return result


def fed_pcdp_exp(global_model, train_loader, test_loader, args):
    train_config = setup_experiment_config(args)
    train_config['freeze_1st'] = 0
    res = init_results_dataframe(args)

    clients = np.array([BaseClient(id_num=k, train_d=train_loader[k], train_config=train_config)
                        for k in range(args.n_clients)])
    server = BaseServer(copy.deepcopy(global_model), test_loader, args.alg, args.n_clients, args.device)

    if not server.cluster_models:
        raise ValueError("server.cluster_models cannot be empty.")

    st = time.time()

    # 核心训练循环
    for r in range(args.tg):
        indicate_dataset = Subset(test_loader[0].dataset, torch.randperm(len(test_loader[0].dataset))[:500])
        vk = get_Vk(indicate_dataset, copy.deepcopy(server.cluster_models[0]), args.k, args.lr_mode, args.lr,
                    args.device, freeze=False) if args.eps > 0 else None

        server.train_clients(clients, r=r, sampling_rate=1.0, vk=vk)

        group_clients = clients[server.sampled_clients]
        group_model = server.avg_aggregate(group_clients)
        server.cluster_models[0] = group_model

        if (r + 1) % 1 == 0:
            test_acc = fed_eval(server, args.device)
            res.loc[0, f'test_acc{r + 1}'] = test_acc
            print(f'[{r + 1}/{args.tg}] {server.name} | Acc: {100 * test_acc:.2f}%')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = finalize_and_save_experiment(res, clients, server, args, st)

    return result


def decfl_exp(global_model, train_loader, test_loader, args):
    train_config = setup_experiment_config(args)
    res = init_results_dataframe(args)

    # DeCFL 专属梯度分割配置
    grad_split_map = {
        'FashionMnist': [-(1600 * 10) - 10, -10],
        'Mnist': [-(800 * 10) - 10, -10],
        'Cifar10': [-(128 * 10) - 10, -10],
        'text-distill': [-(768 * 10) - 10, -10],
        'text-roberta': [-(1024 * 10) - 10, -10]
    }
    if args.dataset not in grad_split_map:
        raise NotImplementedError(f"Dataset {args.dataset} not supported")
    grad_split = grad_split_map[args.dataset]

    clients = np.array([DeCflClient(id_num=k, train_d=train_loader[k], train_config=train_config)
                        for k in range(args.n_clients)])

    beta = 0.001 * np.mean([client.data_size for client in clients])
    server = DeCflServer(copy.deepcopy(global_model), test_loader, args.alg, args.n_clients, args.device, beta)

    if not server.cluster_models:
        raise ValueError("server.cluster_models cannot be empty.")

    st = time.time()

    # VK 初始化
    indicate_dataset = Subset(test_loader[0].dataset, torch.randperm(len(test_loader[0].dataset))[:500])
    vk = get_Vk(indicate_dataset, copy.deepcopy(server.cluster_models[0]), args.k, args.lr_mode, args.lr,
                args.device, args.freeze_1st) if (args.k > 0 and args.eps > 0) else None

    cluster_pyx = None

    # 核心训练循环
    for r in range(args.tg):
        if r == 0:
            grads, clients_pre_py, py_js = server.train_clients(clients, r=r, sampling_rate=1.0, vk=vk)
            print('py_js: ', py_js)

            dis, cluster_pyx = server.client_grouping(grads, grad_split, clients_pre_py, k=args.g1,
                                                      grad_correction=args.gc)
            pyx_ari = cluster_eval(cluster_pyx, args.prior_cls)
            print('pyx_ari: ', pyx_ari)
            print('cluster: ', cluster_pyx)

            for i, group_indices in enumerate(cluster_pyx):
                for client_idx in group_indices:
                    server.client_groups[client_idx] = i

            res.loc[0, 'pyx_ari'] = pyx_ari
            res.loc[0, 'py_js'] = py_js
        else:
            server.train_clients(clients, r=r, sampling_rate=args.sampling_rate, vk=None)
            if cluster_pyx is None:
                raise RuntimeError("Clustering (cluster_pyx) was not performed in round 0.")

        # 聚合阶段
        sample_clients_idx = server.sampled_clients
        group_clients_idx = [[c for c in cls if c in sample_clients_idx] for cls in cluster_pyx]

        for i, idx in enumerate(group_clients_idx):
            if not idx: continue

            if args.ha:
                client_dict = {cid: clients[cid] for cid in idx}
                group_model = server.hierarchical_aggregation(client_dict, clients_pre_py)
            else:
                group_clients = [clients[k] for k in idx]
                group_model = server.avg_aggregate(group_clients)

            server.cluster_models[i] = group_model

        if (r + 1) % 1 == 0:
            test_acc = fed_eval(server, args.device)
            res.loc[0, f'test_acc{r + 1}'] = test_acc
            print(f'[{r + 1}/{args.tg}] {server.name} | Acc: {100 * test_acc:.2f}%')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return finalize_and_save_experiment(res, clients, server, args, st)


def noise_exp(config_list, global_model, train_loader, test_loader, args):
    column_names = (['alg'] + ['dataset'] + ['client'] + ['sigma'] + ['eps'] + ['clip'] + ['sr'] + ['first_local_step'] + ['local_step'] +
                    ['pyx_nmi'] + ['pyx_ari'] + ['py_js'] + ['time_cost'] + ['peak_memory'] +
                    [f"test_acc{i+1}" for i in range(args.tg)])
    result = pd.DataFrame(columns=column_names)
    exp = 'noise'

    args.sampling_rate = 1.0

    for eps, clip in config_list:
        print(f"\n{exp}_eps_{eps}_exp\n")
        args.eps = eps
        args.train_config['clip'] = clip
        if args.eps > 0:
            steps = int((args.tg-1) * args.train_config['local_steps'] + args.train_config['first_local_steps'])
            if args.alg == 'fedrc':
                steps *= args.groups1
            args.train_config['noise'] = get_noise_multiplier(
                target_epsilon=eps,
                target_delta=args.delta,
                sample_rate=args.train_config['batch_ratio'],
                steps=steps,
                accountant='prv'
            )
        else:
            args.train_config['noise'] = 0
        print('sigma:', args.train_config['noise'])

        if args.alg in ['decfl', 'decfl-wo-ha', 'decfl-wo-gc']:
            res = decfl_exp(copy.deepcopy(global_model), train_loader, test_loader, args)
        elif args.alg == 'fedavg':
            res = fedavg_exp(copy.deepcopy(global_model), train_loader, test_loader, args)
        elif args.alg == 'fedprox':
            res = fedprox_exp(copy.deepcopy(global_model), train_loader, test_loader, args)
        elif args.alg == 'scaffold':
            res = scaffold_exp(copy.deepcopy(global_model), train_loader, test_loader, args)
        elif args.alg == 'flexcfl':
            res = flexcfl_exp(copy.deepcopy(global_model), train_loader, test_loader, args)
        elif args.alg == 'fesem':
            res = fesem_exp(copy.deepcopy(global_model), train_loader, test_loader, args)
        elif args.alg == 'fed_pcdp':
            res = fed_pcdp_exp(copy.deepcopy(global_model), train_loader, test_loader, args)
        elif args.alg == 'fedrc':
            res = fedrc_exp(copy.deepcopy(global_model), train_loader, test_loader, args)
        else:
            raise Exception
        result = pd.concat([result, res])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result.to_csv(f'./res/{args.dataset}'
                  f'_g1{args.alg}'
                  f'_g1{args.groups1}'
                  f'_g2{args.groups2}'
                  f'_alpha{args.alpha}'
                  f'_rounds{args.tg}'
                  f'_{exp}_exp'
                  f'_{timestamp}.csv')


def heterogeneous_noise_exp(global_model, train_loader, test_loader, client_eps_list, args):
    args.gc = True
    args.ha = True
    args.sampling_rate = 1.0

    torch.cuda.reset_peak_memory_stats()

    alg = 'decfl'
    noise = args.train_config.get('noise')

    if args.dataset == 'FashionMnist':
        grad_split = [-(1600 * 10) - 10, -10]
    elif args.dataset == 'Mnist':
        grad_split = [-(800 * 10) - 10, -10]
    elif args.dataset == 'Cifar10':
        grad_split = [-(128 * 10) - 10, -10]
    elif args.dataset == "text-distill":
        grad_split = [-(768 * 10) - 10, -10]
    elif args.dataset == "text-roberta":
        grad_split = [-(1024 * 10) - 10, -10]
    else:
        raise NotImplementedError

    # generate client and server
    clients = []
    args.eps = np.mean(client_eps_list)
    for k in range(args.n_clients):
        args.train_config['noise'] = get_noise_multiplier(
            target_epsilon=client_eps_list[k],
            target_delta=args.delta,
            sample_rate=args.train_config['batch_ratio'],
            steps=int((args.tg - 1) * args.train_config['local_steps'] + args.train_config['first_local_steps']),
            accountant='prv'
        )
        clients.append(DeCflClient(id_num=k, train_d=train_loader[k], device=args.device, train_config=args.train_config))
    clients = np.array(clients)
    beta = 0.001 * np.mean([client.data_size for client in clients])
    server = DeCflServer(copy.deepcopy(global_model), test_loader, alg, args.n_clients, args.device, beta)

    # run
    st = time.time()
    if not server.cluster_models:
        raise ValueError("server.cluster_models cannot be empty.")
    indicate_dataset = Subset(test_loader[0].dataset, torch.randperm(len(test_loader[0].dataset))[:500])
    if args.k > 0 and args.eps > 0 and args.dataset != 'text-distill':
        vk = get_Vk(args, indicate_dataset, copy.deepcopy(server.cluster_models[0]))
    else:
        vk = None

    column_names = (['alg'] + ['dataset'] + ['client'] + ['sigma'] + ['eps'] + ['clip'] + ['sr'] + ['first_local_step'] + ['local_step'] +
                    ['pyx_nmi'] + ['pyx_ari'] + ['py_js'] + ['time_cost'] + ['peak_memory'] +
                    [f"test_acc{i+1}" for i in range(args.tg)])
    res = pd.DataFrame(columns=column_names)

    for r in range(args.tg):
        if args.debug and r >= 1:
            continue
        # client training
        if r == 0:
            grads, clients_pre_py, py_js = server.train_clients(clients, sampling_rate=1.0, vk=vk, r=r)
            print('py_js: ', py_js)

            dis, cluster_pyx = server.client_grouping(grads, grad_split, clients_pre_py, k=args.groups1, grad_correction=args.gc)

            pyx_nmi, pyx_ari = cluster_eval(cluster_pyx, args.prior_cls)
            print('pyx_nmi: ', pyx_nmi)
            print('pyx_ari: ', pyx_ari)
            print('cluster: ', cluster_pyx)

            # 初始化客户端所属组映射
            for i, group_indices in enumerate(cluster_pyx):
                for client_idx in group_indices:
                    server.client_groups[client_idx] = i

            res.loc[0, 'pyx_nmi'] = pyx_nmi
            res.loc[0, 'pyx_ari'] = pyx_ari
            res.loc[0, 'py_js'] = py_js
        else:
            server.train_clients(clients, sampling_rate=args.sampling_rate, vk=None, r=r)
            if cluster_pyx is None:
                raise RuntimeError("Clustering (cluster_pyx) was not performed in round 0.")

        # aggregation
        sample_clients_idx = server.sampled_clients
        group_clients_idx = [[c for c in cls if c in sample_clients_idx] for cls in cluster_pyx]
        for i, idx in enumerate(group_clients_idx):
            if not idx:
                continue
            # 执行聚合
            if args.ha:
                client_dict = {cid: clients[cid] for cid in idx}
                group_model = server.hierarchical_aggregation(client_dict, clients_pre_py)
            else:
                group_clients = [clients[k] for k in idx]
                group_model = server.avg_aggregate(group_clients)

            server.cluster_models[i] = group_model

        if (r+1) % 1 == 0:
            test_acc = fed_eval(server, clients, args.device)
            res.loc[0, f'test_acc{r+1}'] = test_acc
            print(f'{server.name}_noise{noise}_sr{args.sampling_rate}_acc{r+1}: {test_acc}')

        torch.cuda.empty_cache()
        gc.collect()

    end = time.time()

    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    if args.train_config.get('noise') > 0:
        eps = get_eps(clients, server.sample_account, args)
    else:
        eps = ['inf']

    res.loc[0, 'dataset'] = args.dataset
    res.loc[0, 'client'] = args.n_clients
    res.loc[0, 'alg'] = alg
    res.loc[0, 'sigma'] = args.train_config.get('noise')
    res.loc[0, 'eps'] = max(eps)
    res.loc[0, 'clip'] = args.train_config['clip']
    res.loc[0, 'sr'] = args.sampling_rate
    res.loc[0, 'first_local_step'] = args.train_config.get('first_local_steps')
    res.loc[0, 'local_step'] = args.train_config.get('local_steps')
    res.loc[0, 'time_cost'] = end - st
    res.loc[0, 'peak_memory'] = peak_memory_mb

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res.to_csv(f'./res/{args.dataset}'
               f'_g1{args.groups1}'
               f'_g2{args.groups2}'
               f'_alpha{args.alpha}'
               f'_rounds{args.tg}'
               f'_heterogeneous_exp'
               f'_{timestamp}.csv')

    return res


def flexcfl_exp(global_model, train_loader, test_loader, args):
    train_config = setup_experiment_config(args)
    res = init_results_dataframe(args)

    clients = np.array([FlexCflClient(id_num=k, train_d=train_loader[k], train_config=train_config)
                        for k in range(args.n_clients)])
    server = FlexCflServer(copy.deepcopy(global_model), test_loader, args.alg, args.n_clients, args.device)

    if not server.cluster_models:
        raise ValueError("server.cluster_models cannot be empty.")

    st = time.time()
    for r in range(args.tg):
        grads = server.train_clients(clients, r=r)
        if r == 0:
            cluster = server.group_cold_start(grads, k=args.g1)
            pyx_ari = cluster_eval(cluster, args.prior_cls)
            res.loc[0, 'pyx_ari'] = pyx_ari
            print('pyx_ari: ', pyx_ari)
            print('cluster: ', cluster)

        sample_clients_idx = server.sampled_clients
        group_clients_idx = [[c for c in cls if c in sample_clients_idx] for cls in cluster]
        for i in range(len(cluster)):
            idx = group_clients_idx[i]
            if len(idx) > 0:
                group_clients = clients[idx]
                group_model = server.avg_aggregate(group_clients)  # 组内聚合
                server.cluster_models[i] = group_model

        server.InterGroupAggregation()  # 组间聚合
        for i in range(len(cluster)):
            for k in cluster[i]:
                server.client_groups[k] = i

        if (r + 1) % 1 == 0:
            test_acc = fed_eval(server, args.device)
            res.loc[0, f'test_acc{r + 1}'] = test_acc
            print(f'[{r + 1}/{args.tg}] {server.name} | Acc: {100 * test_acc:.2f}%')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return finalize_and_save_experiment(res, clients, server, args, st)


def fesem_exp(global_model, train_loader, test_loader, args):
    train_config = setup_experiment_config(args)
    train_config['freeze_1st'] = 0
    res = init_results_dataframe(args)

    clients = np.array([FeSemClient(id_num=k, train_d=train_loader[k], train_config=train_config)
                        for k in range(args.n_clients)])
    server = FeSemServer(copy.deepcopy(global_model), test_loader, args.alg, args.n_clients, args.device)

    if not server.cluster_models:
        raise ValueError("server.cluster_models cannot be empty.")

    st = time.time()

    # 核心训练循环
    N = args.n_clients
    K = args.g1
    for r in range(args.tg):
        # E-step
        if r == 0:
            dws = np.zeros([N, K])
            client_params = server.train_clients(clients, sampling_rate=1.0, r=r)
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
            client_params = server.train_clients(clients, sampling_rate=1.0, r=r)
            for i in range(args.n_clients):
                client_state = client_params[i].to(args.device)
                for j in range(args.g1):
                    cls_state = tensor_dict_to_vector(server.cluster_models[j].state_dict()).to(args.device)
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

        if (r + 1) % 1 == 0:
            test_acc = fed_eval(server, args.device)
            res.loc[0, f'test_acc{r + 1}'] = test_acc
            print(f'[{r + 1}/{args.tg}] {server.name} | Acc: {100 * test_acc:.2f}%')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = finalize_and_save_experiment(res, clients, server, args, st)

    return result


def fedrc_exp(global_model, train_loader, test_loader, args):
    train_config = setup_experiment_config(args)
    train_config['freeze_1st'] = 0
    res = init_results_dataframe(args)

    clients = np.array([FedRCClient(id_num=k, train_d=train_loader[k], train_config=train_config, K=args.g1)
                        for k in range(args.n_clients)])
    server = FedRCServer(copy.deepcopy(global_model), test_loader, args.alg, args.n_clients, args.device, args.dataset, K=args.g1)

    st = time.time()

    # 核心训练循环
    sampling_rate = args.sampling_rate
    for r in range(args.tg):
        # 1. 客户端采样
        if sampling_rate < 1.0:
            sampled_clients = server.select_clients(sampling_rate)
        else:
            sampled_clients = np.array(list(range(args.n_clients)))
        server.sample_account[sampled_clients] += 1
        server.sampled_clients = sampled_clients

        # 2. 获取当前的 K 个模型实例
        # 注意这里获取的是模型的引用，实际深拷贝发生在客户端内部
        client_updates = []
        global_models = list(server.get_global_models().values())

        # 3. 客户端本地训练
        for cid in sampled_clients:
            client = clients[cid]
            # 客户端无需自带架构，直接传入全局模型实例列表
            client.train(model=global_models, r=r, vk=None)
            updated_states, pi_weights = client.model_params, client.pi_weights
            client_updates.append((updated_states, pi_weights))

            # 打印当前客户端最倾向的聚类
            best_cluster = np.argmax(pi_weights)
            print(f"Client {cid} 最倾向于 Cluster {best_cluster} (权重: {pi_weights[best_cluster]:.3f})")

        # 4. 服务端执行 FedRC 加权聚合
        server.aggregate(client_updates)

        if (r + 1) % 1 == 0:
            test_acc = fed_eval(server, args.device)
            res.loc[0, f'test_acc{r + 1}'] = test_acc
            print(f'[{r + 1}/{args.tg}] {server.name} | Acc: {100 * test_acc:.2f}%')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = finalize_and_save_experiment(res, clients, server, args, st)

    return result
