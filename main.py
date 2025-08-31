import copy
import os
from options import args_parser
from dataset import split_dataset
from train import cmp_experiment, abl_experiment, noise_experiment, sampling_experiment
import shutil
from warnings import simplefilter
import time
from utils import set_random, create_keras_model
import pandas as pd
import torch
import multiprocessing as mp


def get_train_config(data):
    config = {
        'Mnist': {
            'lr_mode': 'SGD',
            'local_steps': 20,
            'batch_ratio': 0.02,
            'lr': 0.1,  # 学习率
            'lr_decay': 0.998,  # 学习率衰减系数(per step)
            'noise': 1.0,  # DPSGD 噪声尺度
            'noise_decay': 1.0,  # DPSGD 噪声衰减系数(per step)
            'clip': 1.0,  # DPSGD 裁剪阈值
        },
        'FashionMnist': {
            'lr_mode': 'SGD',
            'local_steps': 20,
            'batch_ratio':0.02,
            'lr': 0.1,  # 学习率
            'lr_decay': 0.998,  # 学习率衰减系数(per step)
            'noise': 1.0,  # DPSGD 噪声尺度
            'noise_decay': 1.0,  # DPSGD 噪声衰减系数(per step)
            'clip': 1.0,  # DPSGD 裁剪阈值
        },
        'Cifar10': {
            'lr_mode': 'SGD',
            'local_steps': 20,
            'batch_ratio':0.02, # 设为-1时，batch_size为64
            'lr': 0.1,  # 学习率
            'lr_decay': 0.998,  # 学习率衰减系数(per step)
            'noise': 1.0,  # DPSGD 噪声尺度
            'noise_decay': 1.0,  # DPSGD 噪声衰减系数(per step)
            'clip': 1.0,  # DPSGD 裁剪阈值
        }
    }
    return config.get(data, None)


def get_fl_config(dataset, Scenario):
    """
        先划分p(y|x)异构，再划分p(y)异构
        groups1=1，groups2=1，无p(y|x)异构，无p(y)异构；
        groups1=2，groups2=1，有两种p(y|x)异构，无p(y)异构；
        groups1=1，groups2=2，无p(y|x)异构，有两种p(y)异构；
        groups1=2，groups2=2，有两种p(y|x)异构，p(y|x)一致时仍有两种p(y)异构；
    """

    args = args_parser()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.dataset = dataset
    # FL参数
    if dataset == 'Cifar10':
        args.rounds = 10  # FL轮数
        args.k = 400  # 梯度降维,-1则不降维(受显存限制，可酌情减小)
    else:
        args.rounds = 10
        args.k = 400  # 梯度降维,-1则不降维
    # 客户端数据异构参数
    args.Scenario = Scenario
    if args.Scenario == 1:  # py|x + py异构
        args.groups1 = 5  # 组数量，每组p(y|x)不同，假设已知
        args.groups2 = 5  # 组数量，每组p(y)不同
        args.clients_per_group = 1  # p(y|x)和p(y)均一致的客户端数量
        args.K = args.groups1
        args.n_clients = args.groups1 * args.groups2 * args.clients_per_group  # 总客户端数=groups1*groups2*clients_per_group
        args.alpha = 1.0  # p(y) dirichlet paras
        args.overlap_ratio = False  # p(y|x) dirichlet paras
        args.same_py = True  # 不同p(y|x)组内的p(y)异构是否保持一致
        args.prior_cls = [i // int(args.n_clients/args.K) for i in range(args.n_clients)]
    elif args.Scenario == 2:  # py|x异构
        args.groups1 = 5  # 组数量，每组p(y|x)不同，假设已知
        args.groups2 = 1  # 组数量，每组p(y)不同
        args.clients_per_group = 5  # p(y|x)和p(y)均一致的客户端数量
        args.K = args.groups1
        args.n_clients = args.groups1 * args.groups2 * args.clients_per_group  # 总客户端数=groups1*groups2*clients_per_group
        args.alpha = 1.0  # p(y) dirichlet paras
        args.overlap_ratio = False  # p(y|x) dirichlet paras
        args.same_py = True  # 不同p(y|x)组内的p(y)异构是否保持一致
        args.prior_cls = [i // int(args.n_clients/args.K) for i in range(args.n_clients)]
    else:  # py异构
        args.groups1 = 1  # 组数量，每组p(y|x)不同，假设已知
        args.groups2 = 5  # 组数量，每组p(y)不同
        args.clients_per_group = 5  # p(y|x)和p(y)均一致的客户端数量
        args.K = args.groups1
        args.n_clients = args.groups1 * args.groups2 * args.clients_per_group  # 总客户端数=groups1*groups2*clients_per_group
        args.alpha =1.0  # p(y) dirichlet paras
        args.overlap_ratio = False  # p(y|x) dirichlet paras
        args.same_py = True  # 不同p(y|x)组内的p(y)异构是否保持一致
        args.prior_cls = [i // int(args.n_clients/args.K) for i in range(args.n_clients)]


    print(args.prior_cls)
    args.primary_aggregation = True

    # 客户端训练相关
    args.train_config = get_train_config(args.dataset)
    # 保存路径
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

    return args


def main():
    # 构造数据
    train, val, test, label_maps, report, (all_client_data, all_client_labels, transform) = split_dataset(dataset_name=args.dataset,
                                                         num_concept_groups=args.groups1,
                                                         num_label_groups=args.groups2,
                                                         num_clients_per_group=args.clients_per_group,
                                                         alpha=args.alpha,
                                                         overlap_ratio=args.overlap_ratio,
                                                         same_py=args.same_py,
                                                         batch_ratio=args.train_config.get('batch_ratio'))
    # 将 per-client 原始数组打包传给子进程，避免直接 pickle DataLoader/FedRCClient
    client_arrays_pack = {
        'client_arrays': [(all_client_data[i], all_client_labels[i]) for i in range(len(all_client_data))],
    }
    print(label_maps)
    print('dataset: ', args.dataset)

    # 构造全局模型
    global_model = create_keras_model(args.dataset).to(device=args.device)
    total_params = sum(p.numel() for p in global_model.parameters())
    args.params_num = total_params
    print('Total params: {}'.format(total_params))

    args.sampling_rate = 1.0
    for noise in [0, 1.0, 1.5, 2.0]:  # 0, 1.0, 1.5, 2.0
        args.train_config['noise'] = noise
        # 消融实验
        abl_experiment(train, val, test, copy.deepcopy(global_model), args)
        # 对比实验
        # cmp_experiment(train, val, test, copy.deepcopy(global_model), args, client_arrays_pack)

    print('finish')


if __name__ == '__main__':
    os.environ["WANDB_API_KEY"] = 'e5dc3f1f4d367ec3a412d359ae4cfc222deacfa2'
    # os.environ["WANDB_MODE"] = "offline"
    simplefilter(action="ignore", category=FutureWarning)
    simplefilter(action="ignore", category=UserWarning)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    Scenario = 1 # 1混合异构；2单一异构
    for dataset in ['Cifar10', 'FashionMnist', 'Mnist']:  # 'Cifar10', 'FashionMnist', 'Mnist'
        args = get_fl_config(dataset, Scenario)
        set_random(args.seed)
        main()
