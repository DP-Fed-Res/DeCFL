import os
import copy
import torch
import argparse

from warnings import simplefilter

from dataset import get_dataloader
from utils import set_random, create_keras_model
from exp import decfl_exp, fedavg_exp, fedprox_exp, scaffold_exp, fed_pcdp_exp, flexcfl_exp, fesem_exp, fedrc_exp


def parse_args(args_string=None):
    """封装参数解析器，支持传入字符串或使用系统参数"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str, default='fedavg', help='指定联邦学习算法')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID")
    parser.add_argument('--seed', type=int, default=123, help="随机数种子")

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='Mnist', help="数据集类型")
    parser.add_argument('--g1', type=int, default=2, help="组数量，每组p(y|x)不同")
    parser.add_argument('--g2', type=int, default=10, help="组数量，每组p(y)不同")
    parser.add_argument('--per_group', type=int, default=1, help="每组中客户端数量")
    parser.add_argument('--split', type=str, default='dirichlet', help="客户端数据分布")
    parser.add_argument('--same_py', type=int, default=1, help="不同概念分布组的标签分布是否一致")
    parser.add_argument('--alpha', type=float, default=1.0, help="dirichlet paras")
    parser.add_argument('--n_class', type=int, default=10, help="total classes")

    # 客户端参数
    parser.add_argument('--lr_mode', type=str, default='SGD', help="优化器类型")
    parser.add_argument('--freeze_1st', type=int, default=0, help="首轮是否冻结特征层")
    parser.add_argument('--tc_1st', type=int, default=150, help="首轮本地训练步数")
    parser.add_argument('--tc', type=int, default=50, help="本地训练步数")
    parser.add_argument('--br', type=float, default=0.02, help="批次比例")
    parser.add_argument('--lr', type=float, default=0.1, help="学习率")

    parser.add_argument('--eps', type=float, default=1.0, help="DP paras")
    parser.add_argument('--clip', type=float, default=1.0, help="DP paras")
    parser.add_argument('--delta', type=float, default=1e-5, help="DP paras")

    # 联邦参数
    parser.add_argument('--tg', type=int, default=20, help="FL rounds")
    parser.add_argument('--sampling_rate', type=float, default=1.0, help="客户端采样率")

    # decfl参数
    parser.add_argument('--k', type=int, default=400, help="子空间降维参数")
    parser.add_argument('--gc', type=int, default=1, help='GC机制开关: 1开启, 0关闭')
    parser.add_argument('--ha', type=int, default=1, help='HA机制开关: 1开启, 0关闭')

    if args_string:
        return parser.parse_args(args_string.split())
    else:
        return parser.parse_args()


def main(params):
    l_train_loader, g_test_loader = get_dataloader(params.dataset, params.g1, params.g2, params.per_group, params.alpha, params.same_py, params.br, params.device)

    global_model = create_keras_model(params.dataset).to(device=params.device)
    total_params = sum(p.numel() for p in global_model.parameters())
    params.params_num = total_params
    print('Total params: {}'.format(total_params))

    if args.alg in ['decfl', 'decfl-wo-ha', 'decfl-wo-gc']:
        res = decfl_exp(copy.deepcopy(global_model), l_train_loader, g_test_loader, params)
    elif args.alg == 'fedavg':
        res = fedavg_exp(copy.deepcopy(global_model), l_train_loader, g_test_loader, params)
    elif args.alg == 'fedprox':
        res = fedprox_exp(copy.deepcopy(global_model), l_train_loader, g_test_loader, params)
    elif args.alg == 'scaffold':
        res = scaffold_exp(copy.deepcopy(global_model), l_train_loader, g_test_loader, params)
    elif args.alg == 'flexcfl':
        res = flexcfl_exp(copy.deepcopy(global_model), l_train_loader, g_test_loader, params)
    elif args.alg == 'fesem':
        res = fesem_exp(copy.deepcopy(global_model), l_train_loader, g_test_loader, params)
    elif args.alg == 'fed_pcdp':
        res = fed_pcdp_exp(copy.deepcopy(global_model), l_train_loader, g_test_loader, params)
    elif args.alg == 'fedrc':
        res = fedrc_exp(copy.deepcopy(global_model), l_train_loader, g_test_loader, params)
    else:
        raise Exception

    print('finish')
    return res


if __name__ == '__main__':
    simplefilter(action="ignore", category=FutureWarning)
    simplefilter(action="ignore", category=UserWarning)

    IS_DEBUG = True  # 设置为 True: 在IDE直接跑预设参数 | 设置为 False: 真实命令行模式

    if IS_DEBUG:
        # 在这里写想要 Debug 的参数
        debug_command = ("--alg decfl --dataset Cifar10 --g1 4 --g2 5 --alpha 1.0 "
                         "--tc_1st 150 --tc 50 --br 0.02 --lr 0.1 "
                         "--tg 20 --eps 5 --clip 15.0 --gpu 0 --freeze_1st 1")
        args = parse_args(debug_command)
    else:
        # 真实命令行模式
        args = parse_args()

    args.n_clients = args.g1 * args.g2 * args.per_group
    args.prior_cls = [i // int(args.n_clients / args.g1) for i in range(args.n_clients)]

    # 设置 GPU 环境
    gpu = getattr(args, 'gpu', 0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    args.device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    print("最终运行参数:", args)
    set_random(args.seed)
    main(args)