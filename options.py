import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=111, help="随机数种子")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--dataset', type=str, default='Cifar10', help="数据集类型")
    parser.add_argument('--n_class', type=int, default=10, help="total classes")
    parser.add_argument('--params_num', type=float, default=0, help="模型参数量")
    parser.add_argument('--mode', type=str, default='decfl', help="联邦学习方法名称")

    # 数据集构建相关
    parser.add_argument('--groups1', type=int, default=1, help="组数量，每组p(y|x)不同")
    parser.add_argument('--groups2', type=int, default=1, help="组数量，每组p(y)不同")
    parser.add_argument('--clients_per_group', type=int, default=1, help="每组中客户端数量")
    parser.add_argument('--alpha', type=float, default=1.0, help="dirichlet paras")
    parser.add_argument('--overlap_ratio', type=float, default=0, help="py|x paras")
    parser.add_argument('--ratio', type=float, default=0.5, help="组1数据/全量数据")
    parser.add_argument('--n_clients', type=int, default=100, help="number of users: n")

    parser.add_argument('--n_mc', type=int, default=5, help="shard paras")
    parser.add_argument('--balance', type=int, default=None, help="客户端数据量是否一,None或True")
    parser.add_argument('--split', type=str, default='dirichlet', help="客户端数据分布")
    parser.add_argument('--abnormal', type=int, default=0, help="异常数量，p(y|x)整体一致时，存在个别异常客户端p(y|x)不同")

    # 联邦训练相关
    parser.add_argument('--rounds', type=int, default=10, help="rounds of training")
    parser.add_argument('--local_step', type=int, default=100, help="the number of local steps: E")
    parser.add_argument('--lr_mode', type=str, default='SGD', help="learning optimal")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight_decay")
    parser.add_argument('--sampling_rate', type=float, default=1.0, help="客户端采样率")
    parser.add_argument('--ckpt_dir', type=str, default=None, help="重要信息存储路径")

    # 差分隐私相关
    parser.add_argument('--delta', type=float, default=1e-5, help="DP paras")
    parser.add_argument('--k', type=int, default=200, help="子空间降维参数")
    # parser.add_argument('--k_step', type=int, default=50, help="子空间降维参数")

    args = parser.parse_args()
    return args
