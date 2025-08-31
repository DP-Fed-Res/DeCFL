from scipy.spatial.distance import jensenshannon
import numpy as np
import torch
from sklearn.metrics import silhouette_score
import skfuzzy as fuzz
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import copy
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment


def adaptive_kmeans(distance_matrix, max_clusters=10):
    """
    自适应搜索最优类簇数的K-Means算法。

    参数:
    - distance_matrix: 距离矩阵（numpy数组）
    - max_clusters: 最大尝试的类簇数（默认为10）

    返回:
    - best_k: 最优类簇数
    - best_labels: 最优聚类标签
    - best_score: 最优轮廓系数
    """
    # 转换为相似性矩阵（K-Means需要输入特征矩阵，这里假设距离矩阵可以转换为特征）
    # 如果距离矩阵是平方形式，可以尝试多维缩放（MDS）或其他降维方法
    # 这里简单使用距离矩阵作为输入（假设已经是合适的特征）
    X = distance_matrix

    best_k = 2
    best_labels = None
    best_score = -1

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)

        # 计算轮廓系数
        score = silhouette_score(X, labels)

        # 更新最优结果
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    return best_labels, best_k, best_score


def cosine_distance(vec1, vec2):
    """计算两个向量之间的余弦距离"""
    if torch.is_tensor(vec1):
        vec1 = vec1.cpu().detach().numpy()
    if torch.is_tensor(vec2):
        vec2 = vec2.cpu().detach().numpy()
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return 1 - similarity


def inference_dis(vec1, vec2):
    similarity = []
    for i in range(vec1.shape[0]):
        if vec1.loc[i, 'pred'] == vec2.loc[i, 'pred']:
            sign = 1
        else:
            sign = -1.2
        weight = min(vec1.loc[i, 'weight'], vec2.loc[i, 'weight'])
        if weight > 0.1:
            similarity.append(sign * weight)
    if len(similarity) > 0:
        return max(0, (-np.mean(similarity)+0.2)/0.4)
    else:
        return 1.0


def compute_pairwise_dis(vec, metric='cosine'):
    """
    :param vec : n_sample * features
    :return: distances : n_sample * n_sample
    """
    n = len(vec)
    distances = np.zeros([n, n])
    for i in range(n):
        for j in range(i + 1, n):
            if metric == 'cosine':
                dis = cosine_distance(vec[i], vec[j])
            elif metric == 'euclidean':
                dis = np.linalg.norm(vec[i] - vec[j])
            elif metric == 'inference':
                dis = inference_dis(copy.deepcopy(vec[i]), copy.deepcopy(vec[j]))
            elif metric == 'fcw':
                similarity_matrix = cosine_similarity(vec[i], vec[j])
                row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
                best_similarity = similarity_matrix[row_ind, col_ind].mean()
                dis = 1 - best_similarity
            elif metric == 'js':
                vec[i] = vec[i] / np.sum(vec[i])
                vec[j] = vec[j] / np.sum(vec[j])
                dis = jensenshannon(vec[i], vec[j])
            else:
                raise NotImplementedError
            distances[i][j] = dis
            distances[j][i] = dis
    return distances


def get_cluster_num(dis):
    n = len(dis)
    l = dis[np.triu_indices(n, k=1)]
    T1 = np.mean(l) + 0.5 * np.std(l)
    T2 = np.mean(l)
    canopies = canopy_clustering_distance_matrix(dis, T1, T2)
    estimated_k = len(canopies)
    return estimated_k, canopies


def try_cluster(dis):
    """
    Find the optimal number of clusters using hierarchical clustering and silhouette score.

    Args:
        dis (np.ndarray): Precomputed distance matrix of shape (n, n).

    Returns:
        Best silhouette score and corresponding number of clusters (k).
        If no valid k is found, returns 1, 1.
    """
    n = dis.shape[0]
    best_score = -1
    best_k = 1

    for k in range(2, n):
        agg_cluster = AgglomerativeClustering(
            n_clusters=k,
            metric='precomputed',
            linkage='average'
        )
        labels = agg_cluster.fit_predict(dis)
        sil_score = silhouette_score(dis, labels, metric='precomputed')

        if sil_score > best_score:
            best_score = sil_score
            best_k = k

    # Return default 1, 1 only if no valid clustering was found (unlikely in practice)
    return best_k if best_score > 0 else 1


def canopy_clustering_distance_matrix(distance_matrix, t1, t2):
    '''
    基于距离矩阵的Canopy聚类算法实现。
    参数：
    - distance_matrix: 预先计算的距离矩阵，形状为 (n_samples, n_samples)
    - t1: 阈值 T1，较大
    - t2: 阈值 T2，较小，要求 t1 > t2
    返回：
    - canopies: 一个列表，每个元素是一个Canopy的点索引列表
    '''
    if t2 >= t1:
        raise ValueError("T2 必须小于 T1")
    n_samples = distance_matrix.shape[0]
    # 创建一个集合，包含所有未处理的数据点索引
    remaining_points = set(range(n_samples))
    canopies = []
    while remaining_points:
        # 随机选择一个点作为当前Canopy的中心
        center_idx = remaining_points.pop()
        # 当前Canopy包含的点索引列表，包括中心点
        current_canopy = [center_idx]
        # 临时集合，用于在遍历过程中移除紧密集内的点
        delete_set = set()
        # 获取中心点与其他点的距离
        distances = distance_matrix[center_idx]
        for idx in remaining_points:
            distance = distances[idx]
            if distance < t1:
                # 如果距离小于 T1，将点加入当前Canopy
                current_canopy.append(idx)
            if distance < t2:
                # 如果距离小于 T2，将点标记为需要移除（紧密集）
                delete_set.add(idx)
        # 从剩余点中移除紧密集内的点
        remaining_points -= delete_set
        # 将当前Canopy加入结果列表
        canopies.append(current_canopy)
    return canopies


def fuzz_cluster(dis, estimated_k):
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    data = mds.fit_transform(dis)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data.T, c=estimated_k, m=2, error=0.005, maxiter=1000)
    best_u = u.T
    k = 0
    theta = 0.45
    while k != len(dis):
        clusters = []
        for i in range(best_u.shape[1]):
            cluster_ids = np.where(best_u[:, i] > theta)[0]
            clusters.append(cluster_ids)
        flattened_array = [element for row in clusters for element in row]
        k = len(set(flattened_array))
        theta -= 0.05
    theta += 0.05
    flattened_array = [element for row in clusters for element in row]
    element_counts = Counter(flattened_array)
    duplicate_elements = [element for element, count in element_counts.items() if count > 1]
    # print("重叠客户端：", duplicate_elements)
    return clusters, best_u