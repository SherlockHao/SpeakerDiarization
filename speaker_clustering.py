"""
speaker_clustering - 说话人聚类库

此库提供了一个用于对音频嵌入向量进行聚类的接口，使用AgglomerativeClustering算法。
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

class SpeakerClustering:
    """
    说话人聚类类
    使用AgglomerativeClustering算法对音频嵌入向量进行聚类
    """
    
    def __init__(self, n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.71):
        """
        初始化聚类器
        
        Args:
            n_clusters (int, optional): 聚类数量，如果为None则使用distance_threshold
            metric (str): 距离度量方法
            linkage (str): 链接准则
            distance_threshold (float): 距离阈值，用于确定聚类数量
        """
        self.cluster_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=metric, 
            linkage=linkage,
            distance_threshold=distance_threshold
        )
        
    def cluster_embeddings(self, embeddings):
        """
        对嵌入向量进行聚类
        
        Args:
            embeddings (list or numpy.ndarray): 嵌入向量列表或数组，形状为(n_samples, n_features)
            
        Returns:
            numpy.ndarray: 聚类标签数组，形状为(n_samples,)
        """
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        
        if len(embeddings.shape) == 1:
            # 如果只有一个嵌入向量，需要扩展为2D数组
            embeddings = embeddings.reshape(1, -1)
        
        cluster_labels = self.cluster_model.fit_predict(embeddings)
        return cluster_labels

def initialize_clustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.71):
    """
    初始化聚类器的便捷函数
    
    Args:
        n_clusters (int, optional): 聚类数量，如果为None则使用distance_threshold
        metric (str): 距离度量方法
        linkage (str): 链接准则
        distance_threshold (float): 距离阈值，用于确定聚类数量
        
    Returns:
        SpeakerClustering: 初始化的聚类器实例
    """
    return SpeakerClustering(n_clusters, metric, linkage, distance_threshold)