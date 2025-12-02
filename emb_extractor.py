"""
emb_extractor - 语音嵌入提取库

此库提供了一个用于提取音频嵌入向量的接口，使用pyannote/wespeaker-voxceleb-resnet34-LM模型。
"""
from huggingface_hub import login, snapshot_download
from pyannote.audio import Inference, Model
import torch
import numpy as np
import os

class EmbeddingExtractor:
    """
    音频嵌入提取器类
    使用pyannote/wespeaker-voxceleb-resnet34-LM模型提取音频嵌入向量
    """
    
    def __init__(self, api_key=None, offline_mode=False):
        """
        初始化嵌入提取器
        
        Args:
            api_key (str, optional): Hugging Face API密钥，如果未提供则尝试从环境变量获取
            offline_mode (bool, optional): 是否使用离线模式（从本地加载模型），默认为False
            
        Raises:
            ValueError: 当API密钥未提供且不在离线模式时抛出
        """
        self.offline_mode = offline_mode
        
        # 如果不是离线模式，需要API密钥
        if not offline_mode:
            if api_key is None:
                api_key = os.getenv("HF_API_KEY")
                if api_key is None:
                    raise ValueError("API密钥未提供，且环境变量HF_API_KEY未设置")
            
            # 使用APIKEY进行Hugging Face认证
            login(token=api_key)

        # 由于PyTorch版本兼容性问题，需要临时允许安全全局
        import torch.serialization

        # 定义需要允许的类 - 包含更多可能的类
        try:
            from pyannote.audio.core.task import Specifications, Problem, Resolution
            torch.serialization.add_safe_globals([
                torch.torch_version.TorchVersion, 
                Specifications, 
                Problem,
                Resolution
            ])
        except ImportError:
            pass

        # 临时修改torch.load以处理weights_only问题
        original_torch_load = torch.load
        def patched_torch_load(f, map_location=None, **kwargs):
            kwargs.setdefault('weights_only', False)
            return original_torch_load(f, map_location=map_location, **kwargs)
        torch.load = patched_torch_load

        self.original_torch_load = original_torch_load

        device = torch.device("cpu")
        
        if offline_mode:
            # 离线模式：从本地加载模型
            local_model_path = self._get_local_model_path()
            self.embedding_model = Model.from_pretrained(
                local_model_path,
                use_auth_token=api_key if api_key else None,
                cache_dir=None,  # 不使用缓存，直接使用本地路径
                local_files_only=True  # 只使用本地文件
            )
        else:
            # 在线模式：从Hugging Face Hub加载模型
            self.embedding_model = Model.from_pretrained(
                "pyannote/wespeaker-voxceleb-resnet34-LM",
                use_auth_token=api_key
            )
        
        self.embedding_model.to(device)

        # 创建推理实例
        self.embedding_inference = Inference(self.embedding_model, window="whole")
        self.embedding_inference.model.to(device)

    def _get_local_model_path(self):
        """
        获取本地模型路径
        
        Returns:
            str: 本地模型路径
        """
        # 检查默认的Hugging Face缓存路径
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_dir = os.path.join(cache_dir, "models--pyannote--wespeaker-voxceleb-resnet34-LM")
        
        if os.path.exists(model_dir):
            # 找到最新的快照目录
            snapshots_dir = os.path.join(model_dir, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshot_dirs = [d for d in os.listdir(snapshots_dir) if 
                                os.path.isdir(os.path.join(snapshots_dir, d))]
                if snapshot_dirs:
                    # 通常使用main引用的目录
                    ref_file = os.path.join(model_dir, "refs", "main")
                    if os.path.exists(ref_file):
                        with open(ref_file, 'r') as f:
                            commit_hash = f.read().strip()
                        snapshot_path = os.path.join(snapshots_dir, commit_hash)
                        if os.path.exists(snapshot_path):
                            return snapshot_path
                    
                    # 如果main引用不可用，使用最新的目录
                    latest_snapshot = sorted(snapshot_dirs, 
                                           key=lambda x: os.path.getmtime(os.path.join(snapshots_dir, x)),
                                           reverse=True)[0]
                    return os.path.join(snapshots_dir, latest_snapshot)
        
        raise FileNotFoundError(f"未找到本地模型文件，请确保模型已下载到: {model_dir}")

    def extract_embedding(self, wav_path):
        """
        从音频文件路径提取嵌入向量
        
        Args:
            wav_path (str): 音频文件路径
            
        Returns:
            numpy.ndarray: 音频的嵌入向量，形状为(256,)
            
        Raises:
            FileNotFoundError: 当音频文件不存在时抛出
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"音频文件不存在: {wav_path}")
        
        embedding = self.embedding_inference(wav_path)
        
        # 确保输出是一维数组
        if len(embedding.shape) > 1:
            embedding = embedding.reshape(-1)
            
        return embedding

def initialize_extractor(api_key=None, offline_mode=False):
    """
    初始化嵌入提取器的便捷函数
    
    Args:
        api_key (str, optional): Hugging Face API密钥
        offline_mode (bool, optional): 是否使用离线模式（从本地加载模型），默认为False
        
    Returns:
        EmbeddingExtractor: 初始化的嵌入提取器实例
    """
    return EmbeddingExtractor(api_key, offline_mode)

