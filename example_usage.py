"""
emb_extractor库使用示例
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import emb_extractor
from scipy.spatial.distance import cdist

def main():
    print("=== 在线模式示例 ===")
    # 1. 在线模式初始化提取器（需要API密钥）
    api_key = os.getenv("HF_KEY")   # 在实际使用中，请从环境变量或其他安全位置获取
    try:
        extractor = emb_extractor.initialize_extractor(api_key, offline_mode=False)
        
        print("在线提取器初始化成功！")
        
        # 2. 提取音频嵌入向量
        audio_path1 = './data/emb_yiya.wav'
        audio_path2 = './data/out_yiya_en.wav'
        
        embedding1 = extractor.extract_embedding(audio_path1)
        embedding2 = extractor.extract_embedding(audio_path2)
        
        print(f"音频1嵌入向量形状: {embedding1.shape}")
        print(f"音频2嵌入向量形状: {embedding2.shape}")
        
        # 3. 计算两个音频的相似度
        embedding1_2d = embedding1.reshape(1, -1)
        embedding2_2d = embedding2.reshape(1, -1)
        distance = cdist(embedding1_2d, embedding2_2d, metric='cosine')[0,0]
        
        print(f"两个音频之间的余弦距离: {distance}")
        print("在线模式处理完成！")
    except Exception as e:
        print(f"在线模式失败: {e}")

    print("=== 离线模式示例 ===")
    # 4. 离线模式初始化提取器（不需要网络连接）
    try:
        offline_extractor = emb_extractor.initialize_extractor(offline_mode=True)
        
        print("离线提取器初始化成功！")
        
        # 5. 提取音频嵌入向量
        embedding1_offline = offline_extractor.extract_embedding('./data/emb_yiya.wav')
        embedding2_offline = offline_extractor.extract_embedding('./data/out_yiya_en.wav')
        
        print(f"离线模式 - 音频1嵌入向量形状: {embedding1_offline.shape}")
        print(f"离线模式 - 音频2嵌入向量形状: {embedding2_offline.shape}")
        
        # 6. 计算两个音频的相似度
        embedding1_2d_off = embedding1_offline.reshape(1, -1)
        embedding2_2d_off = embedding2_offline.reshape(1, -1)
        distance_off = cdist(embedding1_2d_off, embedding2_2d_off, metric='cosine')[0,0]
        
        print(f"离线模式 - 两个音频之间的余弦距离: {distance_off}")
        print("离线模式处理完成！")
    except Exception as e:
        print(f"离线模式失败: {e}")

if __name__ == "__main__":
    main()
