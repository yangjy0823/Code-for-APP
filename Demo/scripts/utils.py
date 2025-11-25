# scripts/utils.py

import os
import pickle
from glob import glob
from datetime import datetime
import torch

def find_latest_file(directory: str, pattern: str) -> str or None:
    """
    在指定目录中查找包含特定模式的、最新创建的文件。

    Args:
        directory (str): 要搜索的目录路径。
        pattern (str): 文件名中必须包含的模式/子字符串。

    Returns:
        str or None: 如果找到，返回最新文件的完整路径；否则返回None。
    """
    try:
        search_pattern = os.path.join(directory, f"*{pattern}*")
        files = glob(search_pattern)
        if not files:
            return None
        latest_file = max(files, key=os.path.getctime)
        return latest_file
    except Exception as e:
        print(f"Error finding latest file in '{directory}' with pattern '{pattern}': {e}")
        return None

def save_model(model, filepath: str):
    """
    智能保存模型，自动区分Sklearn和PyTorch模型。
    对于PyTorch模型，只保存state_dict。

    Args:
        model: 要保存的模型对象。
        filepath (str): 完整的文件保存路径 (包括文件名)。
    """
    try:
        # 确保目标目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 判断模型类型
        if isinstance(model, torch.nn.Module):
            # PyTorch模型，保存state_dict
            torch.save(model.state_dict(), filepath)
            print(f"PyTorch model state_dict saved to: {filepath}")
        else:
            # 默认为Sklearn或类似对象，使用pickle
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"Sklearn model saved to: {filepath}")
            
    except Exception as e:
        print(f"Error saving model to '{filepath}': {e}")


def load_model(filepath: str, model_instance: torch.nn.Module = None):
    """
    智能加载模型，自动区分Sklearn和PyTorch模型。
    加载PyTorch模型时，必须提供一个已初始化的模型实例。

    Args:
        filepath (str): 模型的完整路径。
        model_instance (torch.nn.Module, optional): 
            对于PyTorch模型，这是必需的。一个空的模型结构，用于加载权重。

    Returns:
        加载好的模型对象。
    
    Raises:
        ValueError: 如果尝试加载PyTorch模型但未提供model_instance。
        FileNotFoundError: 如果文件路径不存在。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
        
    try:
        # 根据文件扩展名或内容判断模型类型
        # 为简单起见，我们约定PyTorch模型以 .pth 结尾
        if filepath.endswith(".pth"):
            if model_instance is None:
                raise ValueError("Loading a PyTorch model requires providing a 'model_instance'.")
            # 加载state_dict到提供的模型实例中
            model_instance.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
            print(f"PyTorch model loaded from: {filepath}")
            return model_instance
        else:
            # 默认为pickle文件
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            print(f"Sklearn model loaded from: {filepath}")
            return model
            
    except Exception as e:
        print(f"Error loading model from '{filepath}': {e}")
        raise