import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1. 数据集类（自定义 PyTorch Dataset）
# ============================================================
class WineDataset(Dataset):
    """
    WineDataset 用于读取葡萄酒质量数据集，并将其封装成
    PyTorch 可直接使用的 Dataset 格式。
    
    每一条数据包含：
    - 特征：前 11 列（连续数值特征）
    - 标签：最后 1 列（葡萄酒质量评分，回归任务）
    """
    def __init__(self, filepath):
        """
        初始化数据集
        :param filepath: CSV 文件路径
        """
        # 用 Python 列表暂存特征和标签，读取完成后再统一转 tensor
        features = []
        labels = []
        
        # 使用 with open 方式读取文件，避免文件未正确关闭
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # 跳过第一行（表头）
            for line in lines[1:]:
                # 去除每一行首尾的空白符（换行符、空格等）
                line = line.strip()
                
                # 原始 wine 数据集一般用分号 ; 分隔
                # 为了兼容不同格式，这里做一次判断
                if ';' in line:
                    parts = line.split(';')
                else:
                    parts = line.split(',')
                
                # 将字符串列表转换为 float32 的 numpy 数组
                data = np.array(parts, dtype=np.float32)
                
                # 前 11 列作为输入特征
                feature = data[:11]
                
                # 最后一列作为标签（回归值）
                # 使用 data[[-1]] 保证 shape 是 (1,)
                label = data[[-1]]
                
                features.append(feature)
                labels.append(label)
        
        # 将 Python list → numpy array → torch Tensor
        # x_data: [N, 11]
        # y_data: [N, 1]
        self.x_data = torch.from_numpy(np.array(features, dtype=np.float32))
        self.y_data = torch.from_numpy(np.array(labels, dtype=np.float32))
        
        # 数据集大小
        self.len = len(self.x_data)
    
    def __len__(self):
        """
        返回数据集样本数量
        DataLoader 内部会调用这个函数
        """
        return self.len

    def __getitem__(self, index):
        """
        根据索引 index 返回一条样本
        :return: (feature, label)
        """
        return self.x_data[index], self.y_data[index]

# ============================================================
# 数据加载器构建函数
# ============================================================
def get_dataset_loader(train_file_path, val_file_path, test_file_path, batch_size):
    """
    构建训练 / 验证 / 测试 DataLoader
    """
    # 分别加载训练、验证、测试数据集
    train_dataset = WineDataset(train_file_path)
    val_dataset = WineDataset(val_file_path)
    test_dataset = WineDataset(test_file_path)
    
    # 数据集大小统计
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)
    
    print(f"训练集: {train_size} | 验证集: {val_size} | 测试集: {test_size}")
    
    # DataLoader：
    # - 训练集 shuffle=True（每个 epoch 打乱数据）
    # - 验证 / 测试集不打乱
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

batch_size = 32
# 数据路径设置
train_file_path = 'wine_train.csv'
val_file_path = 'wine_val.csv'
test_file_path = 'wine_test.csv'

# 得到数据集
train_loader, val_loader, test_loader = get_dataset_loader(train_file_path, val_file_path, test_file_path, batch_size)

for i, data in enumerate(train_loader):
    x_features, labels = data
    import pdb;pdb.set_trace()