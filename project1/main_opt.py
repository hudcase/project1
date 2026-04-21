import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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


# ============================================================
# 2. 模型定义（多层感知机 MLP）
# ============================================================
class WineQualityModel(torch.nn.Module):
    """
    用于葡萄酒质量预测的回归模型
    网络结构：
        11 → 6 → 4 → 1
    """
    def __init__(self, input_dim=11):
        super(WineQualityModel, self).__init__()
        
        # 三个全连接层
        self.linear1 = torch.nn.Linear(input_dim, 20)
        self.linear2 = torch.nn.Linear(20, 10)
        self.linear3 = torch.nn.Linear(10, 1)
        
        # 激活函数（ReLU）
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        # self.activation = torch.nn.ReLU()
        # 测试集 MSE Loss: 0.5846
        # self.activation = torch.nn.Tanh()
        # 测试集 MSE Loss: 0.7286
        # self.activation = torch.nn.ReLU6()
        # 测试集 MSE Loss: 0.6522
        self.activation = torch.nn.PReLU()
        # 测试集 MSE Loss: 0.5660
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.dropout(x)

        x = self.activation(self.linear2(x))
        x = self.dropout(x)

        x = self.linear3(x)
        return x


    # def forward(self, x):
    #     """
    #     前向传播
    #     """
    #     x = self.activation(self.linear1(x))
    #     x = self.activation(self.linear2(x))
        
    #     # 回归任务最后一层不加激活函数
    #     x = self.linear3(x)
    #     return x


# ============================================================
# 3. 验证 / 测试评估函数
# ============================================================
def evaluate(model, data_loader, criterion, device):
    """
    在验证集或测试集上评估模型
    """
    # 切换到评估模式（影响 Dropout / BatchNorm）
    model.eval()
    
    total_loss = 0
    predictions = []
    actuals = []
    
    # 评估阶段不计算梯度，节省显存和计算
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 保存预测值和真实值（用于分析）
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    
    # 平均 loss
    avg_loss = total_loss / len(data_loader)
    return avg_loss, predictions, actuals


# ============================================================
# 4. 主流程
# ============================================================
def main():
    # ---------------- 超参数 ----------------
    epochs = 1000
    learning_rate = 0.0001
    batch_size = 128
    patience = 20
    patience_counter = 0
    # ---------------- 数据路径 ----------------
    train_file_path = 'wine_train.csv'
    val_file_path = 'wine_val.csv'
    test_file_path = 'wine_test.csv'
        
    # ---------------- 数据加载 ----------------
    train_loader, val_loader, test_loader = get_dataset_loader(
        train_file_path,
        val_file_path,
        test_file_path,
        batch_size
    )

    # ---------------- 设备选择 ----------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ---------------- 模型 / 损失 / 优化器 ----------------
    model = WineQualityModel(input_dim=11).to(device)
    
    # 回归任务常用 MSELoss
    criterion = torch.nn.MSELoss()
    
    # https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html
    # 使用 SGD 优化器
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # 测试集 MSE Loss: 0.5519
    # optimizer = torch.optim.SGD(model.parameters(), momentum = 0.9, lr=learning_rate)
    # # 测试集 MSE Loss: 0.5353
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    # 测试集 MSE Loss: 4.2505
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # # 测试集 MSE Loss: 0.5116
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # # 测试集 MSE Loss: 0.5078
    

    # ---------------- 训练准备 ----------------
    best_val_loss = float('inf')
    
    print("\n开始训练...")
    print("=" * 50)
    
    train_losses = []
    val_losses = []

    # ====================================================
    # 训练循环（不封装，保持原始结构）
    # ====================================================
    for epoch in range(epochs):
        
        # 切换为训练模式
        model.train()
        total_loss = 0
        
        # 遍历一个 epoch 中的所有 batch
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 参数更新
            optimizer.step()
            
            total_loss += loss.item()
        
        # 训练集平均 loss
        train_loss = total_loss / len(train_loader)
        
        # ---------------- 验证 ----------------
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 日志输出
        print(f"Epoch [{epoch+1:3d}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              )
        
        # ---------------- 早停 / 保存最优模型 ----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0   # 有提升，计数器清零
        else:
            patience_counter += 1  # 没提升，计数器 +1

        # 触发早停
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # # ---------------- 早停 / 保存最优模型 ----------------
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), 'best_model.pth')

    # ====================================================
    # 5. Loss 曲线绘制
    # ====================================================
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training & Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    
    # 保存图片到当前目录
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # ====================================================
    # 6. 测试集评估
    # ====================================================
    print("\n" + "=" * 50)
    print("在测试集上评估最佳模型...")
    
    # 加载验证集上最优的模型参数
    model.load_state_dict(torch.load('best_model.pth'))
    
    # 测试
    test_loss, predictions, actuals = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"测试集 MSE Loss: {test_loss:.4f}")
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    # 输出前 5 个样本的预测结果对比
    print("\n预测 vs 实际 (前5个样本):")
    for i in range(10):
        print(f"  预测: {predictions[i]:6.2f} | 实际: {actuals[i]:6.2f}")


# ============================================================
# 程序入口
# ============================================================
if __name__ == '__main__':
    main()
