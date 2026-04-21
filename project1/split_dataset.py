import random

# 定义文件路径
input_file = 'winequality-white.csv'
train_file = 'wine_train.csv'
val_file = 'wine_val.csv'
test_file = 'wine_test.csv'

# 设置随机种子，保证结果可复现
random.seed(42)

# 1. 读取原始文件，分离表头和数据行
header = ''  # 存储表头
data_lines = []  # 存储所有数据行

with open(input_file, 'r', encoding='utf-8') as f:
    # 读取第一行作为表头
    header = f.readline().strip()
    # 读取剩余所有数据行（跳过空行）
    for line in f:
        line = line.strip()
        if line:  # 避免空行干扰
            data_lines.append(line)

# 2. 按随机数划分数据行
train_lines = []
val_lines = []
test_lines = []

total_lines = len(data_lines)
for line in data_lines:
    # 生成0-1之间的随机数
    rand_num = random.random()
    # 按区间划分
    if rand_num <= 0.7:
        train_lines.append(line)
    elif rand_num <= 0.85:  # 0.7 < rand_num <= 0.85
        val_lines.append(line)
    else:  # rand_num > 0.85
        test_lines.append(line)

# 3. 定义写入文件的函数（避免重复代码）
def write_csv(file_name, header, data_lines):
    with open(file_name, 'w', encoding='utf-8') as f:
        # 先写入表头
        f.write(header + '\n')
        # 再写入数据行（每行一个）
        for line in data_lines:
            f.write(line + '\n')

# 4. 写入三个划分后的文件
write_csv(train_file, header, train_lines)
write_csv(val_file, header, val_lines)
write_csv(test_file, header, test_lines)

# 5. 打印划分结果，验证比例
print(f"原始数据总行数（不含表头）: {total_lines}")
print(f"训练集行数: {len(train_lines)} ({len(train_lines)/total_lines*100:.1f}%)")
print(f"验证集行数: {len(val_lines)} ({len(val_lines)/total_lines*100:.1f}%)")
print(f"测试集行数: {len(test_lines)} ({len(test_lines)/total_lines*100:.1f}%)")

# 验证文件是否包含表头（可选）
print("\n验证表头是否存在：")
for fname in [train_file, val_file, test_file]:
    with open(fname, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        print(f"{fname} 表头: {first_line[:50]}...")  # 只显示前50个字符，避免过长