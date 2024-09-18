import argparse
import os
import sys
import torch
import torchvision.models as models
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from dataloader import CustomDataset

# 1. 定义命令行解析器对象
parser = argparse.ArgumentParser(description='Input params')

# 2. 添加命令行参数
parser.add_argument('--model_path', type=str, default="")
parser.add_argument('--test_path', type=str, default="/tmp/dataset")

# 3. 从命令行中结构化解析参数
args = parser.parse_args()
model_path = args.model_path
test_path = args.test_path
print('model path is {}, and test_path is {}'.format(model_path, test_path))

print("Start dataset loading!")
key_path = os.path.join(test_path, "key")
query_path = os.path.join(test_path, "query")
keyfolder = os.listdir(key_path)
queryfolder = os.listdir(query_path)
if len(keyfolder) == 0 or len(queryfolder) == 0:
    print("Input data is not correct!")
    sys.exit(1)
key_label_origin = []
key_data_origin = []
query_label_origin = []
query_data_origin = []
tmp = keyfolder + queryfolder

label_names = list(set(tmp))
label_dict = dict(zip(label_names, range(len(label_names))))
for root, dirs, files in os.walk(test_path):
    for file in files:
        # 构建每个文件的完整路径
        full_file_path = os.path.join(root, file)
        part = full_file_path.split("/")
        if part[-3] == "key":
            key_label_origin.append(label_dict[part[-2]])
            key_data_origin.append(full_file_path)
        elif part[-3] == "query":
            query_label_origin.append(label_dict[part[-2]])
            query_data_origin.append(full_file_path)

# 加载测试数据集

key_dataset = CustomDataset(key_data_origin, key_label_origin)
query_dataset = CustomDataset(query_data_origin, query_label_origin)
# 创建数据加载器
key_loader = DataLoader(key_dataset, batch_size=8, shuffle=False)
query_loader = DataLoader(query_dataset, batch_size=1, shuffle=False)

key_features = []

# 假设我们的模型在同一个目录下,这里随便找了一个resnet18的模型，到时候替换为你们自己的模型
resnet = models.resnet18(pretrained=True)
reid_model = torch.nn.Sequential(*(list(resnet.children())[:-1]))

key_labels = []
# 生成字典特征
for images, labels in key_loader:
    # images = images.cuda()
    # 计算图像的特征
    with torch.no_grad():
        features = reid_model(images)

    # 保存特征
    key_features.append(features)
    key_labels.append(labels)
# 将所有的特征拼接成一个大的张量
key_features = torch.cat(key_features, dim=0)
key_labels=torch.stack(key_labels)
key_labels = key_labels.view(-1)
#
# 初始化评估指标
rank_n_correct = 0
total_queries = 0
ap_scores = []

# 对测试集中的每一个查询样本进行推理
rank_n = 1
query_feature = []
query_label_origin = []
for query_features, query_labels in query_loader:
    # query_features = query_features.cuda()
    # query_labels = query_labels.cuda()

    # 计算查询样本的特征
    with torch.no_grad():
        query_output = reid_model(query_features)
    query_feature.append(query_output)
    # 计算查询样本和库中所有样本的距离

    query_output = torch.squeeze(query_output).unsqueeze(dim=0)
    key_features = torch.squeeze(key_features)

    dists = torch.cdist(query_output, key_features)
    # 对距离进行排序并获取索引
    indices = dists.argsort()

    # 计算Rank-N精度
    if query_labels[0] in key_labels[indices[0][:rank_n]]:
        rank_n_correct += 1
    total_queries += 1
    # 计算mAP
    relevance = (key_labels[indices[0]] == query_labels[0]).numpy()
    ap = average_precision_score(relevance, -dists[0][indices[0]].numpy())
    ap_scores.append(ap)

# 计算最终的Rank-N精度和mAP
rank_n_accuracy = rank_n_correct / total_queries
mean_ap = sum(ap_scores) / len(ap_scores)

print(f'Rank-{rank_n} Accuracy: {rank_n_accuracy:.2f}')
print(f'mAP: {mean_ap:.2f}')