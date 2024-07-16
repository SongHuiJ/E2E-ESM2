import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.metrics import matthews_corrcoef
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

class MyModel(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=2):
        super(MyModel, self).__init__()
        self.esm2 = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
    def forward(self, input_ids):
        pooled_output = self.esm2(input_ids).pooler_output
        print(self.esm2(input_ids))
        # print(pooled_output)
        # pooled_output = self.bert_model(input_ids)
        return pooled_output


class MyDataSet(Data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D", do_lower_case=False)
        # self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D", do_lower_case=False)
    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.label[idx]
        inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=60, truncation=True)
        print(inputs)
        input_ids = inputs.input_ids.squeeze(0)
        # token_type_ids=inputs.token_type_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
        # return input_ids,token_type_ids,attention_mask,label
        return input_ids, attention_mask, label
        # # 1. 构建字母到数字的映射
        # protein_alphabet = "ACDEFGHIKLMNPQRSTVWY"  # 蛋白质字母表
        # char_to_index = {char: index for index, char in enumerate(protein_alphabet)}
        # # 3. 将蛋白质序列编码为整数序列
        # encoded_sequence = [char_to_index[char] for char in text]
        # # 4. 创建PyTorch张量
        # inputs = torch.tensor(encoded_sequence, dtype=torch.int64)
        # # input_ids = inputs.input_ids.squeeze(0)
        # input_ids = inputs.squeeze(0)
        # # attention_mask = inputs.attention_mask.squeeze(0)
        # label = int(label)
        # label = torch.tensor(label)
        # return input_ids, label

    def __len__(self):
        return len(self.data)



df = pd.read_csv('./data/realdata.csv')
print('一共有{}条数据'.format(len(df)))
df.info()
use_df = df[:]
use_df.head(10)
features = df['feature'].tolist()
labels = df['label'].tolist()

df1 = pd.read_csv('./data/fakedata.csv')
print('一共有{}条数据'.format(len(df1)))
df1.info()
use_df = df1[:]
use_df.head(10)
features1 = df1['feature'].tolist()
labels1 = df1['label'].tolist()

real_dataset = MyDataSet(features, labels)
realloader = Data.DataLoader(real_dataset, batch_size=1, shuffle=False)
fake_dataset = MyDataSet(features1, labels1)
fakeloader = Data.DataLoader(fake_dataset, batch_size=1, shuffle=False)

model = MyModel()

for name, parameters in model.named_parameters():
    print(name, ';', parameters.size())

print(model)

for input_ids, _, _ in realloader:
    real_input = input_ids
    print(real_input.shape)
    real_pred = model(input_ids)
    print(real_pred)

for input_ids, _, _  in fakeloader:
    fake_input = input_ids
    fake_pred = model(input_ids)
    fake_pred = model(input_ids)

real_input = real_input.detach().numpy().reshape(-1)
fake_input = fake_input.detach().numpy().reshape(-1)
cosine_similarity_before = np.dot(real_input, fake_input) / (np.linalg.norm(real_input) * np.linalg.norm(fake_input))
print("扩充前余弦相似度:", cosine_similarity_before)


# 计算余弦相似度
real_pred = real_pred.detach().numpy().reshape(-1)
fake_pred = fake_pred.detach().numpy().reshape(-1)
cosine_similarity_after = np.dot(real_pred, fake_pred) / (np.linalg.norm(real_pred) * np.linalg.norm(fake_pred))
print("扩充后余弦相似度:", cosine_similarity_after.mean())

import numpy as np
def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)
cosine_similarity_before = euclidean_distance(real_input, fake_input)
print("扩充前欧几里得距离:", cosine_similarity_before)
cosine_similarity_after = euclidean_distance(real_pred, fake_pred)
print("扩充后欧几里得距离:", cosine_similarity_after)

def pearson_correlation(vector1, vector2):
    return np.corrcoef(vector1, vector2)[0, 1]
cosine_similarity_before = pearson_correlation(real_input, fake_input)
print("扩充前皮尔逊相关系数:", cosine_similarity_before)
cosine_similarity_after = pearson_correlation(real_pred, fake_pred)
print("扩充后皮尔逊相关系数:", cosine_similarity_after)

import numpy as np

def tanimoto_coefficient(set1, set2):
    intersection = len(np.intersect1d(set1, set2))
    union = len(np.union1d(set1, set2))
    return intersection / union

cosine_similarity_before = tanimoto_coefficient(real_input, fake_input)
print("扩充前tanimoto系数:", cosine_similarity_before)
cosine_similarity_after = tanimoto_coefficient(real_pred, fake_pred)
print("扩充后tanimoto系数:", cosine_similarity_after)


# 从文件或其他数据源获取蛋白质结构的坐标数据，然后将其存储为NumPy数组
# 假设protein1_coords和protein2_coords是两个包含坐标数据的NumPy数组

# 检查原子数量是否相同
if len(real_input) != len(fake_input):
    raise ValueError("原子数量不匹配")
# 计算每对原子之间的坐标差的平方
diff_squared = (real_input - fake_input) ** 2
# 计算坐标差的平方的和
sum_of_squared_diff = np.sum(diff_squared)
# 计算RMSD
rmsd_before = np.sqrt(sum_of_squared_diff / len(real_pred))
print("扩充前RMSD:", rmsd_before)

if len(real_pred) != len(fake_pred):
    raise ValueError("原子数量不匹配")
# 计算每对原子之间的坐标差的平方
diff_squared = (real_pred - fake_pred) ** 2
# 计算坐标差的平方的和
sum_of_squared_diff = np.sum(diff_squared)
# 计算RMSD
rmsd_after = np.sqrt(sum_of_squared_diff / len(real_pred))

print("扩充后RMSD:", rmsd_after)
