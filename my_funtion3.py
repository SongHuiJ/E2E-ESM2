import argparse
from implementations.translator import tensor2str, str2tensor
from models import *
from implementations.data_utils import *
import torch
from torch.autograd import Variable
import numpy as np
import os

torch.autograd.set_detect_anomaly(True)
# CUDA_LAUNCH_BLOCKING = 1
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# use_cuda = True if torch.cuda.is_available() else False  # 选择使用CPU还是GPU
use_cuda = True if torch.cuda.is_available() else False  # 选择使用CPU还是GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 选择使用GPU还是CPU训练模型

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0001,
                    help="learning rate")
parser.add_argument("--generator_model", type=str,
                    default="Gen_Lin_Block", help="choose generator model")
parser.add_argument("--sample_itr", type=int, default=7,
                    help="sample_itr*batch will be number of generates sequences")
parser.add_argument("--ut", type=str, default="P-PS",
                    help="Choose data update type: P-PS, PR-PS, R-S (start-end, R:Random, P:Positive, S:Synthetic)")
parser.add_argument("--classification", type=str, default="binary",
                    help="binary or multi for discriminator classification task")
parser.add_argument("--motif", action='store_true',
                    help="choose whether or not you want to include motif restriction. Default:False, place --motif if you want it to be True.")
parser.add_argument("--rev", type=str, default=None,
                    help="Choose revd data type: red, shuf, rep, revr (red-shuf-rep-revr or red-shuf-rep or red or shuf-rep or None)")
parser.add_argument("--data_size", type=int, default=2000, help="max data size")
opt = parser.parse_args()

generator_model = opt.generator_model  # default="Gen_Lin_Block"

classification = opt.classification
ut = opt.ut
z_step = 0.0001  # 噪声的学习率。
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
a_list = ['D', 'L', 'G', 'P', 'I', 'S', 'E', 'R', 'V', 'T', 'N', 'A', 'K', 'H', 'Q', 'M', 'Y', 'F', 'W', 'C', 'Z']
motif_list = []


def prepare_model(in_dim, max_len, amino_num):  # 准备模型
    if generator_model == "Gen_Lin_Block_CNN":
        G = Gen_Lin_Block_CNN(in_dim, max_len, amino_num,
                              opt.hidden, opt.batch)
    if generator_model == "Gen_Lin_Block":
        G = Gen_Lin_Block(in_dim, None, None)
    if use_cuda:
        G = G.cuda()
    print(G)
    return G


def my_seq_to_onehot(seq_arr, max_len):  # list, int

    for i, seq in enumerate(seq_arr):
        zs = max_len - len(seq)
        seq += ["Z"] * zs  # 长度不满足要求的序列在后面补“Z”

        df = pd.DataFrame(seq)

        enc = OneHotEncoder(sparse=False, categories=[a_list])

        seq_oh = enc.fit_transform(df)

        if i == 0:
            seq_nparr = seq_oh

        else:
            seq_nparr = np.block([[[seq_nparr]], [[seq_oh]]])  # 将两个数组组合在一起

    seq_nparr = seq_nparr.reshape(-1, len(a_list) * max_len)

    amino_num = len(a_list)  # default = 21

    return seq_nparr, amino_num, a_list


def generate_sample(sample_itr, batch_size, max_len, amino_num, G, a_list, motif_list):  # 生成序列
    sampled_seqs = []
    for _ in range(sample_itr):
        z = to_var(torch.randn(batch_size, max_len * amino_num))
        G.eval()
        sampled_seqs_tensor = G(z)
        sampled_seqs_tensor = sampled_seqs_tensor.reshape(
            -1, max_len, amino_num)
        sampled_seqs += tensor2str(sampled_seqs_tensor,
                                   a_list, motif_list, output=False)
    G.train()
    return sampled_seqs


def main():
    # ADNKFNKEQQNAFYEILHLPNLNEEQRNGFIQSLKDDPSQSANLLAEAKKLNDAQAPK
    # VDNKFNKEQQNAFYEILHLPNLTEEQRNAFIQSLKDDPSQSANLLAEAKKLNDAQAPK
    my_seq = [
        ['V', 'D', 'N', 'K', 'F', 'N', 'K', 'E', 'Q', 'Q', 'N', 'A', 'F', 'Y', 'E', 'I', 'L', 'H', 'L', 'P', 'N',
         'L', 'T', 'E', 'E', 'Q', 'R', 'N', 'A', 'F', 'I', 'Q', 'S', 'L', 'K', 'D', 'D', 'P', 'S', 'Q', 'S', 'A',
         'N', 'L', 'L', 'A', 'E', 'A', 'K', 'K', 'L', 'N', 'D', 'A', 'Q', 'A', 'P', 'K'],
        ['V', 'D', 'N', 'K', 'F', 'N', 'K', 'E', 'Q', 'Q', 'N', 'A', 'F', 'Y', 'E', 'I', 'L', 'H', 'L', 'P', 'N',
         'L', 'T', 'E', 'E', 'Q', 'R', 'N', 'A', 'F', 'I', 'Q', 'S', 'L', 'K', 'D', 'D', 'P', 'S', 'Q', 'S', 'A',
         'N', 'L', 'L', 'A', 'E', 'A', 'K', 'K', 'L', 'N', 'D', 'A', 'Q', 'A', 'P', 'K']]

    # my_seq = [
    #     ['A', 'D', 'N', 'K', 'F', 'N', 'K', 'E', 'Q', 'Q', 'N', 'A', 'F', 'Y', 'E', 'I', 'L', 'H', 'L', 'P', 'N',
    #      'L', 'N', 'E', 'E', 'Q', 'R', 'N', 'G', 'F', 'I', 'Q', 'S', 'L', 'K', 'D', 'D', 'P', 'S', 'Q', 'S', 'A',
    #      'N', 'L', 'L', 'A', 'E', 'A', 'K', 'K', 'L', 'N', 'D', 'A', 'Q', 'A', 'P', 'K'],
    #     ['A', 'D', 'N', 'K', 'F', 'N', 'K', 'E', 'Q', 'Q', 'N', 'A', 'F', 'Y', 'E', 'I', 'L', 'H', 'L', 'P', 'N',
    #      'L', 'N', 'E', 'E', 'Q', 'R', 'N', 'G', 'F', 'I', 'Q', 'S', 'L', 'K', 'D', 'D', 'P', 'S', 'Q', 'S', 'A',
    #      'N', 'L', 'L', 'A', 'E', 'A', 'K', 'K', 'L', 'N', 'D', 'A', 'Q', 'A', 'P', 'K']]

    # my_seq = [
    #     ['V', 'D', 'A', 'K', 'F', 'D', 'K', 'E', 'A', 'Q', 'E', 'A', 'F', 'Y', 'E', 'I', 'L', 'H', 'L', 'P', 'N',
    #      'L', 'T', 'E', 'E', 'Q', 'R', 'N', 'A', 'F', 'I', 'Q', 'S', 'L', 'K', 'D', 'D', 'P', 'S', 'V', 'S', 'K',
    #      'A', 'I', 'L', 'A', 'E', 'A', 'K', 'K', 'L', 'N', 'D', 'A', 'Q', 'A', 'P', 'K'],
    #     ['V', 'D', 'A', 'K', 'F', 'D', 'K', 'E', 'A', 'Q', 'E', 'A', 'F', 'Y', 'E', 'I', 'L', 'H', 'L', 'P', 'N',
    #      'L', 'T', 'E', 'E', 'Q', 'R', 'N', 'A', 'F', 'I', 'Q', 'S', 'L', 'K', 'D', 'D', 'P', 'S', 'V', 'S', 'K',
    #      'A', 'I', 'L', 'A', 'E', 'A', 'K', 'K', 'L', 'N', 'D', 'A', 'Q', 'A', 'P', 'K']]
    # ADNKFNKEQQNAFYEILHLPNLNEEQRNGFIQSLKDDPSQSANLLAEAKKLNDAQAPK
    want_seq = "VDAKFDKEAQEAFYEILHLPNLTEEQRNAFIQSLKDDPSVSKAILAEAKKLNDAQAPK"

    max_len = 58  # 最大长度
    amino_num = 21  # 20种氨基酸加Z补长。
    in_dim = max_len * amino_num  # amino_num default = 21. 58*21=1218
    seq_nparr, _, amino_acid_species = my_seq_to_onehot(my_seq, max_len)
    want_label = torch.from_numpy(seq_nparr)
    want_label = want_label.cuda()  # Tensor(1, 1218)
    # want_label = want_label  # Tensor(1, 1218)

    G = prepare_model(in_dim, max_len, amino_num)
    G_optimizer = optim.Adam(G.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    G.load_state_dict(torch.load('./checkpoint/WGAN2/g_weights.pth'))  # 加载生成器

    for i in range(2000):
        G.train()
        G_optimizer.zero_grad()
        z_original = Variable(Tensor(np.random.normal(0, 1, (2, in_dim))))  # (1, 1218)
        z_original.requires_grad = True
        z = z_original
        for j in range(5):
            generate_fake_data = G(z)  # 通过噪声生成序列
            # E_loss = torch.sum(torch.pow(generate_fake_data - want_label, 2), dim=-1)  # L2
            E_loss = torch.sum(generate_fake_data - want_label)
            gradient = autograd.grad(
                outputs=E_loss,
                inputs=z,
                grad_outputs=torch.ones_like(E_loss)
            )[0]
            z = z - z_step * gradient
        z_final = z
        optimisation_cost = torch.mean(torch.sum((z_final - z_original), -1))

        generate_fake_data = G(z_final)  # 通过噪声生成序列。G（优化Z）
        recons_loss = torch.mean(torch.sum(torch.pow(want_label - generate_fake_data, 2), dim=-1))  # 长码误差1

        want_label_2 = want_label
        want_label_2 = want_label_2.view(2, 58, 21)
        index = torch.tensor([17, 18, 19, 20, 21, 22,
                              35, 36, 37, 38, 39,  54, 55, 56, 57]).cuda()
        # want_label_2.index_fill_(1, index, 0)
        want_label_2.index_fill_(1, index, 0)
        want_label_2 = want_label_2.view(2, 1218)  # 2*1218

        generate_fake_data_2 = generate_fake_data.clone()
        # generate_fake_data_2 = generate_fake_data
        generate_fake_data_2 = generate_fake_data_2.view(2, 58, 21)
        generate_fake_data_2.index_fill_(1, index, 0)
        # generate_fake_data_2.index_fill_(1, index, 0)
        generate_fake_data_2 = generate_fake_data_2.view(2, 1218)  # 2*1218

        mask_loss = torch.mean(
            torch.sum(torch.pow(want_label_2 - generate_fake_data_2, 2), dim=-1))  # 掩码误差

        all_loss = mask_loss + 0.00001 * recons_loss

        all_loss.backward()
        G_optimizer.step()

        if i % 10 == 0:
            g_fake_data_all = generate_fake_data.reshape(
                -1, max_len, amino_num)
            sampled_seqs = tensor2str(g_fake_data_all, a_list, motif_list, output=False)
            print("final_seq:", sampled_seqs)
            print("opt_loss:", optimisation_cost)


if __name__ == '__main__':
    main()
