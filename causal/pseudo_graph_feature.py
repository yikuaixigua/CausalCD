import cv2
import numpy as np
import torch

import torch
import argparse
from utils.utils import gpu_setup, load_yaml, load_state_dict, show_yaml, build_save_dir, TensorboardWriter, save_state_dict
from models.model import create_model
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from data_proc.st_dataset import ST_DatasetLoad
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import csv
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
imgN = 0
N = 32
def get_args():
    """ 执行参数 """
    parser = argparse.ArgumentParser(description='spatial-time graph network with cl and transformer')

    parser.add_argument('--cfg_path', default='../configs.yaml', type=str, help="配置文件路径")

    parser.add_argument('--device', default='1', nargs='+', help="训练GPU id")

    parser.add_argument('--local_rank', default=-1, type=int, help='多GPU训练固定参数')

    print('cuda available with GPU:', torch.cuda.get_device_name(0))

    return parser.parse_args()

def main(args):
    # 加载配置文件
    cfgs = load_yaml(args.cfg_path)
    # 加载GPU
    device_ids = [args.device]
    device = torch.device("cuda:{}".format(device_ids[args.local_rank]))
    # 加载模型
    model = create_model(cfgs)

    checkpoint = torch.load('../checkpoints/best_w.pkl')
    checkpoint_model = checkpoint['module']
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in checkpoint_model.items() if k in model_dict.keys()}
    model.load_state_dict(state_dict, strict=False)
    print('Success load pre-trained model!')
    model.eval().to(device)
    print('参数配置：')
    show_yaml(trace=print, args=cfgs)
# 注释
    dataset = ST_DatasetLoad('../data_proc/dataset_finetune' + str(imgN) +'-pseudo.pkl', train_ratio=1, shuffle=False)
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    print('#########Success load predict dataset!#########')
    train_loader = DataLoader(trainset, batch_size=100, shuffle=False, drop_last=False,
                              collate_fn=dataset.collate)


    with open('graph_feature' + str(imgN) + '.csv', 'w', newline='') as csvfile:
        tensor_writer = csv.writer(csvfile)
        for iters, (points, labels, batch_graphs, batch_snorm_n, batch_snorm_e) in tqdm(enumerate(train_loader),
                                                                                        desc='Data_Predicting',
                                                                                        unit='iters'):
            # 清空GPU缓存
            torch.cuda.empty_cache()

            # 数据加载GPU
            points = points.data.numpy()
            points = torch.Tensor(points)
            points = points.to(device)
            batch_graphs.ndata['feat'] = torch.tensor(batch_graphs.ndata['feat'].detach().numpy().T[0])
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
            try:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
                sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
                sign_flip[sign_flip >= 0.5] = 1.0
                sign_flip[sign_flip < 0.5] = -1.0
                batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
            except:
                batch_lap_pos_enc = None

            try:
                batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
            except:
                batch_wl_pos_enc = None
            output = model(points, batch_graphs, batch_x, batch_lap_pos_enc, batch_wl_pos_enc)
            output = output.cpu().detach().numpy()
            for row in output:
                tensor_writer.writerow(row)
# 注释
    df = pd.read_csv('graph_feature' + str(imgN) + '.csv')
    feature_kmeans(df, N)


def feature_kmeans(df, k):
    kmeans = KMeans(n_clusters=k, random_state=0)  # 使用16个聚类中心
    kmeans.fit(df)
    labels = kmeans.labels_




    # 计算每个簇的样本数
    counts = np.bincount(labels) / len(labels)
    counts = pd.DataFrame(counts.T)


    #  保存聚类中心到CSV文件
    centers = kmeans.cluster_centers_
    centers_df = pd.DataFrame(centers)
    # 保存到新的CSV文件
    counts.to_csv('clusters_num' + str(imgN) + '_' + str(N) + '.csv', index=False, header=False)
    centers_df.to_csv('cluster_centers' + str(imgN) + '_' + str(N) + '.csv', index=False, header=False)

    #
    # pca = PCA(n_components=3)
    # X_pca = pca.fit_transform(df)
    # centerspca = pca.transform(kmeans.cluster_centers_)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # 绘制数据点
    # scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, s=1, cmap='viridis', label=labels)
    # # ax.scatter(centerspca[:, 0], centerspca[:, 1], centerspca[:, 2], c='red', s=200, alpha=0.5, marker='X')
    # # ax.set_title('3D PCA Visualization of K-means Clustering')
    # # ax.set_xlabel('Principal Component 1')
    # # ax.set_ylabel('Principal Component 2')
    # # ax.set_zlabel('Principal Component 3')
    # #
    # # # 显示图例
    # # legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
    # # ax.add_artist(legend1)
    #
    # # 显示图表
    # plt.savefig('clustered_data_3d-16.png', dpi=300)
    # plt.show()

    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=50, cmap='viridis')
    #
    # plt.scatter(centerspca[:, 0], centerspca[:, 1], c='red', s=200, alpha=0.5, marker='X')
    # plt.title('PCA + K-means Clustering')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.show()

# 显示图表
plt.show()
if __name__=="__main__":
    args = get_args()
    main(args)