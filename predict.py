import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from models.finetune_causal_model import create_model
from scipy.spatial.distance import cdist
import argparse, json
from sklearn.decomposition import PCA
import pickle
from utils.utils import gpu_setup, load_yaml, load_state_dict
from data_proc.st_dataset import ST_DatasetLoad
import warnings
from skimage.util import img_as_float
from skimage.segmentation import slic, mark_boundaries
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import torch
n_seg = 3000
imgtest = 0
imgN = 0
CN = 32
from thop import clever_format, profile

warnings.filterwarnings("ignore")
def get_args():
    """ 执行参数 """
    parser = argparse.ArgumentParser(description='CausalCD')

    parser.add_argument('--cfg_path', default='configs.yaml', type=str, help="配置文件路径")

    parser.add_argument('--device', default='1', nargs='+', help="训练GPU id")

    parser.add_argument('--local_rank', default=-1, type=int, help='多GPU训练固定参数')

    print('cuda available with GPU:', torch.cuda.get_device_name(0))
    return parser.parse_args()

def Fill_border(image, border_size):
    # 获取原始图像的尺寸
    height, width = image.shape[:2]
    border_size = int(border_size)

    # 创建一个新的画布，尺寸为原始图像尺寸加上边框大小的两倍
    new_height = height + 2 * border_size
    new_width = width + 2 * border_size
    bordered_image = np.zeros((int(new_height), int(new_width), 3), dtype=np.uint8)

    # 将原始图像放置在新画布中心位置
    bordered_image[border_size:height+border_size, border_size:width+border_size] = image

    # 在四周添加黑色边框
    bordered_image[:border_size, :] = 0  # 上边框
    bordered_image[height+border_size:, :] = 0  # 下边框
    bordered_image[:, :border_size] = 0  # 左边框
    bordered_image[:, width+border_size:] = 0  # 右边框

    return bordered_image






if __name__=="__main__":
    start = time.time() 
    # 加载配置文件
    args = get_args()
    
    cfgs = load_yaml(args.cfg_path)
  
    # device = gpu_setup(cfgs['GPU']['use'], cfgs['GPU']['id'])
    device_ids = [args.device]
    device = torch.device("cuda:{}".format(device_ids[args.local_rank]))
    # 加载模型
    model = create_model(cfgs,device)
    model, *unuse = load_state_dict(cfgs['Train']['final_model'], model)

    print('#########Success load trained model!#########')
    model.eval().to(device)

    # # 输入双时相图像数据，计算差分图
    boundary = 16
    img1 = cv2.imread('datasets/' + 't1_' + str(imgtest) +'.bmp')
    img2 = cv2.imread('datasets/' + 't2_' + str(imgtest) +'.bmp')
    img = cv2.absdiff(img1, img2)
    label =  cv2.imread('datasets/' + 'label' + str(imgtest) +'.bmp')
    img1 = Fill_border(img1, boundary/2)
    img2 = Fill_border(img2, boundary/2)
    img = Fill_border(img, boundary/2)
    label = Fill_border(label, boundary/2)
    patch_size = 9
    step = 1
    h, w = img.shape[:2]
    img = img_as_float(img)
    # #  注释
    # superpixels1 = slic(img1, n_segments=500, compactness=20, channel_axis=-1)
    # superpixels2 = slic(img2, n_segments=500, compactness=20, channel_axis=-1)
    # superpixels = np.dstack((superpixels1, superpixels2))
    # superpixels0 = np.zeros(superpixels1.shape, dtype=int)
    # sp_num = 1
    # for i in range(superpixels1.max() + 1):
    #     for j in range(superpixels2.max() + 1):
    #         exists = np.any(np.all(superpixels == [i, j], axis=2))
    #         if (exists):
    #             mask = np.all(superpixels == [i, j], axis=2)
    #             count = np.sum(mask)
    #             if (count < 30 and sp_num > 1):
    #                 superpixels0[mask] = sp_num - 1
    #             else:
    #                 superpixels0[mask] = sp_num
    #                 sp_num += 1
    # #
    # # marked_img = mark_boundaries(img, superpixels)
    # # plt.imshow(marked_img)
    # # plt.axis('off')  # 不显示坐标轴
    # # plt.show()
    # # segmented_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # # for segment_label in np.unique(superpixels):
    # #     segmented_img[superpixels == segment_label] = np.random.randint(0, 255, 3)
    # #
    # # # 显示分割图
    # # plt.imshow(segmented_img)
    # # plt.axis('off')
    # # plt.show()
    #
    # data_total = []
    # sp_label = []
    # sp_data = []
    # for i in range(int(boundary / 2), h - int(boundary / 2), step):
    #     for j in range(int(boundary / 2), w - int(boundary / 2), step):
    #         data = []
    #         sp_label.append(label[i][j][0])
    #         # 构建空时地理数据-正样本
    #         img_patch1 = img1[i - int(patch_size / 2):i + int(patch_size / 2) + 1,
    #                      j - int(patch_size / 2):j + int(patch_size / 2) + 1]
    #         img_patch2 = img2[i - int(patch_size / 2):i + int(patch_size / 2) + 1,
    #                      j - int(patch_size / 2):j + int(patch_size / 2) + 1]
    #         for ii in range(patch_size):
    #             for jj in range(patch_size):
    #                 feature1 = torch.tensor([ii, jj, 1, img_patch1[ii][jj][0]]).float()
    #                 feature2 = torch.tensor([ii, jj, 2, img_patch2[ii][jj][0]]).float()
    #                 feature1 = torch.unsqueeze(feature1, dim=0)
    #                 feature2 = torch.unsqueeze(feature2, dim=0)
    #                 data.append(feature1)
    #                 data.append(feature2)
    #         data = torch.cat(data, dim=0)
    #         data = torch.unsqueeze(data, dim=0)
    #         data_total.append(data)
    #         # 构建空时语义图数据-正样本
    #         list_neighbor = []
    #         seg = superpixels0[i][j]
    #         for ii in range(i - int(boundary / 2), i + int(boundary / 2)):
    #             for jj in range(j - int(boundary / 2), j + int(boundary / 2)):
    #                 if (superpixels0[ii][jj] == seg):
    #                     list_neighbor.append([ii, jj])
    #         list_neighbor = np.array(list(list_neighbor))
    #         distances = cdist(np.array([[i, j]]), list_neighbor)
    #         closest_indices = np.argsort(distances)[0][:15]
    #         closest_points = list_neighbor[closest_indices]
    #         sp_coord0 = []
    #         sp_intensity = []
    #         for n in range(len(closest_points)):
    #             coord = closest_points[n]
    #             sp_coord0.append(coord)
    #             intensity = img[coord[0]][coord[1]][0]
    #             sp_intensity.append(intensity)
    #         sp_intensity = np.array(sp_intensity, np.float32)
    #         sp_intensity = torch.from_numpy(sp_intensity)
    #         sp_coord = np.array(sp_coord0, np.float32)
    #         sp_data.append([sp_intensity, sp_coord])
    #
    # sp_label = np.array(sp_label, np.int32)
    # data_total = torch.cat(data_total, dim=0)
    # with open('dataset_predict_true_'+ str(imgtest) + '.pkl', 'wb') as f:
    #     pickle.dump((sp_label, data_total, sp_data), f, protocol=2)
    # print('#########Success save predict dataset!#########')
     # 注释
    # 加载数据
    dataset = ST_DatasetLoad('dataset_predict'+ str(imgtest) + '.pkl', train_ratio = 1, shuffle=False)
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    print('#########Success load predict dataset!#########')
    train_loader = DataLoader(trainset, batch_size=100, shuffle=False, drop_last=False,
                              collate_fn=dataset.collate)
    target_array = np.zeros((h, w))
    label_total = [ ]
    # 读取分层聚类中心特征
    cf = pd.read_csv('causal/cluster_centers' + str(imgN) + '_' + str(CN) + '.csv', header=None)
    cf = torch.tensor(cf.values, dtype=torch.float32).to(device)

    # 读取分层聚类个数
    cn = pd.read_csv('causal/clusters_num' + str(imgN) + '_' + str(CN) + '.csv', header=None)
    cn = torch.tensor(cn.values, dtype=torch.float32).to(device)
    for iters, (points, labels, batch_graphs, batch_snorm_n, batch_snorm_e) in tqdm(enumerate(train_loader), desc='Data_Predicting', unit='iters'):
        # 清空GPU缓存
        torch.cuda.empty_cache()

        # 数据加载GPU
        points = points.data.numpy()
        points = torch.Tensor(points)
        points = points.to(device)
        batch_graphs.ndata['feat'] = torch.tensor(batch_graphs.ndata['feat'].detach().numpy().T[0])
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        labels = labels.to(device)
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
        output = model(points, batch_graphs, batch_x, cf, cn, batch_lap_pos_enc, batch_wl_pos_enc)
        # from sklearn.cluster import KMeans
        #
        # kmeans = KMeans(n_clusters=2, random_state=0).fit(outc0.cpu().detach().numpy())
        # cluster_labels = kmeans.labels_
        # # 使用PCA进行降维，从384维降到2维
        # from sklearn.manifold import TSNE
        # tsne = TSNE(n_components=2, random_state=42,perplexity=80)
        # reduced_data = tsne.fit_transform(outc0.cpu().detach().numpy())
        # # pca = PCA(n_components=2)
        # # reduced_data = pca.fit_transform(outc0.cpu().detach().numpy())
        # # 绘制聚类图
        # fig= plt.figure(figsize=(10, 8))
        # labels[labels == 255] = 1
        # # ax = fig.add_subplot(111, projection='3d')
        # # labels[labels == 255] = 1
        # # # 为两个聚类分别绘制3D散点图
        # # ax.scatter(reduced_data[labels.cpu().detach().numpy().flatten() == 0][:, 0], reduced_data[labels.cpu().detach().numpy().flatten() == 0][:, 1],
        # #            reduced_data[labels.cpu().detach().numpy().flatten() == 0][:, 2], label='Cluster 1', alpha=0.5)
        # # ax.scatter(reduced_data[labels.cpu().detach().numpy().flatten() == 1][:, 0], reduced_data[labels.cpu().detach().numpy().flatten() == 1][:, 1],
        # #            reduced_data[labels.cpu().detach().numpy().flatten() == 1][:, 2], label='Cluster 2', alpha=0.5)
        # #
        # # # 添加图例
        # # ax.legend()
        # #
        # # # 添加坐标轴标签和标题
        # # ax.set_xlabel('Principal Component 1')
        # # ax.set_ylabel('Principal Component 2')
        # # ax.set_zlabel('Principal Component 3')
        # # ax.set_title('3D PCA Projection of the Clustering')
        # #
        # # # 显示图表
        # # plt.show()
        # # 为两个聚类分别绘制散点图
        # plt.scatter(reduced_data[labels.cpu().detach().numpy().flatten() == 0][:, 0],
        #             reduced_data[labels.cpu().detach().numpy().flatten() == 0][:, 1], label='Cluster 1', alpha=0.5)
        # plt.scatter(reduced_data[labels.cpu().detach().numpy().flatten() == 1][:, 0],
        #             reduced_data[labels.cpu().detach().numpy().flatten() == 1][:, 1], label='Cluster 2', alpha=0.5)
        # # 添加图例
        # plt.legend()
        # # 添加坐标轴标签和标题
        # plt.xlabel('Principal Component 1')
        # plt.ylabel('Principal Component 2')
        # plt.title('2D PCA Projection of the Clustering')
        # plt.show()
        output= output.detach().argmax(dim=1)

        # flops, params = profile(model, inputs=(points, batch_graphs, batch_x,  cf, cn,batch_lap_pos_enc, batch_wl_pos_enc))
        #
        # # thop返回的flops是一个非常大的数，因此我们用clever_format来格式化它
        # flops, params = clever_format([flops, params], "%.3f")
        #
        #
        # print(f"模型的参数数量: {params}")
        # print(f"模型的FLOPs: {flops}")
        # output = model(points, batch_graphs, batch_x, batch_lap_pos_enc, batch_wl_pos_enc).detach().argmax(
        #     dim=1)
        label_total.append(output)

    label_total = torch.cat(label_total, dim=0)
    label_total = label_total.reshape(h - 2 * int(boundary / 2), w - 2 * int(boundary / 2))
    label_array = label_total.cpu().numpy()
    # start_row = int(boundary / 2)
    # start_col = int(boundary / 2)
    # end_row = h - int(boundary / 2)
    # end_col = w - int(boundary / 2)
    # target_array[start_row:end_row, start_col:end_col] = label_array
    normalized_array = np.interp(label_array, (label_array.min(), label_array.max()), (0, 255))
    image_array = normalized_array.astype(np.uint8)
    cv2.imwrite('changemap_'+'_' + str(imgtest) + '_' + str(CN) + '_pseudo_causal_w.bmp', image_array)
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
    print('######### Predict Over!#########')
