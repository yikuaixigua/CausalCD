import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import os
N = 3
imgN = 16
star = time.time()  # 计时
img = plt.imread(r'../datasets/' + 'di' + str(imgN) +'.bmp')
label_img = cv2.imread(r'../datasets/' + 'label' + str(imgN) +'.bmp')
# 读取图片信息，存储在一个三维数组中
row = img.shape[0]
col = img.shape[1]
plt.figure(1)
plt.subplot(221)
plt.imshow(img)


def fcm(data, threshold, k, m):
    # 0.初始化


    data = data.reshape(-1, 3)
    cluster_center = np.zeros([k, 3])  # 簇心
    distance = np.zeros([k, row * col])  # 欧氏距离
    times = 0  # 迭代次数
    goal_j = np.array([])  # 迭代终止条件：目标函数
    goal_u = np.array([])  # 迭代终止条件：隶属度矩阵元素最大变化量
    # 1.初始化U
    u = np.random.dirichlet(np.ones(k), row * col).T  # 形状（k, col*rol），任意一列元素和=1
    #  for s in range(50):
    while 1:
        times += 1
        print('循环：', times)
        # 2.簇心update
        for i in range(k):
            cluster_center[i] = np.sum((np.tile(u[i] ** m, (3, 1))).T * data, axis=0) / np.sum(u[i] ** m)
        # 3.U update
        # 3.1欧拉距离
        for i in range(k):
            distance[i] = np.sqrt(np.sum((data - np.tile(cluster_center[i], (row * col, 1))) ** 2, axis=1))
        # 3.2目标函数
        goal_j = np.append(goal_j, np.sum((u ** m) * distance ** 2))
        # 3.3 更新隶属度矩阵
        oldu = u.copy()  # 记录上一次隶属度矩阵
        u = np.zeros([k, row * col])
        for i in range(k):
            for j in range(k):
                u[i] += (distance[i] / distance[j]) ** (2 / (m - 1))
            u[i] = 1 / u[i]
        goal_u = np.append(goal_u, np.max(u - oldu))  # 隶属度元素最大变化量
        print('隶属度元素最大变化量', np.max(u - oldu), '目标函数', np.sum((u ** m) * distance ** 2))
        # 4.判断：隶属度矩阵元素最大变化量是否小于阈值
        if np.max(u - oldu) <= threshold:
            break
    return u, goal_j, goal_u

#计算聚类后每一类别像素平均值
def class_mean(img_show, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pseudo_img = np.ones(img.shape[0] * img.shape[1])
    pseudo_label = np.ones(img.shape[0] * img.shape[1])
    img = img.reshape(img.shape[0] * img.shape[1])
    pixel_value = np.zeros(N)
    pixel_num = np.zeros(N)
    pixel_mean = np.zeros(N)
    for i in range(img_show.shape[0]):
        pixel_value[img_show[i]] += img[i]
        pixel_num[img_show[i]] += 1
    for j in range(N):
        pixel_mean[j] = pixel_value[j] / pixel_num[j]
    # 找到最大值的索引
    index_of_max = np.argmax(pixel_mean)
    # 找到最小值的索引
    index_of_min = np.argmin(pixel_mean)

    pseudo_num = pixel_num[index_of_max] + pixel_num[index_of_min]
    print(pixel_num[index_of_max])
    print(pixel_num[index_of_min])
    pseudo_img[img_show == index_of_min] = 0
    pseudo_img[img_show == index_of_max] =255
    pseudo_img = pseudo_img.reshape([row, col])
    cv2.imwrite(str(imgN) + '.bmp', pseudo_img)
    # pseudo_label[img_show == index_of_min] = 0
    # pseudo_label[img_show == index_of_max] = 255
    # pseudo_label = pseudo_img.reshape([row, col])
    # cv2.imwrite('pseudo_label0.bmp', pseudo_img)
    return  pseudo_img, pseudo_num

#计算伪标签和真实label之间关系f
def pseudo_truth(label_img, pseudo_img, pseudo_num):
    ratio = (np.sum(label_img == pseudo_img) / pseudo_num)
    print(f"positative:{np.sum(label_img == pseudo_img):.2f}")
    print(f"negative:{pseudo_num-np.sum(label_img == pseudo_img):.2f}")
    print(f"ratio:{ratio:.4f}")


if __name__ == '__main__':
    if img.ndim == 2:
        img = np.stack((img,) * 3, axis=-1)
    img_show, goal1_j, goal2_u = fcm(img, 1e-09, N, 2)
    img_show = np.argmax(img_show, axis=0)

    label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
    pseudo_img, pseudo_num = class_mean(img_show, img)
    pseudo_truth(label_img, pseudo_img, pseudo_num)


