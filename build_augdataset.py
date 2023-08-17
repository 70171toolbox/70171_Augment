import cv2
import numpy as np
from utils.detector_data_augmentation import Augmentation


# dataset路徑，也是aug後圖片存放的路徑，需要放絕對路徑
data_path = 'D:/ray_workspace/Augment/Dateset'

# 若設定img_size則會得到固定大小的Aug圖片，mean, std要先另外計算，則可以做正規化。
img_size = None
mean, std = None, None

# Information 的路徑，要按照指定規定
info_path = 'train_info.txt'

# 原dataset 的圖片數量
original_data_num = 35

aug = Augmentation(img_size, mean, std)

with open(info_path, encoding='utf-8') as f:
    train_lines = f.readlines()[:original_data_num]

with open(info_path, 'r') as f:
    data_num = len(f.readlines())

for i in range(original_data_num):
    info = train_lines[i].split()

    img = cv2.imread(info[0])
    targets = np.array([list(map(np.int64,map(float, target.split(',')))) for target in info[1:]])
    bboxes = targets[:, 0:4]
    labels = targets[:, 4]

    while(True):
        img_a, bboxes_a, labels_a = aug(img, bboxes, labels)
        if len(bboxes_a) != 0:
            break

    info_a = []
    for j in range(len(bboxes_a)):
        info_a.append(f'{bboxes_a[j][0]},{bboxes_a[j][1]},{bboxes_a[j][2]},{bboxes_a[j][3]},{labels_a[j]}')

    with open(info_path, 'a') as f:
        f.write('\n'+ data_path + f'/aug_{(data_num/original_data_num-1)*35+i}.jpg' + ' ' + ' '.join(info_a[:]))

    cv2.imwrite(data_path + f'/aug_{(data_num/original_data_num-1)*35+i}.jpg', img_a)

# data_path = 'aa'
# bboxes_a = [[1, 2, 3, 4], [6, 7, 8, 4]]
# labels_a = [1, 3]
# info_a = []
# for i in range(len(bboxes_a)):
#     info_a.append(f'{bboxes_a[i][0]},{bboxes_a[i][1]},{bboxes_a[i][2]},{bboxes_a[i][3]},{labels_a[i]}')
# print(data_path + ' ' + ' '.join(info_a[:]))

