import os
import json
from os.path import join

import numpy as np
import scipy
from scipy import io
import scipy.misc
from PIL import Image
import pandas as pd
# import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, list_dir, check_integrity, extract_archive, verify_str_arg

import imageio
import cv2
import pdb

class Nutrition(Dataset):
    def __init__(self, image_path, txt_dir, transform=None):

        file = open(txt_dir, 'r')
        lines = file.readlines()
        self.images = []
        self.labels = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []
        # pdb.set_trace()
        for line in lines:
            image = line.split()[0]  # side_angles/dish_1550862840/frames_sampled5/camera_A_frame_010.jpeg
            label = line.strip().split()[1]  # 类别 1-
            calories = line.strip().split()[2]
            mass =  line.strip().split()[3]
            fat = line.strip().split()[4]
            carb = line.strip().split()[5]
            protein = line.strip().split()[6]

            self.images += [os.path.join(image_path, image)]  # 每张图片路径
            self.labels += [str(label)]
            self.total_calories += [np.array(float(calories))]
            self.total_mass += [np.array(float(mass))]
            self.total_fat += [np.array(float(fat))]
            self.total_carb += [np.array(float(carb))]
            self.total_protein += [np.array(float(protein))]
        # pdb.set_trace()
        # self.transform_rgb = transform[0]

        self.transform = transform
       
    def __getitem__(self, index):
        # img = cv2.imread(self.images[index])  
        # try:
        #     # img = cv2.resize(img, (self.imsize, self.imsize))
        #     img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) # cv2转PIL
        # except:
        #     print("图片有误：",self.images[index])
        img = Image.open(self.images[index]).convert('RGB')
        if self.transform is not None:
            try:
                #lmj  RGB-D图像尺寸不同,按照不同比例缩放
                if 'realsense_overhead' in self.images[index]:
                    # pdb.set_trace()
                    self.transform.transforms[0].size = (267, 356)
                    # print(self.transform)
                img = self.transform(img)
            except:
                # print('trans_img', img)
                print('trans_img有误')
        return img, self.labels[index], self.total_calories[index], self.total_mass[index], self.total_fat[index], self.total_carb[index], self.total_protein[index]

    def __len__(self):
        return len(self.images)



############################################
import os
import json
import pandas as pd
import numpy as np
import re
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation
import random

class Nutrition_RGBD(Dataset):
    def __init__(self, dataset_path, split_info_path, nutrition_file_path, mode='train', 
                 target_frames=[100, 150, 199], transform=None, transform_D=None):
        """
        MetaFood3D数据集加载器
        
        Args:
            dataset_path: 数据集根目录路径
            split_info_path: split_info.json文件路径
            nutrition_file_path: MetaFood3D_nutrition.xlsx文件路径
            mode: 'train'或'test'
            target_frames: 要读取的帧索引列表
            transform: RGB图像变换
            transform_D: 深度图像变换
        """
        self.dataset_path = dataset_path
        self.mode = mode
        self.target_frames = target_frames
        self.transform = transform
        self.transform_D = transform_D
        
        # 加载划分信息
        with open(split_info_path, 'r') as f:
            self.split_info = json.load(f)
        
        # 加载营养信息
        self.nutrition_df = pd.read_excel(nutrition_file_path)
        
        # 根据模式选择实例
        self.instances = self.split_info['train_instances'] if mode == 'train' else self.split_info['test_instances']
        
        # 初始化数据列表
        self.rgb_images = []
        self.depth_images = []
        self.labels = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []
        
        # 构建数据列表
        self._build_data_list()
    
    def _find_nutrition_info(self, food_category, food_instance):
        """根据食物类别和实例名称查找营养信息"""
        # 在营养数据中查找匹配的行
        mask = (self.nutrition_df['Object_name'] == food_category) & \
               (self.nutrition_df['Food_Type'] == food_instance)
        
        matching_rows = self.nutrition_df[mask]
        
        if len(matching_rows) == 0:
            print(f"警告: 未找到 {food_category}/{food_instance} 的营养信息")
            return None
        
        # 返回第一行匹配的数据
        return matching_rows.iloc[0]
    
    def _find_frame_file(self, directory, target_frame):
        """在目录中查找包含目标帧数字的文件"""
        if not os.path.exists(directory):
            return None
        
        # 获取目录中的所有文件
        files = os.listdir(directory)
        
        # 查找包含目标帧数字的文件
        for file in files:
            # 使用正则表达式匹配文件名中的数字
            frame_numbers = re.findall(r'\d+', file)
            
            # 检查是否包含目标帧数字
            for frame_str in frame_numbers:
                try:
                    frame_num = int(frame_str)
                    if frame_num == target_frame:
                        return os.path.join(directory, file)
                except ValueError:
                    continue
        
        return None
    
    def _build_data_list(self):
        """构建数据列表"""
        for instance_path in self.instances:
            food_category, food_instance = instance_path.split('/')
            
            # 获取营养信息
            nutrition_info = self._find_nutrition_info(food_category, food_instance)
            if nutrition_info is None:
                continue
            
            # 构建图像路径
            rgb_base_path = os.path.join(self.dataset_path, food_category, food_instance, "original")
            depth_base_path = os.path.join(self.dataset_path, food_category, food_instance, "depth")
            
            # 检查路径是否存在
            if not os.path.exists(rgb_base_path) or not os.path.exists(depth_base_path):
                print(f"警告: {instance_path} 的图像路径不存在")
                continue
            
            # 添加目标帧
            for frame_idx in self.target_frames:
                # 查找RGB图像文件
                rgb_path = self._find_frame_file(rgb_base_path, frame_idx)
                if rgb_path is None:
                    print(f"警告: {instance_path} 的RGB帧 {frame_idx} 不存在")
                    continue
                
                # 查找深度图像文件
                depth_path = self._find_frame_file(depth_base_path, frame_idx)
                if depth_path is None:
                    print(f"警告: {instance_path} 的深度帧 {frame_idx} 不存在")
                    continue
                
                # 检查文件是否存在
                if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                    print(f"警告: {instance_path} 的帧 {frame_idx} 文件不存在")
                    continue
                
                # 添加到数据列表
                self.rgb_images.append(rgb_path)
                self.depth_images.append(depth_path)
                self.labels.append(f"{food_category}_{food_instance}")
                self.total_calories.append(np.array(float(nutrition_info['Energy (Kcal)'])))
                self.total_mass.append(np.array(float(nutrition_info['Weight (g)'])))
                self.total_fat.append(np.array(float(nutrition_info['Fat (g)'])))
                self.total_carb.append(np.array(float(nutrition_info['Carbs (g)'])))
                self.total_protein.append(np.array(float(nutrition_info['Protein (g)'])))
    
    def __len__(self):
        return len(self.rgb_images)
    
    def __getitem__(self, index):
        # 加载RGB图像
        rgb_img = cv2.imread(self.rgb_images[index])
        if rgb_img is None:
            print(f"错误: 无法读取RGB图像 {self.rgb_images[index]}")
            return None
        
        # 加载深度图像
        depth_img = cv2.imread(self.depth_images[index], cv2.IMREAD_GRAYSCALE)
        if depth_img is None:
            print(f"错误: 无法读取深度图像 {self.depth_images[index]}")
            return None
        
        # 转换为PIL图像
        try:
            rgb_img = Image.fromarray(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
            depth_img = Image.fromarray(depth_img)
        except Exception as e:
            print(f"图像转换错误: {e}")
            return None
        
        # 应用变换
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
        
        if self.transform_D is not None:
            depth_img = self.transform_D(depth_img)
        
        return (
            rgb_img, 
            self.labels[index], 
            self.total_calories[index], 
            self.total_mass[index], 
            self.total_fat[index], 
            self.total_carb[index], 
            self.total_protein[index], 
            depth_img
        )
    def __len__(self):
        return len(self.rgb_images)
##############################################
    
        
#20210526
class Food(Dataset):

    def __init__(self, txt_dir, image_path, transform=None):
        data_txt = open(txt_dir, 'r')
        imgs = []
        for line in data_txt:
            line = line.strip()
            words = line.split(' ')
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.image_path = image_path
 
    def __len__(self):
        
        return len(self.imgs)
  
    def __getitem__(self, index):
        img_name, label = self.imgs[index]

        # label = list(map(int, label))
        # print label

        # print type(label)
   
        image = Image.open(os.path.join(self.image_path, img_name)).convert('RGB')

        # print img
        if self.transform is not None:
            img = self.transform(image)
            # print img.size()
            # label =torch.Tensor(label)

            # print label.size()
        return img, label

class ImagesForMulCls(Dataset):

    def __init__(self, args, ims_root, category_list,ingredient_list, imsize=224, bbxs=None, transform=None):

        self.root = ims_root
        if args.dataset == 'food172':
            self.images_fn, self.clusters, self.mul_clusters = self.get_imgs_food172(ims_root, ingredient_list)
        elif args.dataset == 'food101':
            self.images_fn, self.clusters, self.mul_clusters = self.get_imgs_food101(ims_root, category_list, ingredient_list)
        self.imsize = imsize
        self.transform = transform

    def get_imgs_food101(self, ims_root, category_list, ingredient_list):
        # pdb.set_trace()
        if not os.path.exists(category_list) or not os.path.exists(ingredient_list):
            print('!!!THE FILE ROOT NOT EXIST!!!')
            pass
        category_file = open(category_list)
        ingredient_file = open(ingredient_list)
        images = [] # 图片路径
        clusters = [] # 类别标签
        mul_clusters = [] # 食材列表

        for line in category_file.readlines():
            image = line.split()[0]  # apple_pie/1005649.jpg
            label = line.strip().split()[1]  # 食物类别
            images += [os.path.join(ims_root, image)]  # 每张图片路径
            clusters += [int(label)]  # 对应的标签
        for line in ingredient_file.readlines():
             mult_label = line.strip().split()[1:]  # 食材标签
             mul_clusters += [[int(i) for i in mult_label]]  # 食材列表

        return images, np.array(clusters), np.array(mul_clusters)

    def get_imgs_food172(self, ims_root, ingredient_list):
        pdb.set_trace()
        if not os.path.exists(ingredient_list):
            print(ingredient_list)
            pass
        file = open(ingredient_list)
        lines = file.readlines()
        images = []
        clusters = []
        mul_clusters = []
        for line in lines:
            image = line.split()[0]  # 1/21_20.jpg
            label = line.strip().split()[1]  # 食物类别
            mult_label = line.strip().split()[2:]  # 食材标签

            images += [os.path.join(ims_root, image)]  # 每张图片路径
            clusters += [int(label)]  # 对应的标签
            mul_clusters += [[int(i) for i in mult_label]]  # 食材列表
        return images, np.array(clusters), np.array(mul_clusters)

    def __getitem__(self, index):
        img = cv2.imread(self.images_fn[index])  # self.images_fn[index] ：图片路径
        try:
            img = cv2.resize(img, (self.imsize, self.imsize))
            img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) # cv2装PIL
        except:
            print("图片有误：",self.images_fn[index])
            # print("图片有误：")
        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                # print('trans_img', img)
                print('trans_img有误')

        return img, self.clusters[index], self.mul_clusters[index]  # 图片 类别标签 食材列表

    def __len__(self):
        return len(self.images_fn)

class CUB():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))

        # pdb.set_trace()
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        if self.is_train:
            # self.train_img = [scipy.misc.imread(os.path.join(self.root, 'images', train_file)) for train_file in
            #                   train_file_list[:data_len]]
            self.train_img = [imageio.imread(os.path.join(self.root, 'images', train_file)) for train_file in
                              train_file_list[:data_len]]
            
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
            self.train_imgname = [x for x in train_file_list[:data_len]]
        if not self.is_train:
            # self.test_img = [scipy.misc.imread(os.path.join(self.root, 'images', test_file)) for test_file in
            #                  test_file_list[:data_len]]
            self.test_img = [imageio.imread(os.path.join(self.root, 'images', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
            self.test_imgname = [x for x in test_file_list[:data_len]]
    def __getitem__(self, index):
        if self.is_train:
            img, target, imgname = self.train_img[index], self.train_label[index], self.train_imgname[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)
        else:
            img, target, imgname = self.test_img[index], self.test_label[index], self.test_imgname[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)