from models import myresnet
import torch
from torchvision.transforms import transforms
import torch.nn as nn
from mydataset import Nutrition_RGBD
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from collections import OrderedDict
from models import myresnet,pvt
import pandas as pd  

test_transform = transforms.Compose([
    transforms.Resize((238, 238)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
data_root = "/Datasets/nutrition5k_dataset"

nutrition_rgbd_ims_root = os.path.join(data_root, 'imagery')
nutrition_test_txt = os.path.join(data_root, 'imagery', 'rgbd_test_processed.txt')  # depth_color.png
nutrition_test_rgbd_txt = os.path.join(data_root, 'imagery', 'rgb_in_overhead_test_processed.txt')  # rbg.png

testset = Nutrition_RGBD(nutrition_rgbd_ims_root, nutrition_test_rgbd_txt, nutrition_test_txt, transform = test_transform,transform_D = test_transform)

test_loader = DataLoader(testset,
                         batch_size=1,
                         shuffle=False,
                         num_workers=4,
                         pin_memory=True
                         )

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
#############################################
net_rgb = pvt.PolypPVT()
net_rgbd = pvt.PolypPVT()
pretrained_dict = net_rgb.state_dict()
#####################################################################
net_cat = myresnet.RGBD_Fusion()

checkpoint_path = '/Path/ckpt_RGBD.pth'

models_state_dict = torch.load(checkpoint_path)
net_rgb.load_state_dict(models_state_dict["net"])

new_state_dict_catd = OrderedDict()
for k, v in models_state_dict['net_d'].items():
    name = k[7:] if k.startswith('module') else k
    new_state_dict_catd[name] = v
net_rgbd.load_state_dict(new_state_dict_catd)

new_state_dict_cat = OrderedDict()
for k, v in models_state_dict['net_cat'].items():
    name = k[7:] if k.startswith('module') else k
    new_state_dict_cat[name] = v
net_cat.load_state_dict(new_state_dict_cat)

net_rgb.to(device)
net_rgbd.to(device)
net_cat.to(device)

net_rgb.eval()
net_rgbd.eval()
net_cat.eval()

criterion = nn.L1Loss()

epoch_iterator = tqdm(test_loader,
                          desc="Testing... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)

test_loss = 0
calories_loss = 0
mass_loss = 0
fat_loss = 0
carb_loss = 0
protein_loss = 0
import pandas as pd
import numpy as np
import torch
import csv

# 定义一个函数来保存CSV和计算指标
def save_and_calculate_metrics(output_list, true_list, column_name, file_path):
    # 将列表转换为 DataFrame
    df = pd.DataFrame({
        'total_calories': true_list,
        'output_calories': output_list
    })
    # 保存 DataFrame 到 CSV 文件
    df.to_csv(file_path, index=False)
    print(f"保存完成: {file_path}")

    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 计算 MAE 和 MAPE
    mae = np.mean(np.abs(df['total_calories'].astype(float).values - df['output_calories'].astype(float).values))
    mape = 100 * np.mean(np.abs(df['total_calories'].astype(float).values - df['output_calories'].astype(float).values)) / np.mean(df['total_calories'].astype(float).values)
    print(f"Mean Absolute Error ({column_name}_MAE): {mae}")
    print(f"Mean Absolute Percentage Error ({column_name}_MAPE): {mape}")

# 初始化列表
total_calories_list = []
total_mass_list = []
total_fat_list = []
total_carb_list = []
total_protein_list = []

output_calories_list = []
output_mass_list = []
output_fat_list = []
output_carb_list = []
output_protein_list = []

with torch.no_grad():
    for batch_idx, x in enumerate(epoch_iterator):
        inputs = x[0].to(device)
        total_calories = x[2].to(device).float()
        total_mass = x[3].to(device).float()
        total_fat = x[4].to(device).float()
        total_carb = x[5].to(device).float()
        total_protein = x[6].to(device).float()
        inputs_rgbd = x[7].to(device)

        x1, x2, x3, x4 = net_rgb(inputs)
        F1, F2, F3, F4 = net_rgbd(inputs_rgbd)
        outputs = net_cat([x1, x2, x3, x4], [F1, F2, F3, F4])

        # 收集真实值和预测值
        total_calories_list.append(total_calories.item())
        total_mass_list.append(total_mass.item())
        total_fat_list.append(total_fat.item())
        total_carb_list.append(total_carb.item())
        total_protein_list.append(total_protein.item())

        output_calories_list.append(outputs[0].cpu().detach().numpy())
        output_mass_list.append(outputs[1].cpu().detach().numpy())
        output_fat_list.append(outputs[2].cpu().detach().numpy())
        output_carb_list.append(outputs[3].cpu().detach().numpy())
        output_protein_list.append(outputs[4].cpu().detach().numpy())

# 定义文件路径
base_path = './saved/'

# 循环调用函数保存和计算指标
save_and_calculate_metrics(output_calories_list, total_calories_list, 'calories', base_path + 'calories_output.csv')
save_and_calculate_metrics(output_mass_list, total_mass_list, 'mass', base_path + 'mass_output.csv')
save_and_calculate_metrics(output_fat_list, total_fat_list, 'fat', base_path + 'fat_output.csv')
save_and_calculate_metrics(output_carb_list, total_carb_list, 'carb', base_path + 'carb_output.csv')
save_and_calculate_metrics(output_protein_list, total_protein_list, 'protein', base_path + 'protein_output.csv')