# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
'''https://blog.csdn.net/u013841196/article/details/82941410 采用网络多阶段特征融合'''
from re import S
import torch
import torch.nn as nn
from torch.nn.modules import module
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision.models.inception import InceptionOutputs
import torchvision.transforms as transforms
import os
import argparse
from models import *
from models import myresnet,pvt
from timm.models import *
from timm.models import create_model
from utils.utils import progress_bar,load_for_transfer_learning,logtxt,check_dirs
from utils_data import get_DataLoader
from utils.utils_scheduler import WarmupCosineSchedule
from mydataset import Food
import pdb
from tqdm import tqdm
import random
import numpy as np
from collections import OrderedDict
import csv
import random
from utils.AutomaticWeightedLoss import AutomaticWeightedLoss

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=0.9, type=float, help='weight decay') # 5e-4
parser.add_argument('--min_lr', default=2e-4, type=float, help='minimal learning rate')#2e-4
parser.add_argument('--dataset', choices=["nutrition_rgbd","nutrition_rgb","food101","food172","cub200/CUB_200_2011","cifar10","cifar100"], default='cifar10',
                    help='cifar10 or cifar100')
parser.add_argument('--b', type=int, default=8,
                    help='batch size')
parser.add_argument('--resume', '-r', type=str,
                    help='resume from checkpoint')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--num_classes', type=int, default=1024, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--model', default='T2t_vit_t_14', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception":必须和t2t_vit.py中的 default_cfgs 命名相同')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.0)')
parser.add_argument('--drop_connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop_block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img_size', type=int, default=224, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--bn_tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn_momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn_eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--initial_checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--transfer_learning', default=False,
                    help='Enable transfer learning')
parser.add_argument('--transfer_model', type=str, default=None,
                    help='Path to pretrained model for transfer learning')
parser.add_argument('--transfer_ratio', type=float, default=0.01,
                    help='lr ratio between classifier and backbone in transfer learning')
parser.add_argument('--data_root', type=str, default = "/Datasets/nutrition5k_dataset", help="our dataset root")
parser.add_argument('--run_name',type=str, default="editname")
parser.add_argument('--print_freq', type=int, default=200,help="the frequency of write to logtxt" )
parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
parser.add_argument('--mul_cls_num', default=174, type=int, metavar='N', help='ingradient class number') #353 
parser.add_argument('--multi_task',action='store_true',  help='multi-task classification')
parser.add_argument('--pool', default='spoc', type=str, help='pool function')
parser.add_argument('--embed_dim', default=384, type=int, help='T2t_vit_7,T2t_vit_10,T2t_vit_12:256;\
T2t_vit_14:384; T2t_vit_19:448; T2t_vit_24:512')
parser.add_argument('--seed', type=int, default=42,help="random seed for initialization")
parser.add_argument('--portion_independent',action='store_true',  help='Nutrition5K: Portion Independent Model')
parser.add_argument('--direct_prediction',action='store_true',  help='Nutrition5K: direct_prediction Model')
parser.add_argument('--rgbd',action='store_true',  help='4 channels')
parser.add_argument('--gradnorm',action='store_true',  help='GradNorm')
parser.add_argument('--alpha', '-a', type=float, default=0.12)
parser.add_argument('--sigma', '-s', type=float, default=100.0)
parser.add_argument('--rgbd_zscore',action='store_true',  help='4 channels')#train+test标准化
parser.add_argument('--rgbd_zscore_foronly_train_or_test_respectedly',action='store_true',  help='4 channels') #分别对train标准化和对test标准化
parser.add_argument('--rgbd_minmax',action='store_true',  help='4 channels')
parser.add_argument('--rgbd_after_check', action='store_true',  help='remained data after we check the dataset')
parser.add_argument('--rnn_layers', type=int, default=1)
parser.add_argument('--mixup',action='store_true',  help='data augmentation')
parser.add_argument('--use_detect_label',action='store_true',  help='data augmentation')
parser.add_argument('--use_detect_label_cutfeaturemap',action='store_true',  help='需要把transforms.CenterCrop((256,256))去除')

args = parser.parse_args()

set_seed(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  
start_epoch = 0  
global_step = 0 
print('==> Preparing data..')
print(f'learning rate:{args.lr}, weight decay: {args.wd}')
print('==> Building model..')
global net
import timm  

if args.model == 'RGBD_Fusion':
  
    print('==> Load checkpoint..')
    if args.rgbd:
        net = pvt.PolypPVT()
        net2 = pvt.PolypPVT()
        net_cat = myresnet.RGBD_Fusion()

    elif args.use_detect_label_cutfeaturemap:
        net = myresnet.resnet101(bbox = args.use_detect_label_cutfeaturemap)
    else:
        net = myresnet.resnet101()
elif args.model == 'resnet18':
    pretrained_dict = torchvision.models.resnet18(pretrained=True).state_dict()
    net = myresnet.resnet18()
elif args.model == 'resnet50':
    net = myresnet.resnet50()
elif args.model == 'inceptionv3':
    inceptionv3 = torchvision.models.inception_v3(pretrained=True)
    pretrained_dict = inceptionv3.state_dict()
    net = Inception3(aux_logits=False, transform_input = False, rgbd = args.rgbd)
    if args.rgbd:
        net2 = Inception3(aux_logits=False, transform_input = False, rgbd = args.rgbd)
        net_cat = Inception3_concat(args)

elif 'vit_base' in  args.model:
    '''vit_base_patch16_224_in21k  / vit_base_patch16_224''' 
    meta=dict()
    if args.model == 'vit_base_patch16_224_in21k':
        pretrainedvit = create_model("vit_base_patch16_224_in21k", pretrained = True) 
        net = vit.vit_base_patch16_224_in21k(pretrained=False) 
    elif args.model == 'vit_base_patch16_224':
        pretrainedvit = create_model("vit_base_patch16_224", pretrained = True) 
        net = vit.vit_base_patch16_224(pretrained=False)
    pretrained_dict = pretrainedvit.state_dict()
    in_feature = getattr(net, "head").in_features
    meta['in_feature'] = in_feature
    features = dict(list(net.named_children())[:-1])
    net = ViTNutrition(features, meta)
    
elif 'T2t_vit' in args.model:
    meta = {}
    meta['img_size'] = args.img_size
    meta['embed_dim'] = args.embed_dim
    net = create_model(args.model, pretrained=args.pretrained, drop_rate=args.drop, drop_connect_rate=args.drop_connect, drop_path_rate=args.drop_path,drop_block_rate=args.drop_block,global_pool=args.gp,bn_tf=args.bn_tf,
    bn_momentum=args.bn_momentum,bn_eps=args.bn_eps,checkpoint_path=args.initial_checkpoint,img_size=args.img_size)
    if args.transfer_learning:
        print('transfer learning, load t2t-vit pretrained model')
        load_for_transfer_learning(net, args.transfer_model, use_ema=True, strict=False, num_classes=args.num_classes)
    in_feature = getattr(net, "head").in_features 
    meta['in_feature'] = in_feature
    features = dict(list(net.named_children())[:-1]) 
    pretrained_dict = net.state_dict()
    net = T2TNutrition(features,meta)


model_dict = net.state_dict()
for name, param in model_dict.items():
    print("@@@@@@@@@@@@@@@@@@@@@@@@",name, param)

net = net.to(device)
print("@@@@@@@@@@@@@@@@@@@net",net)
if args.rgbd:
    net2 = net2.to(device)
    net_cat = net_cat.to(device)


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    if args.rgbd:
        net2 = torch.nn.DataParallel(net2)
        net_cat = torch.nn.DataParallel(net_cat)
    cudnn.benchmark = True

criterion = nn.L1Loss()
parameters = net.parameters()
optimizer = torch.optim.Adam([
        {'params': (p for name, p in net.named_parameters() if 'bias' not in name)},
        {'params': (p for name, p in net.named_parameters() if 'bias' in name), 'weight_decay': 0.}
    ], lr=1e-4, weight_decay=5e-4)

if args.rgbd:

    optimizer = torch.optim.Adam([
        {'params': net.module.parameters(),'lr':1e-4, 'weight_decay': 1e-5},
        {'params': net2.module.parameters(), 'lr':1e-4, 'weight_decay': 1e-5},
         {'params': net_cat.module.parameters(),'lr':1e-4, 'weight_decay': 1e-5}
         ]) 
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99) 


if args.resume:
    print('==> Resuming from checkpoint..')
    models_state_dict = torch.load(args.resume)
    new_state_dict_rgb = OrderedDict()
    for k, v in models_state_dict['net'].items():
        name = k[7:] if k.startswith('module') else 'module.' + k
        new_state_dict_rgb[name] = v
    net.load_state_dict(new_state_dict_rgb)
    net2.load_state_dict(models_state_dict['net_d'])
    net_cat.load_state_dict(models_state_dict['net_cat'])
    optimizer.load_state_dict(models_state_dict['optimizer'])
    start_epoch = models_state_dict['epoch']

weights = []
task_losses = []
loss_ratios = []
grad_norm_losses = []

trainloader, testloader = get_DataLoader(args)

def train(epoch,net):
    global global_step
    print('\nEpoch: %d' % epoch)
    net.train()
    if args.rgbd:
        net2.train()
        net_cat.train()
    train_loss = 0
    calories_loss = 0
    mass_loss = 0
    fat_loss = 0
    carb_loss = 0
    protein_loss = 0
    epoch_iterator = tqdm(trainloader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
    for batch_idx, x in enumerate(epoch_iterator):    #(inputs, targets,ingredient)
      
        inputs = x[0].to(device)
        total_calories = x[2].to(device).float()
        total_mass = x[3].to(device).float()
        total_fat = x[4].to(device).float()
        total_carb = x[5].to(device).float()
        total_protein = x[6].to(device).float()
        if args.rgbd:
            inputs_rgbd = x[7].to(device)
        if args.use_detect_label_cutfeaturemap:
            bbox = x[9]

        calories_per = total_calories/total_mass
        fat_per = total_fat/total_mass
        carb_per = total_carb/total_mass
        protein_per = total_protein/total_mass

        if args.mixup:
            if not args.rgbd:
                y = [total_calories, total_mass, total_fat, total_carb, total_protein]
                inputs, y_a, y_b, lam = mixup_data(inputs, y, alpha=0.2)
            else:
                pass
        optimizer.zero_grad() 

        if args.use_detect_label_cutfeaturemap:
            outputs = net(inputs, bbox)
        else:
            x1,x2,x3,x4 = net(inputs)
        if args.rgbd:
            if args.model == 'inceptionv3':
                h1,h2,h3,h4,h5 = outputs
                outputs_rgbd = net2(inputs_rgbd)
                d1,d2,d3,d4,d5 = outputs_rgbd
                outputs = net_cat([h1,h2,h3,h4,h5], [d1,d2,d3,d4,d5])
            elif args.model == 'RGBD_Fusion':
                F1,F2,F3,F4 = net2(inputs_rgbd)
                outputs = net_cat([x1,x2,x3,x4],[F1,F2,F3,F4])

        if args.portion_independent:
            calories_per_loss = criterion(outputs[0], calories_per)
            fat_per_loss = criterion(outputs[2], fat_per)
            carb_per_loss = criterion(outputs[3], carb_per)
            protein_per_loss = criterion(outputs[4], protein_per)
            loss = calories_per_loss + fat_per_loss + carb_per_loss + protein_per_loss
            loss.backward()
            optimizer.step()
            global_step += 1
            train_loss += loss.item()
            calories_loss += calories_per_loss.item()
            mass_loss = 0
            fat_loss += fat_per_loss.item()
            carb_loss += carb_per_loss.item()
            protein_loss += protein_per_loss.item()

        elif args.direct_prediction:
            if args.mixup:
                total_calories_loss =  total_calories.shape[0]* mixup_criterion(criterion, outputs[0], y_a[0], y_b[0], lam) / total_calories.sum().item()
                total_mass_loss =  total_mass.shape[0] * mixup_criterion(criterion, outputs[1], y_a[1], y_b[1], lam)  / total_mass.sum().item()
                total_fat_loss = total_fat.shape[0] *  mixup_criterion(criterion, outputs[2], y_a[2], y_b[2], lam) / total_fat.sum().item()
                total_carb_loss =  total_carb.shape[0] * mixup_criterion(criterion, outputs[3], y_a[3], y_b[3], lam) / total_carb.sum().item()
                total_protein_loss =  total_protein.shape[0] * mixup_criterion(criterion, outputs[4], y_a[4], y_b[4], lam)  / total_protein.sum().item()
            else:
                total_calories_loss = total_calories.shape[0]* criterion(outputs[0], total_calories)  / total_calories.sum().item() 
                total_mass_loss = total_calories.shape[0]* criterion(outputs[1], total_mass)  / total_mass.sum().item()
                total_fat_loss = total_calories.shape[0]* criterion(outputs[2], total_fat)  / total_fat.sum().item()
                total_carb_loss = total_calories.shape[0]* criterion(outputs[3], total_carb) / total_carb.sum().item()
                total_protein_loss = total_calories.shape[0]* criterion(outputs[4], total_protein)  / total_protein.sum().item()

            loss = total_calories_loss + total_mass_loss + total_fat_loss + total_carb_loss + total_protein_loss
            if not args.gradnorm:
                loss.backward()
            optimizer.step()
            global_step += 1

            train_loss += loss.item()
            calories_loss += total_calories_loss.item()
            mass_loss += total_mass_loss.item()
            fat_loss += total_fat_loss.item()
            carb_loss += total_carb_loss.item()
            protein_loss += total_protein_loss.item()


        if (batch_idx+1) % args.print_freq == 0 or batch_idx+1 == len(trainloader):
            logtxt(log_file_path, 'Epoch: [{}][{}/{}]\t'
                    'Loss: {:2.5f} \t'
                    'calorieloss: {:2.5f} \t'
                    'massloss: {:2.5f} \t'
                    'fatloss: {:2.5f} \t'
                    'carbloss: {:2.5f} \t'
                    'proteinloss: {:2.5f} \t'
                    'lr:{:.7f}'.format(
                    epoch, batch_idx+1, len(trainloader), 
                    train_loss/(batch_idx+1), 
                    calories_loss/(batch_idx+1),
                    mass_loss/(batch_idx+1),
                    fat_loss/(batch_idx+1),
                    carb_loss/(batch_idx+1),
                    protein_loss/(batch_idx+1),
                    optimizer.param_groups[0]['lr']))
    

best_loss = 10000
def test(epoch,net):
    global best_loss
    net.eval()
    if args.rgbd:
        net2.eval()
        net_cat.eval()
    test_loss = 0
    calories_loss = 0
    mass_loss = 0
    fat_loss = 0
    carb_loss = 0
    protein_loss = 0

    epoch_iterator = tqdm(testloader,
                          desc="Testing... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    csv_rows = []
    with torch.no_grad():
        for batch_idx, x in enumerate(epoch_iterator): 
            inputs = x[0].to(device)
            total_calories = x[2].to(device).float()
            total_mass = x[3].to(device).float()
            total_fat = x[4].to(device).float()
            total_carb = x[5].to(device).float()
            total_protein = x[6].to(device).float()
            if args.rgbd:
                inputs_rgbd = x[7].to(device)
            if args.use_detect_label_cutfeaturemap:
                bbox = x[9]

            calories_per = total_calories/total_mass
            fat_per = total_fat/total_mass
            carb_per = total_carb/total_mass
            protein_per = total_protein/total_mass

            optimizer.zero_grad()
            if args.use_detect_label_cutfeaturemap:
                outputs = net(inputs, bbox)
            else:
                x1,x2,x3,x4 = net(inputs)
                
            if args.rgbd:
                if args.model == 'inceptionv3':
                    h1,h2,h3,h4,h5 = outputs
                    outputs_rgbd = net2(inputs_rgbd)
                    d1,d2,d3,d4,d5 = outputs_rgbd
                    outputs = net_cat([h1,h2,h3,h4,h5], [d1,d2,d3,d4,d5])
                elif args.model == 'RGBD_Fusion':
                    F1,F2,F3,F4 = net2(inputs_rgbd)
                    outputs = net_cat([x1,x2,x3,x4],[F1,F2,F3,F4])

            #loss
            if args.portion_independent:
                calories_total_loss = criterion(outputs[0], calories_per)
                fat_total_loss = criterion(outputs[2], fat_per)
                carb_total_loss = criterion(outputs[3], carb_per)
                protein_total_loss = criterion(outputs[4], protein_per)
                loss = calories_total_loss + fat_total_loss + carb_total_loss + protein_total_loss
                if epoch % 1 ==0:

                    for i in range(len(x[1])):
                        dish_id = x[1][i]
                        calories = outputs[0][i] * total_mass[i]
                        mass =  total_mass[i]
                        fat = outputs[2][i] * total_mass[i]
                        carb = outputs[3][i] * total_mass[i]
                        protein = outputs[4][i] * total_mass[i]
                        dish_row = [dish_id, calories.item(), mass.item(), fat.item(), carb.item(), protein.item()]
                        csv_rows.append(dish_row)
                    
                # pdb.set_trace()
                test_loss += loss.item()
                calories_loss += calories_total_loss.item()
                mass_loss = 0
                fat_loss += fat_total_loss.item()
                carb_loss += carb_total_loss.item()
                protein_loss += protein_total_loss.item()
            
            elif args.direct_prediction:
                calories_total_loss = total_calories.shape[0]* criterion(outputs[0], total_calories) /total_calories.sum().item()
                mass_total_loss = total_calories.shape[0]* criterion(outputs[1], total_mass)  /total_mass.sum().item()
                fat_total_loss = total_calories.shape[0]* criterion(outputs[2], total_fat) /total_fat.sum().item()
                carb_total_loss = total_calories.shape[0]* criterion(outputs[3], total_carb) /total_carb.sum().item()
                protein_total_loss = total_calories.shape[0]* criterion(outputs[4], total_protein) /total_protein.sum().item()


                loss = calories_total_loss + mass_total_loss+ fat_total_loss + carb_total_loss + protein_total_loss

                if epoch % 1 ==0:
                    for i in range(len(x[1])):
                        dish_id = x[1][i]
                        calories = outputs[0][i]
                        mass =  outputs[1][i]
                        fat = outputs[2][i]
                        carb = outputs[3][i]
                        protein = outputs[4][i]
                        dish_row = [dish_id, calories.item(), mass.item(), fat.item(), carb.item(), protein.item()]
                        csv_rows.append(dish_row)

                test_loss += loss.item()
                calories_loss += calories_total_loss.item()
                mass_loss += mass_total_loss.item()
                fat_loss += fat_total_loss.item()
                carb_loss += carb_total_loss.item()
                protein_loss += protein_total_loss.item()


            epoch_iterator.set_description(
                    "Testing Epoch[%d] | loss=%2.5f | calorieloss=%2.5f | massloss=%2.5f| fatloss=%2.5f | carbloss=%2.5f | proteinloss=%2.5f | lr: %.5f" % (epoch, test_loss/(batch_idx+1), calories_loss/(batch_idx+1), mass_loss/(batch_idx+1), fat_loss/(batch_idx+1), carb_loss/(batch_idx+1),protein_loss/(batch_idx+1), optimizer.param_groups[0]['lr'])
                )
        print("!!!!!!!!!!len(testloader)",len(testloader))
        logtxt(log_file_path, 'Test Epoch: [{}][{}/{}]\t'
                    'Loss: {:2.5f} \t'
                    'calorieloss: {:2.5f} \t'
                    'massloss: {:2.5f} \t'
                    'fatloss: {:2.5f} \t'
                    'carbloss: {:2.5f} \t'
                    'proteinloss: {:2.5f} \t'
                    'lr:{:.7f}\n'.format(
                    epoch, batch_idx+1, len(testloader), 
                    test_loss/len(testloader), 
                    calories_loss/len(testloader),
                    mass_loss/len(testloader),
                    fat_loss/len(testloader),
                    carb_loss/len(testloader),
                    protein_loss/len(testloader),
                    optimizer.param_groups[0]['lr']))

    if best_loss > test_loss:
        best_loss = test_loss
        print('Saving..')
        net = net.module if hasattr(net, 'module') else net
        state = {
            'net': net.state_dict(),
            'net_d' : net2.state_dict(),
            'net_cat' : net_cat.state_dict(),
            'optimizer':optimizer.state_dict(),
            'epoch': epoch
        }
        savepath = f"./saved/new/regression_{args.dataset}_{args.model}_{args.run_name}"
        check_dirs(savepath)
        torch.save(state, os.path.join(savepath,f"ckpt_RGBD.pth"))

        
    if epoch % 1 == 0:
        new_csv_rows = []
        predict_values = dict()
        # pdb.set_trace()
        key = ''
        for iterator in csv_rows:
            if key != iterator[0]:
                key = iterator[0]
                predict_values[key] = []
                predict_values[key].append(iterator[1:])
            else:
                predict_values[key].append(iterator[1:])
        # pdb.set_trace()
        for k,v in predict_values.items():
            nparray = np.array(v)
            predict_values[k] = np.mean(nparray,axis=0) #每列求均值
            new_csv_rows.append([k, predict_values[k][0], predict_values[k][1], predict_values[k][2], predict_values[k][3], predict_values[k][4]])

        headers = ["dish_id", "calories", "mass", "fat", "carb", "protein"]
        csv_file_path = os.path.join("logs_nutrition2",f'checkpoint_{args.dataset}_{args.model}_{args.run_name}',"epoch{}_result_image.csv".format(epoch))



def get_grad_norm_losses(gradNormModel, task_loss,initial_task_loss):
    weighted_task_loss = torch.mul(gradNormModel.weights, task_loss)
    loss = torch.sum(weighted_task_loss)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    gradNormModel.weights.grad.data = gradNormModel.weights.grad.data * 0.0
    if hasattr(net, 'fc2'):
        W = net.get_last_shared_layer()
    else:
        W = net.module.get_last_shared_layer()
    norms = []
    for i in range(len(task_loss)):
        gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
        norms.append(torch.norm(torch.mul(gradNormModel.weights[i], gygw[0])))
    norms = torch.stack(norms)
    if torch.cuda.is_available():
        loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
    else:
        loss_ratio = task_loss.data.numpy() / initial_task_loss
    # r_i(t)
    inverse_train_rate = loss_ratio / np.mean(loss_ratio)

    if torch.cuda.is_available():
        mean_norm = np.mean(norms.data.cpu().numpy())
    else:
        mean_norm = np.mean(norms.data.numpy())

    constant_term = torch.tensor(mean_norm * (inverse_train_rate ** args.alpha), requires_grad=False)
    if torch.cuda.is_available():
        constant_term = constant_term.cuda()
    grad_norm_loss = torch.as_tensor(torch.sum(torch.abs(norms - constant_term)))
    gradNormModel.weights.grad = torch.autograd.grad(grad_norm_loss, gradNormModel.weights)[0]
    return grad_norm_loss


def mixup_data(x, y, alpha=0.2, use_cuda=True):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index,:] 
    y_a = y
    y_b = []
    for i in range(len(y)):
        y_b.append(y[i][index])
    return mixed_x, y_a, y_b, lam
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return  lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


log_file_path = os.path.join("/Path/logs/new",f'checkpoint_{args.dataset}_{args.model}_{args.run_name}',"train_log.txt")
check_dirs(os.path.join("/Path/logs/new",f'checkpoint_{args.dataset}_{args.model}_{args.run_name}'))
logtxt(log_file_path, str(vars(args)))
for epoch in range(start_epoch, start_epoch+201):
    train(epoch,net)
    test(epoch,net)
    scheduler.step()




