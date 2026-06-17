import logging
from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision.transforms.transforms import CenterCrop
from mydataset import Food,CUB, ImagesForMulCls, Nutrition, Nutrition_RGBD
import pdb

def get_DataLoader(args):
    if args.dataset == 'food101':
        args.num_classes = 101
        train_transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=0.5), # default value is 0.5
                        transforms.Resize((256, 256)), #   600 in TransFG
                        transforms.RandomCrop((224,224)), #   448 in TransFG
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
        # transforms of test dataset
        test_transform = transforms.Compose([
                            transforms.Resize((256, 256)), 
                            transforms.CenterCrop((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])

        food_image_path = os.path.join(args.data_root,'images') #args.data_root : f'{args.data_root}/{args.dataset}'
        food_train_txt = os.path.join(args.data_root,'retrieval_dict','train_full.txt')
        food_test_txt = os.path.join(args.data_root,'retrieval_dict','test_full.txt')

        if args.multi_task: # 类别和食材多任务多标签
            # pdb.set_trace()
            image_size = 224
            ingredient_train_txt = os.path.join(args.data_root,'retrieval_dict','multi_label_food101','train_ingredient.txt')
            ingredient_test_txt = os.path.join(args.data_root,'retrieval_dict','multi_label_food101','test_ingredient.txt')
            #food101 类别标签 和 食材标签 不在同一个txt文件， food172是同一个文件
            trainset = ImagesForMulCls(args, food_image_path,food_train_txt,ingredient_train_txt,image_size,transform=train_transform)
            testset = ImagesForMulCls(args, food_image_path,food_test_txt,ingredient_test_txt,image_size,transform=test_transform)
        else: #单任务分类，单标签
            trainset = Food(txt_dir = food_train_txt, image_path = food_image_path, transform = train_transform)
            testset = Food(txt_dir = food_test_txt, image_path = food_image_path, transform = test_transform)

    elif args.dataset == 'food172':
        print('food172参考/home/isia/lmj/20210303/Experiment/code_new_wzlmm/dataset172.py 写法；\
        图片路径、类别标签、食材都在/home/isia/lmj/20210303/Experiment/data/food172/retrieval_dict/train_full_processed.txt 中')

    elif args.dataset == 'cub200/CUB_200_2011':
        train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = CUB(root=args.data_root, is_train=True, transform=train_transform)
        testset = CUB(root=args.data_root, is_train=False, transform = test_transform)

    elif args.dataset == 'nutrition_rgb':

        train_transform = transforms.Compose([
                                    # transforms.RandomRotation(degrees=(0, 180)),
                                    transforms.Resize((270, 480), Image.BILINEAR),
                                    # transforms.RandomHorizontalFlip(),
                                    transforms.CenterCrop((256,256)), #256
                                    # transforms.ColorJitter(hue=0.05),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        test_transform = transforms.Compose([
                                    transforms.Resize((270, 480), Image.BILINEAR),
                                    transforms.CenterCrop((256, 256)), #256
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        if 'T2t_vit' in args.model or 'vit' in  args.model: # imagesize = 224* 224
            pdb.set_trace()
            print(args.model)
            train_transform = transforms.Compose([
                                    transforms.Resize((270, 480)),
                                    transforms.CenterCrop((256,256)),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            test_transform = transforms.Compose([
                                    transforms.Resize((270, 480)),
                                    transforms.CenterCrop((256, 256)),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        nutrition_rgb_ims_root = os.path.join(args.data_root, 'imagery')
        if args.rgbd_after_check:
            # pdb.set_trace()
            nutrition_train_txt = os.path.join(args.data_root, 'imagery','rgb_train_processed_tianhao.txt')
            nutrition_test_txt = os.path.join(args.data_root, 'imagery','rgb_test_processed_tianhao.txt')
        else:
            nutrition_train_txt = os.path.join(args.data_root, 'imagery','rgb_train_processed.txt')
            nutrition_test_txt = os.path.join(args.data_root, 'imagery','rgb_test_processed.txt')
        trainset = Nutrition(image_path = nutrition_rgb_ims_root, txt_dir = nutrition_train_txt, transform = train_transform)
        testset = Nutrition(image_path = nutrition_rgb_ims_root, txt_dir = nutrition_test_txt, transform = test_transform)

    elif args.dataset == 'nutrition_rgbd':
        train_transform = transforms.Compose([
                                    # transforms.RandomRotation(degrees=(0, 180)),
                                    transforms.Resize((320, 448)),
                                    # transforms.RandomHorizontalFlip(),
                                    # transforms.CenterCrop((256,256)),
                                    # transforms.ColorJitter(hue=0.05),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    # transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
                                    ])
        test_transform = transforms.Compose([
                                    transforms.Resize((320, 448)),
                                    # transforms.CenterCrop((256, 256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    # transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
                                    ])

        nutrition_rgbd_ims_root = os.path.join(args.data_root, 'imagery')
        nutrition_train_txt = os.path.join(args.data_root, 'imagery','rgbd_train_processed.txt')
        nutrition_test_txt = os.path.join(args.data_root, 'imagery','rgbd_test_processed.txt') # depth_color.png
        nutrition_train_rgbd_txt = os.path.join(args.data_root, 'imagery','rgb_in_overhead_train_processed.txt')
        nutrition_test_rgbd_txt = os.path.join(args.data_root, 'imagery','rgb_in_overhead_test_processed.txt') # rbg.png
        trainset = Nutrition_RGBD(nutrition_rgbd_ims_root, nutrition_train_rgbd_txt, nutrition_train_txt, transform = train_transform)
        testset = Nutrition_RGBD(nutrition_rgbd_ims_root, nutrition_test_rgbd_txt, nutrition_test_txt, transform = test_transform)



    train_loader = DataLoader(trainset,
                              batch_size=args.b,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True
                              )
    test_loader = DataLoader(testset,
                             batch_size=args.b,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True
                             ) 

    return train_loader, test_loader



