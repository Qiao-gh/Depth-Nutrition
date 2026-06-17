import torch
from torchvision import models, transforms
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils import visualize_cam, target_category
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# ################################################
# ################################################
from thop import profile
from thop import clever_format
from ptflops import get_model_complexity_info
import cv2
import numpy as np

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients
        # self.gradients.append(grad.cpu().detach())

    def __call__(self, x,x_d):
        self.gradients = []
        self.activations = []
        return self.model(x,x_d)

    def release(self):
        for handle in self.handles:
            handle.remove()

class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        # 确保target_layers是列表
        if not isinstance(target_layers, list):
            target_layers = [target_layers]
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        loss = output[target_category]
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)
        # n,h,w
        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            # scaled (N, H, W)
            scaled = self.scale_cam_image(cam, target_size)
            # scaled[:, None, :] (N, 1, H, W)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        # cam_per_target_layer.shape -> (N, 4, H, W)
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        # result.shape -> (N, H, W)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, inputD_tensor, target_category):

        if self.cuda:
            input_tensor = input_tensor.cuda()
            inputD_tensor = inputD_tensor.cuda()

        # 正向传播得到网络输出logits(未经过softmax)
        output = self.activations_and_grads(input_tensor,inputD_tensor)
        # if isinstance(target_category, int):
        #     target_category = [target_category] * input_tensor.size(0)
        #
        # if target_category is None:
        #     target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
        #     print(f"category id: {target_category}")
        # else:
        #     assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = output[target_category]
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    # img = cv2.resize(img, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

###########################################################
from models import myresnet,pvt
from collections import OrderedDict
net_rgb = pvt.PolypPVT()
net_rgbd = pvt.PolypPVT()
net_cat = myresnet.Resnet101_concat()
print(net_cat)
###############################

checkpoint_path = '/Path/saved/new/regression_nutrition_rgbd_resnet101_editname/ckpt_best.pth'
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

class RGBD(nn.Module):
    def __init__(self, channel=32):
        super(RGBD, self).__init__()

        self.net_rgb = net_rgb
        self.net_rgbd = net_rgbd
        self.net_cat = net_cat
               
    def forward(self, x,x_d):
        x1,x2,x3,x4 = self.net_rgb(x)
        x_d1,x_d2,x_d3,x_d4 = self.net_rgbd(x_d)
        outputs = net_cat([x1,x2,x3,x4],[x_d1,x_d2,x_d3,x_d4])

        return outputs

# 实例化模型并加载权重（如果有）
model = RGBD()

# 选择要可视化的任务的输出层
task1_target_layer = model.net_cat.smooth4
# task1_target_layer = model.net_rgb.backbone.block3[17].mlp.dwconv.dwconv

# 准备图像
img_path = '/Path/dish_1559060106/rgb.png'
img_D = '/Path/dish_1559060106/depth_color.png'
img = Image.open(img_path).convert('RGB')
imgD = Image.open(img_D).convert('RGB')
#############################################################
def reshape_transform(tensor, height=7, width=7):
    # 去掉cls token
    result = tensor[:, 1:, :].reshape(tensor.size(0),
    height, width, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result
###############################################
transform = transforms.Compose([
    transforms.Resize((238, 238)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0)
inputD_tensor = transform(imgD).unsqueeze(0)

################################
# 初始化 Grad-CAM
grad_cam_task1 = GradCAM(model, target_layers=task1_target_layer,reshape_transform=None)

target_categorys = [0, 2, 3, 4]

for target_category in target_categorys:
    img = Image.open(img_path).convert('RGB')
    imgD = Image.open(img_D).convert('RGB')

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0)
    inputD_tensor = transform(imgD).unsqueeze(0)
    # 可视化
    grayscale_cam_task1 = grad_cam_task1(input_tensor, inputD_tensor, target_category=target_category)
    
    # 获取可视化的灰度图
    grayscale_cam = grayscale_cam_task1[0, :]
    
    # 将图像转换为[0, 1]范围的float32类型
    img = np.array(img).astype(np.float32) / 255.0
    
    # 在图像上显示CAM
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    
    # 保存可视化结果
    plt.imshow(visualization)
    plt.axis('off')
    plt.savefig(f'/Path/dish_1559060106_{target_category}.png', bbox_inches='tight', pad_inches=0)
    plt.show()


