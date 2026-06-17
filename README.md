# Depth-Nutrition
**Depth-Nutrition: A Novel Food Nutrition Prediction Framework Guided by Deconstruction of Physical Features**

✨ This repository provides the official code for our food nutrition estimation framework. The model predicts food nutrition attributes from visual inputs and is designed for reproducible training, testing, and further research.

# Setup instructions 🛠️

## Environment configuration

We recommend the following environment:

python==3.8.13
torch==1.12.1+cu113
torchvision==0.13.1+cu113

```bash
# Example conda setup:
conda create -n depth-nutrition python=3.8.13 -y
conda activate depth-nutrition

# Install the remaining dependencies
pip install -r requirements.txt
```

## Pretrained weights
Download the pre-trained weight file of [PVT-V2-B3](https://github.com/whai362/PVT/tree/v2/classification) and place it in the project folder `checkpoints/`.

# Dataset 📦

Download the datasets [Nutrition5K](https://github.com/whai362/PVT/tree/v2/classification) and [MetaFood3D](https://github.com/whai362/PVT/tree/v2/classification).

The methods for dividing the training and test sets are specified in the path `/Nutrition5k/imagery/realsense_overhead` and `/MatFood3D`.

The reading and processing of the two datasets (Nutrition5K and MetaFood3D) are carried out in the file `mydataset.py` and in the `mydataset_METAFOOD3D.py` respectively.

# Train 🚀
After preparing as detailed above, you can start a new training job with the following command line flag:
```bash
nohup sh train.sh
```
# Test 🔍
We have provided the test script in the `test_RGBD_multi_fusion.py` file. You can run the following command line for testing:
```bash
python test_RGBD_multi_fusion.py
```

# Checkpoints 📂

We release our public model checkpoints and results here: [Depth-Nutrition](https://pan.baidu.com/s/1k_iwU1YFuyaVgk0Godad9A?pwd=jpei)

**Note on the released checkpoint:** "Due to the incomplete preservation of the original model parameter file used in the paper, the checkpoint in this repository is a replacement version retrained according to the training configuration described in the paper. This model performs close to the reported results on the main evaluation metrics and can be used for code validation, experiment reproduction, and follow-up research. If the original checkpoint is found later, we will update this repository promptly."

### Results table

| Method | Calories | Mass | Fat | Carb. | Protein | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Paper result | `13.27` | `9.94` | `20.32` | `20.43` | `19.76` | `16.74` |
| Released checkpoint | `13.31` | `9.95` | `20.55` | `20.76` | `18.93` | `16.70` |

---

# Citation 📚

If you use this code or our checkpoint in your research, please cite:
```bash
@article{qiao2026depth,
  title={Depth-Nutrition: A Novel Food Nutrition Prediction Framework Guided by Deconstruction of Physical Features},
  author={Qiao, Guanhua and Cheng, Chunyang and Shen, Zhongwei and Zhu, Jinlin and Li, Hui and Wu, Xiaojun},
  journal={IEEE Transactions on Multimedia},
  year={2026},
  publisher={IEEE}
}
```
