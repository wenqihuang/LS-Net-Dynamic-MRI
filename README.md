# Deep Low-rank plus Sparse Network (L+S-Net) for Dynamic MR Imaging

This repository provides a tensorflow implementation used in our publication

**Huang, Wenqi, et al., [*Deep low-Rank plus sparse network for dynamic MR imaging.*](https://www.sciencedirect.com/science/article/abs/pii/S136184152100236X), Medical Image Analysis 73 (2021): 102190.**


 If you use this code and provided data, please refer to:

```
@article{huang2021deep,
  title={Deep low-Rank plus sparse network for dynamic MR imaging},
  author={Huang, Wenqi and Ke, Ziwen and Cui, Zhuo-Xu and Cheng, Jing and Qiu, Zhilang and Jia, Sen and Ying, Leslie and Zhu, Yanjie and Liang, Dong},
  journal={Medical Image Analysis},
  volume={73},
  pages={102190},
  year={2021},
  publisher={Elsevier}
}
```

## Requirements

The framework was build on Tensorflow 2.4 and roughly tested on Tensorflow 2.7.

We provide an requirements file requirements.txt. You can create a new conda environment or virtual environment and execute install the requirements using the following command.

```
pip install -r requirements.txt
```
If you want to visualize the dynamic images in gif in tensorboard, please install `ffmpeg` in advance. On Ubuntu, it can be simply installed by
```
sudo apt install ffmpeg
```
Training the L+S-Net usually needs large GPU memory. If you come across the OOM problem, please try to reduce the `niter` and `n_f` of the network, or reduce the number of coils by coil compression.


## Data
We provided a [jupyter notebook](https://github.com/wenqihuang/LS-Net-Dynamic-MRI/blob/main/create_dummy_data/create_dummy_data.ipynb) for creating a dummy dataset. The corresponding dataloader is also provided in `dataset.py`. New dataset can be easily made by modifying the demo code.


## Training
After creating the dummy dataset, the network can be easily traind by
```
python main.py --gpu=0 --data='DUMMY' --niter=10 --num_epoch=50
```
in which `--gpu` specifies the GPU for training, `--data` specifies the dataset used for training, `--niter` is for number of iteration blocks and `--num_epoch` specifies the training epoches.

Please refer to `main.py` for more configurations.

You can oberserve the progress of the training in Tensorboard using the specified log directory. The command is as follow
```
tensorboard --logdir=./logs
```
Model files will be stored in `./models`


## Testing
To test a trained model with the name <MODEL_ID>

```
python test.py [SAME_PARAMS_AS_TRAINIGN] --weight=models/<MODEL_ID>/epoch-50/ckpt
```
Please note that do not omit the `ckpt` at the end of the weight path.