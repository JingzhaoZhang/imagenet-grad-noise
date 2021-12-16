# imagenet-grad-noise

This code was adapted from https://github.com/jiweibo/ImageNet

It produces the ImageNet experiments in https://arxiv.org/pdf/2110.06256.pdf




## PyTorch

- Use getdata.sh to download wiki103

- Example usage:
```

python train.py  PATH_TO_IMAGENET -save_noise -save_sharpness --save-dir 'resnet' --pretrain_path '/com_space/tengjiaye/'  -j 16 -noise_size 200 -sharpness_batches 200 -b 64


```
The above describes one particular plot for the constant step size training.
The stats needs to be smoothed to avoid periodicity in text. 

## Brief descriptions


noise and grad are computed in a straight forward way as described in https://arxiv.org/abs/2110.06256
Sharpness is computed using power method from https://github.com/leiwu0/sgd.stability
Output can be found in '/home/zhangjingzhao/logs/imagenet_stat/'


- save-dir - str, name of exp
- save_noise - bool, whether to compute and store noise/ grad norm
- noise_size - int, number of batch used to compute the full grad (sample number = noise_size x batch_size)
- save_sharpness - bool, whether to compute and store sharpness
- sharpness_batches - int, number of batch used to compute the sharpness (sample number = noise_size x batch_size)
- epoch_interval - int, number of train epcohs per computation of noise and sharpness


