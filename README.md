# imagenet-grad-noise

This code was adapted from https://github.com/jiweibo/ImageNet

It produces the ImageNet experiments in https://arxiv.org/pdf/2110.06256.pdf




## PyTorch

- Use getdata.sh to download wiki103

- Example usage:
```

python train.py  /mnt/sda1/hehaowei/ImageNet/ -save_noise -save_sharpness --save-dir '/com_space/jingzhao/logs/imagenet_kj/imagenet-res34-nsize500-sf7000-0110' -j 8 -ls constant --lr 0.01 -noise_size 500 -sharpness_batches 500 -b 250 -sf 7000

python train.py  /mnt/sda1/hehaowei/ImageNet/ -save_noise -save_sharpness --save-dir '/com_space/jingzhao/logs/imagenet_kj/imagenet-res101-nsize1000-sf70000-0112' -j 8  -noise_size 1000 -sharpness_batches 10 -b 128 -sf 70000

python train.py  /mnt/sda1/hehaowei/ImageNet/ --save-dir '/com_space/jingzhao/logs/imagenet_kj/imagenet-res101-smooth-train-0124' -j 16 -b 256


python train.py  /home/zhangjingzhao/ImageNet/ImageNet/ -save_noise -save_sharpness --save-dir '/com_space/jingzhao/logs/imagenet_kj/imagenet-res34-nsize600-sf7000-0115-cst0001' -j 16 -ls constant --lr 0.001 -noise_size 600 -sharpness_batches 600 -b 250 -sf 7000

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


