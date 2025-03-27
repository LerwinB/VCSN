# VCSN

[[Github]]( https://github.com/LerwinB/VCSN) [[Paper]]()

## VCSN optimization method
## install

## usage

1.  `sam_active_select.py` MDOM method for select coreset
```
python sam_active_select.py 
```
2.  `VAE-train.py`: train for VAE-DR algrithm
3.  `MMtrain.py`: MMsegmentation training file used for test MDOM 
4.  `MMtest.py`: MMsegmentation evaluation 
```
python MMtest.py configs/deeplabv3plus/deeplabv3plus_r50_HILrgb.py work_dirs/deeplabv3plus_r50_SAM5_kmeans_VAE3/epoch_5.pth --work-dir work_dirs/deeplabv3plus_r50_HIL_VAE3
```
## AAMSD Dataset
### Download
[[Baidu Drive]](https://pan.baidu.com/s/1dqL5CJ2b7mIzGR_w6I4eag?pwd=wqaf )
### usage


