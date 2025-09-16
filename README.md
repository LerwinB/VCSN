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
This dataset is distributed under controlled access. Researchers must first request permission by emailing [syzhao@buaa.edu.cn] with their full name, institution, and intended research purpose. If approved, a unique verification code will be provided, which is required before downloading the dataset. Redistribution of the dataset is prohibited.

### usage


