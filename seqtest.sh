source ~/anaconda3/bin/activate openmmlab;
export CUDA_VISIBLE_DEVICES=1;

for i in 1 2 3 4;do
    python MMtest.py  configs/deeplabv3plus/deeplabv3plus_r50_uesat10.py work_dirs/deeplabv3plus_r50_uesat10_VAEn_30/epoch_${i}.pth --work-dir work_dirs/deeplabv3plus_r50_uesat10_VAEn_30_allval;
done
# python MMtest.py configs/deeplabv3plus/deeplabv3plus_r50_HILrgb.py work_dirs/deeplabv3plus_r50_sam5_VAE3_e20/epoch_20.pth --work-dir work_dirs/deeplabv3plus_r50_HIL_VAE3_e20_val;
# python MMtest.py configs/deeplabv3plus/deeplabv3plus_r50_HILrgb.py work_dirs/deeplabv3plus_r50_Random5_uesatrgb/epoch_5.pth --work-dir work_dirs/deeplabv3plus_r50_HIL_random_val;
# python MMtest.py configs/deeplabv3plus/deeplabv3plus_r50_HILrgb.py work_dirs/deeplabv3plus_r50_full_uesatrgb/epoch_5.pth --work-dir work_dirs/deeplabv3plus_r50_HIL_uefull_val
# for i in 5 10;do
#     python MMtest.py configs/deeplabv3plus/deeplabv3plus_r50_SAM5_uesatrgb.py work_dirs/ULAL${i}/epoch_5.pth --work-dir work_dirs/ULAL${i}_allval;
# done
# for i in 10 15 20;do
#     python MMtest.py configs/deeplabv3plus/deeplabv3plus_r50_SAM5_uesatrgb.py work_dirs/deeplabv3plus_r50_SAM${i}_kmeans_VAE3/epoch_5.pth --work-dir work_dirs/deeplabv3plus_r50_SAM${i}_kmeans_VAE3_allval;
#     python MMtest.py configs/deeplabv3plus/deeplabv3plus_r50_SAM5_uesatrgb.py work_dirs/deeplabv3plus_r50_SAM${i}_kmeans_noUmap/epoch_5.pth --work-dir work_dirs/deeplabv3plus_r50_SAM${i}_kmeans_noUmap_allval;

# done
# for i in 5 10 15 20;do
#     python MMtest.py configs/deeplabv3plus/deeplabv3plus_r50_SAM5_uesatrgb.py work_dirs/deeplabv3plus_r50_SAM${i}_kmeans1_noUmap/epoch_5.pth --work-dir work_dirs/deeplabv3plus_r50_SAM${i}_kmeans1_noUmap_allval;
#     #python MMtest.py configs/deeplabv3plus/deeplabv3plus_r50_SAM5_uesatrgb.py work_dirs/deeplabv3plus_r50_SAM${i}_kmeans_noUmap/epoch_5.pth --work-dir work_dirs/deeplabv3plus_r50_SAM${i}_kmeans_noUmap_allval;
# done

# MMtest.py configs/deeplabv3plus/deeplabv3plus_r50_SAM5_uesatrgb.py work_dirs/deeplabv3plus_r50_Random${i}_uesatrgb/epoch_5.pth --work-dir work_dirs/deeplabv3plus_r50_Random${i}_uesatrgb_val;

# python MMtest.py configs/deeplabv3plus/deeplabv3plus_r50_SAM5_uesatrgb.py work_dirs/deeplabv3plus_r50_SAM5_kmeans_VAE3/epoch_5.pth --work-dir work_dirs/deeplabv3plus_r50_SAM5_kmeans_VAE3_val