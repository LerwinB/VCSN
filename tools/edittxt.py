import os
data_path='data/coco164k'
for j in [5,10,15,20]:
    with open(os.path.join(data_path, f"COCO_sam{j}_VAE3_kmeans_selected.txt"), "r") as file:
        names=file.read()
    jpg_list = names.split('.jpg')

    with open(os.path.join(data_path, f"COCO_sam_VAE3_kmeans{j}.txt"), 'w') as file:
        for img in jpg_list:
            file.write(img + '\n') 
