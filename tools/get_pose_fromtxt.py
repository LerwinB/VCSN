import os
import csv
import pandas as pd
# 指定要提取的文件名列表
data_root='data/UESAT_RGB_53/MMdata'
def transcsv(csv_file_path,out_path):
    df = pd.read_csv(csv_file_path)
    # 按照第一列（filename）筛选
    # filtered_df = df[df['filename'].isin(target_filenames)]
    filtered_df_no_filename = df.drop(columns=['filename'])

    # 保存到新文件，不包含第一列   
    filtered_csv_path = os.path.join(data_root,out_path)

    df['filename'].to_csv(filtered_csv_path.replace('_gt.csv','.txt'), index=False, header=False)
    filtered_df_no_filename.to_csv(filtered_csv_path, index=False)
    print(out_path)

# transcsv(csv_file_path='tools/pose_minival_gt.csv',out_path='Uesat_minival_pose_gt.csv')
# for i in [5,10,15,20]:
#     with open(os.path.join(data_root,f'sam{i}_VAE3_kmeans.txt'),'r') as f:
#         paths=f.readlines()
#     target_filenames = [path.rstrip('\n') for path in paths]


#     # 打开 CSV 文件并读取数据
#     csv_file_path = 'tools/pose_all_gt.csv'
#     transcsv(csv_file_path=csv_file_path,out_path=f'sam{i}_VAE3_kmeans_pose_gt.csv')


tasks=['uesat_Random5','resnet5_kmeans','ULAL5','Entropy5']
for task in tasks: 
    with open(os.path.join(data_root,f'{task}.txt'),'r') as f:
        paths=f.readlines()
    target_filenames = [path.rstrip('\n') for path in paths]
    # 打开 CSV 文件并读取数据
    csv_file_path = 'tools/pose_all_gt.csv'
    transcsv(csv_file_path=csv_file_path,out_path=f'{task}_pose_gt.csv')