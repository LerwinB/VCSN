import re
import numpy as np
import pandas as pd
import os

# 训练日志内容


# 正则表达式提取IoU信息
pattern0 = r'\+---------------\+-------\+-------\+([\s\S]+?)\+---------------\+-------\+-------\+([\s\S]+?)\+---------------\+-------\+-------\+'

# pattern0 = r'\+---------------\+-------\+-------\+([\s\S]+?)\+---------------\+-------\+-------\+'
pattern1 = r'\|\s+([a-z\s]+)\s+\|\s+([\d\.nan]+)\s+\|\s+([\d\.nan]+)\s+\|'
pattern = r'\+---------------\+-------\+-------\+([\s\S]+?)\+---------------\+-------\+-------\+'




def extract_iou(log_content,taskname):
    tables = re.findall(pattern0, log_content)
    if len(tables)==5:
        table = tables[-1]
    else:
        table = tables[0]
    if table:
        last_table = table[-1]

        matches = re.findall(pattern1, last_table, re.IGNORECASE)
        data = {}
        data['task'] = taskname
        iou_values = []
        for class_name, iou_str, acc_str in matches:
            try:
                iou = float(iou_str)
                if not np.isnan(iou) and iou > 0:  # 排除无效类别
                    data[class_name.strip()] = iou
                    iou_values.append(iou)
            except ValueError:
                continue

        mIoU = np.mean(iou_values) if iou_values else 0
        data['mIoU'] = mIoU
        return data
    else:
        print('No valid IoU values found in the log content.')
        return None
    
def get_all_log(work_dir):
    alllogs=[]
    for dirpath, dirnames, filenames in os.walk(work_dir):
        if 'epoch_5.pth' in filenames:
            # 
            dirnames.sort()
            for dirname in reversed(dirnames):
                logpath= os.path.join(dirpath,dirname,dirname+'.log')
                log_content = open(logpath).read()
                tables = re.findall(pattern0, log_content)
                if len(tables)==5 or len(tables)==1:
                    taskname=dirpath.split('/')[-1]
                    alllogs.append([taskname,logpath])
                    break
    
    return alllogs
all_data = pd.DataFrame()
# 提取有效类别的IoU值
logs=get_all_log('work_dirs')
for task,log in logs:
    log_c=open(log).read()
    data = extract_iou(log_c,task)
    if data:
        df_row = pd.DataFrame([data])
        print(data)
        all_data = pd.concat([all_data, df_row], ignore_index=True)


csv_path = 'iou_values.csv'
all_data.to_csv(csv_path, index=True)
print(f"有效类别的IoU已保存到 {csv_path}")