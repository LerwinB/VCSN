import re
import csv
import os
def parse_log(logpath):
    """
    解析日志文件并提取实验结果
    :param logpath: 日志文件路径
    :return: 提取的实验结果列表
    """
    results = []
    mIoUs=[logpath.split('/')[1]]
    # 定义正则表达式模式来匹配实验结果
    pattern = re.compile(r'Iter\(val\) \[\d+/\d+\]\s+aAcc: ([\d.]+)\s+mIoU: ([\d.]+)\s+mAcc: ([\d.]+)')
    
    with open(logpath, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                aAcc = float(match.group(1))
                mIoU = float(match.group(2))
                mAcc = float(match.group(3))
                results.append([aAcc, mIoU, mAcc])
                mIoUs.append(mIoU)
    print(mIoUs)
    return results,mIoUs

def save_to_csv(results, csvpath):
    """
    将实验结果保存到 CSV 文件
    :param results: 实验结果列表
    :param csvpath: CSV 文件路径
    """
    with open(csvpath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['aAcc', 'mIoU', 'mAcc'])
        writer.writerows(results)

def get_all_log(work_dir):
    alllogs=[]
    for dirpath, dirnames, filenames in os.walk(work_dir):
        if 'epoch_5.pth' in filenames:
            # 
            dirnames.sort()
            logpath= os.path.join(dirpath,dirnames[-1],dirnames[-1]+'.log')
            taskname=dirpath.split('/')[-1]
            alllogs.append([taskname,logpath])
    with open('csvresult/all_logs_path.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['task_name', 'log_path'])
        writer.writerows(alllogs)
    return alllogs



# 示例用法

alllogs=get_all_log('work_dirs')
mious=[]
# 解析日志文件并提取实验结果
for log in alllogs:

    results, miou = parse_log(log[-1])
    csvpath = os.path.join('csvresult',log[0]+'.csv')
# 将实验结果保存到 CSV 文件
    save_to_csv(results, csvpath)
    mious.append(miou)


    print(f"实验结果已保存到 {csvpath}")
with open('csvresult/all_miou.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['taskname', 'e1', 'e2','e3','e4','e5'])
    writer.writerows(mious)