import re
import numpy as np
import pandas as pd

# 假设这是从文件读取的日志内容
log_content = """
2024/11/02 02:29:28 - mmengine - INFO -  
+---------------+-------+-------+
|     Class     |  IoU  |  Acc  |
+---------------+-------+-------+
|   background  | 99.85 | 99.94 |
|     panel     | 92.92 | 96.46 |
|     aerial    | 87.93 | 93.15 |
|     spout     |  61.5 | 69.46 |
|     camera    | 88.87 | 93.77 |
| panel support |  66.0 | 71.43 |
|  star sensor  | 41.85 | 45.93 |
|    docking    | 86.28 | 89.96 |
|      body     | 93.45 | 97.58 |
|      arm      |  nan  |  nan  |
|      part     | 76.43 |  84.6 |
|   rodaerial   |  0.0  |  0.0  |
| unkown aerial |  nan  |  nan  |
|     others    | 60.46 | 67.95 |
+---------------+-------+-------+
2024/11/03 03:30:29 - mmengine - INFO -  
dfsafjls
dsfafas
dsafasf

+---------------+-------+-------+
|     Class     |  IoU  |  Acc  |
+---------------+-------+-------+
|   background  | 99.88 | 99.96 |
|     panel     | 93.01 | 96.51 |
|     aerial    | 88.02 | 93.25 |
|     spout     |  62.0 | 70.00 |
|     camera    | 89.10 | 94.00 |
| panel support |  67.0 | 72.00 |
|  star sensor  | 42.00 | 46.50 |
|    docking    | 87.00 | 90.50 |
|      body     | 94.00 | 98.00 |
|      arm      |  nan  |  nan  |
|      part     | 77.00 | 85.00 |
|   rodaerial   |  0.0  |  0.0  |
| unkown aerial |  nan  |  nan  |
|     others    | 61.00 | 68.00 |
+---------------+-------+-------+
"""

# 使用正则表达式匹配每段IoU表，并提取最后一个
pattern = r'\+---------------\+-------\+-------\+([\s\S]+?)\+---------------\+-------\+-------\+'
tables = re.findall(pattern, log_content)

# 提取最后一个表
if tables:
    last_table = tables[-1].strip()

    # 解析最后一个表中的IoU信息
    iou_pattern = r'\|\s+([a-z\s]+)\s+\|\s+([\d\.nan]+)\s+\|\s+([\d\.nan]+)\s+\|'
    matches = re.findall(iou_pattern, last_table, re.IGNORECASE)

    # 提取有效类别的IoU值
    data = {}
    for class_name, iou_str, acc_str in matches:
        try:
            iou = float(iou_str)
            if not np.isnan(iou) and iou > 0:  # 排除无效类别
                data[class_name.strip()] = iou
        except ValueError:
            continue

    # 将单个日志的数据转换为DataFrame行并添加到总表
    df_row = pd.DataFrame([data])
    all_data = pd.concat([pd.DataFrame(), df_row], ignore_index=True)

    # 保存到CSV文件
    csv_path = 'last_log_iou_values.csv'
    all_data.to_csv(csv_path, index=False)
    print(f"最后一个IoU表的有效数据已保存到 {csv_path}")
else:
    print("未找到IoU结果表")
