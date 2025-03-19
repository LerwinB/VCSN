import cv2
import numpy as np
import pandas as pd

# 输入视频文件名
input_video_name = 'output_video.mp4'

# 输出视频文件名
output_video_name = 'output_video_with_optical_flow.mp4'

# Excel输出文件名
excel_name = 'optical_flow_velocity.xlsx'

# 设置LK光流参数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 打开输入视频
cap = cv2.VideoCapture(input_video_name)

# 获取视频的帧速率和尺寸
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 设置视频编码器和输出视频对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

# Excel输出数据
velocity_data = {'Frame': [], 'Avg Velocity': []}

# 读取第一帧
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)

frame_count = 0

while True:
    # 读取当前帧
    ret, next_frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 将图像转换为灰度
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # 计算光流
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_points, None, **lk_params)
     # 获取有效的光流点
    good_new = next_points[status == 1]
    good_old = prev_points[status == 1]

    # 可视化光流轨迹
    for j, (new, old) in enumerate(zip(next_points, prev_points)):
        a, b = new.ravel()
        c, d = old.ravel()
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        next_frame = cv2.line(next_frame, (int(a), int(b)), (int(c), int(d)), color, 2)
        next_frame = cv2.circle(next_frame, (int(a), int(b)), 5, color, -1)

    # 添加帧到输出视频
    video_out.write(next_frame)

    # 记录速度数据
    velocity = good_new - good_old
    avg_velocity = np.mean(np.linalg.norm(velocity, axis=1))
    velocity_data['Frame'].append(frame_count)
    velocity_data['Avg Velocity'].append(avg_velocity)

    # 更新变量
    prev_gray = next_gray.copy()
    prev_points = next_points

# 关闭输入和输出视频对象
cap.release()
video_out.release()

# 将速度数据保存到Excel文件
df = pd.DataFrame(velocity_data)
df.to_excel(excel_name, index=False)

print("视频和Excel文件生成完成：", output_video_name, excel_name)
