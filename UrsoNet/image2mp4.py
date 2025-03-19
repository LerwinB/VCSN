import cv2
import os

# 图像文件夹路径
image_folder = './datasets/src_online'

# 输出视频文件名
video_name = 'output_video.mp4'

# 图像文件名格式
image_format = '{:04d}_rgb.png'  # 例如，image0000.png, image0001.png, ...

# 获取图像文件列表并排序
images = [img for img in os.listdir(image_folder) if img.endswith("_rgb.png")]
images.sort()

# 获取第一张图像的尺寸
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 设置视频编码器和输出视频对象
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

# 计算需要合成的帧数
fps = 1
duration = 1001
frames_to_write = min(int(fps * duration), len(images))

# 将图像逐帧写入视频
for i in range(frames_to_write):
    image_path = os.path.join(image_folder, image_format.format(i))
    video.write(cv2.imread(image_path))

# 关闭输出视频对象
video.release()

print("视频合成完成：", video_name)
