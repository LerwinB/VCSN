from PIL import Image
import os
# 定义图像的路径列表
image_paths = os.listdir('data/UESAT_vis/images') 

# 设定网格大小
rows, cols = 8, 7

# 打开所有图像并获取每张图像的尺寸（假设所有图像大小一致）
images = [Image.open(os.path.join('data/UESAT_vis/images',img_path)) for img_path in image_paths]
width, height = images[0].size
gap = 10


big_image_width = cols * width + (cols - 1) * gap
big_image_height = rows * height + (rows - 1) * gap
big_image = Image.new('RGB', (big_image_width, big_image_height), (255, 255, 255))  # 白色背景

# 将每张小图粘贴到大图中的正确位置，添加缝隙
for index, image in enumerate(images):
    x = (index % cols) * (width + gap)   # 列的偏移量，包含缝隙
    y = (index // cols) * (height + gap) # 行的偏移量，包含缝隙
    big_image.paste(image, (x, y))


# 保存最终的大图
big_image.save("output_7x8_grid.png")
