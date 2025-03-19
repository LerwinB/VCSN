# Use an official Python runtime as a parent image
# FROM tensorflow/tensorflow:1.15.0-gpu-py3
FROM urso

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
# RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Make port 80 available to the world outside this container
# EXPOSE 80

# Define environment variable
# ENV NAME PoseEstimator

# Run pose_estimator.py when the container launches
# CMD ["python", "./tes.py"]
# CMD ["python3", "./pose_estimator.py","train"]
# CMD ["python3", "./pose_estimator.py"," train"," --dataset"," uesat ","--datalist sam5_VAE3_kmeans_pose.txt"," --weights coco --image_scale 0.5 --ori_resolution 24 --rot_aug --rot_image_aug"]