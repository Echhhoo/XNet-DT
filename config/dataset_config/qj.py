import os
import numpy as np
import SimpleITK as sitk

dataset_path = 'your_dataset_path'



def compute_mean_std(dataset_path):
    # 初始化累积器
    mean_accumulator = np.zeros(1)
    std_accumulator = np.zeros(1)
    total_samples = 0
    for image_file in os.listdir(dataset_path):
        if image_file.endswith(".nii") or image_file.endswith(".nii.gz"):
            image_path = os.path.join(dataset_path, image_file)

            # 读取图像
            image = sitk.ReadImage(image_path)
            image_np = sitk.GetArrayFromImage(image)

            # 将像素值缩放到 [0, 1]（如果需要）
            min_val = np.min(image_np)
            max_val = np.max(image_np)
            if max_val!= min_val:
                image_np = (image_np - min_val) / (max_val - min_val)

            # 计算均值和标准差
            mean_accumulator += np.mean(image_np)
            std_accumulator += np.std(image_np)
            total_samples += 1

    # 计算平均值
    mean_values = mean_accumulator / total_samples

    # 计算标准差
    std_values = std_accumulator / total_samples

    print("Mean:", mean_values)
    print("Std:", std_values)

    return mean_values, std_values


# 示例用法
dataset_path ='/home/ac/datb/wch/imagecas/train_sup_100/H'#L,H,
#mask_path = ""
mean_values, std_values = compute_mean_std(dataset_path)

print("MEAN:", mean_values)
print("STD:", std_values)
