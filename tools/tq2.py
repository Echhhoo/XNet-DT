from ctypes import c_float
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import f_oneway
def cohen_d(d1, d2):
    n1, n2 = len(d1), len(d2)
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = np.mean(d1), np.mean(d2)
    return (u1 - u2) / s
# 存储每个epoch的jc和dc值
jc_values_1 = []
jc_values_2 = []
jc_values_3 = []
dc_values_1 = []
dc_values_2 = []
dc_values_3 = []
# /home/ac/datb/wch/checkpoints/sup/ASOCA_DTCWT/xnetv3_3d_add-l=0.0001-e=600-s=6-g=0.99-b=2-w=20-100/sup_xnetv3_cbdice_fold03.txt
# /home/ac/datb/wch/checkpoints/sup/ASOCA_DWT/xnetv2_3d_min-l=0.0001-e=600-s=6-g=0.99-b=2-w=20-100/sup_xnetv2_dwt_dice_fold00.txt
# /home/ac/datb/wch/checkpoints/semi/ASOCA_DTCWT/xnetv3_3d_add-l=0.0001-e=600-s=6-g=0.99-b=2-w=20-20/semi_asoca_xnetv3_fold03.out
# /home/ac/datb/wch/checkpoints/semi/ASOCA_DWT/xnetv2_3d_min-l=0.0001-e=600-s=6-g=0.99-b=2-w=20-20/xnetv2_dwt_semi_fold03.out
# 读取数据并提取jc和dc值
with open('/home/ac/datb/wch/checkpoints/sup/ZDCTA_DWT/xnet3d-l=0.0001-e=200-s=2-g=0.99-b=2-w=20-100/sup_xnetv1_dwt_dice_fold00.out', 'r') as file:
    lines = file.readlines()
    for line in lines:
        lines = file.readlines()
        if 'Val  Jc 1: ' in line:
            parts = line.split('|')
            for part in parts:
                if 'Val  Jc 1: ' in part:
                    value = part.split(':')[-1].strip()
                    jc_values_1.append(float(value))
                elif 'Val  Jc 2: ' in part:
                    value = part.split(':')[-1].strip()
                    jc_values_2.append(float(value))
                # elif 'Val  Jc 3: ' in part:
                #     value = part.split(':')[-1].strip()
                #     jc_values_3.append(float(value))
        elif 'Val  Dc 1: ' in line:
            parts = line.split('|')
            for part in parts:
                if 'Val  Dc 1: ' in part:
                    value = part.split(':')[-1].strip()
                    dc_values_1.append(float(value))
                elif 'Val  Dc 2: ' in part:
                    value = part.split(':')[-1].strip()
                    dc_values_2.append(float(value))
                # elif 'Val  Dc 3: ' in part:
                #     value = part.split(':')[-1].strip()
                #     dc_values_3.append(float(value))

# 生成epoch的索引
epochs = np.arange(1, len(dc_values_1) + 1)

# 比较每个 epoch 的 val jc 1、2、3 的大小，记录每轮最高的数据和来源
best_dc_sources = []
best=0.0
for i in range(0,len(dc_values_1)):
    dc_values = [dc_values_1[i], dc_values_2[i]]
    if(best<np.max(dc_values)):
        best=np.max(dc_values)
        if(np.max(dc_values)>0.75):
            best_dc_index = np.argmax(dc_values)
            best_dc_sources.append(best_dc_index+1)
        # else:
        #     best_dc_sources.append(0)
        else:
            best_dc_sources.append(0)
    
# 统计每个 best_jc 的数量
best_dc1_count = best_dc_sources.count(1)
best_dc2_count = best_dc_sources.count(2)
# best_dc3_count = best_dc_sources.count(3)


print(f"best_dc1_count: {best_dc1_count}")
print(f"best_dc2_count: {best_dc2_count}")
# print(f"best_dc3_count: {best_dc3_count}")

# 初始化存储不同分支数据的列表
epochs_dc1 = []
epochs_dc2 = []
epochs_dc3 = []

# 存储每个分支对应的 epoch
for i in range(len(best_dc_sources)):
    if best_dc_sources[i] == 1:
        epochs_dc1.append(epochs[i])
    elif best_dc_sources[i] == 2:
        epochs_dc2.append(epochs[i])
    # elif best_dc_sources[i] == 3:
    #     epochs_dc3.append(epochs[i])
    
  
# # 将列表转换为 numpy 数组
# dc_values_2 = np.array(dc_values_2)
# dc_values_3 = np.array(dc_values_3)


# # 计算最大值
# max_dc_values_2 = np.max(dc_values_2)
# max_dc_values_3 = np.max(dc_values_3)


# print(f"Maximum value of dc_values_2: {max_dc_values_2}")
# print(f"Maximum value of dc_values_3: {max_dc_values_3}")


# # # 计算方差
# # var_dc_values_2 = np.var(dc_values_2)
# # var_dc_values_3 = np.var(dc_values_3)


# # print(f"Variance of dc_values_2: {var_dc_values_2}")
# # print(f"Variance of dc_values_3: {var_dc_values_3}")


# # 计算两个数组之间的元素级别的方差
# diff = dc_values_2 - dc_values_3
# var_diff = np.var(diff)


# print(f"Variance between dc_values_2 and dc_values_3: {var_diff}")
    
# # ANOVA 检验
# f_stat, p_value = f_oneway( dc_values_2, dc_values_3)

# print(f"F-statistic: {f_stat}, p-value: {p_value}")


# # 判断是否有显著差异
# if p_value < 0.05:
#     print("There is a significant difference among DC1, DC2, and DC3.")
# else:
#     print("There is no significant difference among DC1, DC2, and DC3.")
   
# # 计算 Cohen's d
# d = cohen_d(dc_values_2, dc_values_3)
# print(f"Cohen's d: {d}")

# print(f"效益值为:{abs(d)}")
# # 效应量解释
# if abs(d) < 0.2:
#     print("Small effect size")
# elif abs(d) < 0.5:
#     print("Medium effect size")
# else:
#     print("Large effect size")
     
# # 绘制jc的图像
# plt.figure(figsize=(12, 6))
# plt.plot(epochs, jc_values_1, label='Jc - Branch 1')
# plt.plot(epochs, jc_values_2, label='Jc - Branch 2')
# plt.plot(epochs, jc_values_3, label='Jc - Branch 3')
# plt.xlabel('Epoch')
# plt.ylabel('Jc')
# plt.title('Jc Values per Epoch')
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.savefig('sup_xnetv2_fold03_Jc.png')

# 绘制dc的图像
# plt.figure(figsize=(12, 6))
# plt.plot(epochs, dc_values_1, label='Dc - Branch 1')
# plt.plot(epochs, dc_values_2, label='Dc - Branch 2')
# plt.plot(epochs, dc_values_3, label='Dc - Branch 3')
# plt.xlabel('Epoch')
# plt.ylabel('Dc')
# plt.title('Dc Values per Epoch')
# plt.legend()
# plt.grid(True)
# plt.show()
# plt.savefig('sup_xnetv2_fold03_Dc.png')

# 绘制最佳 jc 来源的图像
plt.figure(figsize=(12, 6))
plt.plot(epochs_dc1, [1] * len(epochs_dc1), drawstyle='steps', marker='o', c='skyblue', markeredgecolor='blue', label='Dc 1')
plt.plot(epochs_dc2, [2] * len(epochs_dc2), drawstyle='steps', marker='o', c='lightcoral', markeredgecolor='orange', label='Dc 2')
plt.plot(epochs_dc3, [3] * len(epochs_dc3), drawstyle='steps', marker='o', c='lightgreen', markeredgecolor='green', label='Dc 3')
plt.xlabel('Epoch')
plt.ylabel('Best Dc Source')
plt.title('Best Dc Source per Epoch')
plt.grid(True)
plt.show()
plt.savefig('sup_xnetv3_fold03_best_dc.png')