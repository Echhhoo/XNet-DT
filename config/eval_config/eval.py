import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import directed_hausdorff
import torch


def evaluate(y_scores, y_true, interval=0.02):
    #  #对输入的预测分数张量进行 softmax 归一化，使得每个样本在不同类别上的预测概率总和为 1
    # y_scores = torch.softmax(y_scores, dim=1)
    # # 提取 softmax 后的预测分数张量中对应类别为 1 的分数
    # # 然后将其从 GPU 转移到 CPU，并从计算图中分离（detach），再转换为 NumPy 数组并展平为一维数组。
    # y_scores = y_scores[:, 1, ...].cpu().detach().numpy().flatten() 
    # # 对真实标签张量进行类似的处理，将其从 GPU 转移到 CPU，转换为 NumPy 数组并展平。
    # y_true = y_true.data.cpu().numpy().flatten()
    
    # # 创建一个从 0 到 0.9（不包括 0.9），步长为 interval 的 NumPy 数组，这个数组将作为不同的阈值来将预测分数转换为二值预测（即大于阈值为 1，小于等于阈值为 0）。
    # thresholds = np.arange(0, 0.9, interval)
    # jaccard = np.zeros(len(thresholds))
    # dice = np.zeros(len(thresholds))
    # y_true.astype(np.int8)

    # for indy in range(len(thresholds)):
    #     threshold = thresholds[indy]
    #     y_pred = (y_scores > threshold).astype(np.int8)

    #     sum_area = (y_pred + y_true)
    #     tp = float(np.sum(sum_area == 2))
    #     union = np.sum(sum_area == 1)
    #     jaccard[indy] = tp / float(union + tp)
    #     dice[indy] = 2 * tp / float(union + 2 * tp)
    # # 找到使得 Jaccard 系数最大的索引
    # thred_indx = np.argmax(jaccard)
    # # 根据最大 Jaccard 系数的索引，获取对应的最大 Jaccard 系数和对应的 Dice 系数
    # m_jaccard = jaccard[thred_indx]
    # m_dice = dice[thred_indx]
    # # 返回使得 Jaccard 系数最大的阈值，以及最大 Jaccard 系数和对应的 Dice 系数
    # return thresholds[thred_indx], m_jaccard, m_dice
    """
    根据给定的预测分数张量和真实标签张量，计算固定阈值0.5下的Jaccard系数和Dice系数

    参数:
    y_scores: 预测分数张量，形状通常为 (batch_size, num_classes,...)
    y_true: 真实标签张量，形状应与y_scores的除类别维度外的其他维度匹配

    返回:
    jaccard: 在阈值为0.5时的Jaccard系数
    dice: 在阈值为0.5时的Dice系数
    """
    # 对输入的预测分数张量进行softmax归一化，使得每个样本在不同类别上的预测概率总和为1
    y_scores = torch.softmax(y_scores, dim=1)
    # 提取softmax后的预测分数张量中对应类别为1的分数
    # 然后将其从GPU转移到CPU，并从计算图中分离（detach），再转换为NumPy数组并展平为一维数组。
    y_scores = y_scores[:, 1,...].cpu().detach().numpy().flatten()
    # 对真实标签张量进行类似的处理，将其从GPU转移到CPU，转换为NumPy数组并展平。
    y_true = y_true.data.cpu().numpy().flatten()

    # 固定阈值设为0.5
    threshold = 0.5
    y_pred = (y_scores > threshold).astype(np.int8)

    sum_area = (y_pred + y_true)
    tp = np.sum(sum_area == 2)
    union = np.sum(sum_area == 1)
    # 计算Jaccard系数（交并比，Intersection over Union，IoU）
    jaccard = tp / (union + tp) if (union + tp) > 0 else 0
    # 计算Dice系数
    dice = 2 * tp / (union + 2 * tp) if (union + 2 * tp) > 0 else 0

    return threshold,jaccard, dice



def evaluate_multi(y_scores, y_true):

    y_scores = torch.softmax(y_scores, dim=1)
    y_pred = torch.max(y_scores, 1)[1]
    y_pred = y_pred.data.cpu().numpy().flatten()
    y_true = y_true.data.cpu().numpy().flatten()

    hist = confusion_matrix(y_true, y_pred)

    hist_diag = np.diag(hist)
    hist_sum_0 = hist.sum(axis=0)
    hist_sum_1 = hist.sum(axis=1)

    jaccard = hist_diag / (hist_sum_1 + hist_sum_0 - hist_diag)
    m_jaccard = np.nanmean(jaccard)
    dice = 2 * hist_diag / (hist_sum_1 + hist_sum_0)
    m_dice = np.nanmean(dice)

    return jaccard, m_jaccard, dice, m_dice




