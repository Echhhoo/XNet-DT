import torch
from monai.transforms import distance_transform_edt
from loss.skeletonize import Skeletonize
from loss.soft_skeleton import SoftSkeletonize

class SoftcbDiceLoss(torch.nn.Module):
    def __init__(self, iter_=10, smooth = 1.):
        super(SoftcbDiceLoss, self).__init__()
        self.smooth = smooth
        # 这种方法保证了较高的拓扑精度，但运行速度较慢。了解实现细节。该方法基于论文“基于梯度优化的骨架化算法”（ICCV, 2023）
        # Topology-preserving skeletonization: https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
        # 该方法运行速度更快，但拓扑精度较低。论文“clDice -一种用于管状结构分割的新型拓扑保持损失函数”（CVPR, 2021）讨论了这种方法。
        # Morphological skeletonization: https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)

    def forward(self, y_pred, y_true, t_skeletonize_flage=False):
        if len(y_true.shape) == 4:
            dim = 2
        elif len(y_true.shape) == 5:
            dim = 3
        else:
            raise ValueError("y_true should be 4D or 5D tensor.")

        y_pred_fore = y_pred[:, 1:]
        y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0] # C foreground channels -> 1 channel
        y_pred_binary = torch.cat([y_pred[:, :1], y_pred_fore], dim=1)
        y_prob_binary = torch.softmax(y_pred_binary, 1)
        y_pred_prob = y_prob_binary[:, 1] # predicted probability map of foreground
        
        with torch.no_grad():
            y_true = torch.where(y_true > 0, 1, 0).squeeze(1).float() # ground truth of foreground
            y_pred_hard = (y_pred_prob > 0.5).float()
        
            if t_skeletonize_flage:
                skel_pred_hard = self.t_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.t_skeletonize(y_true.unsqueeze(1)).squeeze(1)
            else:
                skel_pred_hard = self.m_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.m_skeletonize(y_true.unsqueeze(1)).squeeze(1)

        skel_pred_prob = skel_pred_hard * y_pred_prob

        q_vl, q_slvl, q_sl = get_weights(y_true, skel_true, dim, prob_flag=False)
        q_vp, q_spvp, q_sp = get_weights(y_pred_prob, skel_pred_prob, dim, prob_flag=True)

        w_tprec = (torch.sum(torch.multiply(q_sp, q_vl))+self.smooth)/(torch.sum(combine_tensors(q_spvp, q_slvl, q_sp))+self.smooth)
        w_tsens = (torch.sum(torch.multiply(q_sl, q_vp))+self.smooth)/(torch.sum(combine_tensors(q_slvl, q_spvp, q_sl))+self.smooth)

        cb_dice_loss = 1 -2.0 * (w_tprec * w_tsens) / (w_tprec + w_tsens)
        
        return cb_dice_loss
    
class SoftclMDiceLoss(torch.nn.Module):
    def __init__(self, iter_=10, smooth = 1.):
        super(SoftclMDiceLoss, self).__init__()
        self.smooth = smooth
        
        # Topology-preserving skeletonization: https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')
        
        # Morphological skeletonization: https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)

    def forward(self, y_pred, y_true, t_skeletonize_flage=False):
        if len(y_true.shape) == 4:
            dim = 2
        elif len(y_true.shape) == 5:
            dim = 3
        else:
            raise ValueError("y_true should be 4D or 5D tensor.")

        y_pred_fore = y_pred[:, 1:]
        y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0] # C foreground channels -> 1 channel
        y_pred_binary = torch.cat([y_pred[:, :1], y_pred_fore], dim=1)
        y_prob_binary = torch.softmax(y_pred_binary, 1)
        y_pred_prob = y_prob_binary[:, 1] # predicted probability map of foreground
        
        with torch.no_grad():
            y_true = torch.where(y_true > 0, 1, 0).squeeze(1).float() # ground truth of foreground
            y_pred_hard = (y_pred_prob > 0.5).float()
        
            if t_skeletonize_flage:
                skel_pred_hard = self.t_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.t_skeletonize(y_true.unsqueeze(1)).squeeze(1)
            else:
                skel_pred_hard = self.m_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.m_skeletonize(y_true.unsqueeze(1)).squeeze(1)
        
        skel_pred_prob = skel_pred_hard * y_pred_prob

        q_vl, q_slvl, _ = get_weights(y_true, skel_true, dim, prob_flag=False)
        q_vp, q_spvp, _ = get_weights(y_pred_prob, skel_pred_prob, dim, prob_flag=True)

        q_sl = skel_true
        q_sp = skel_pred_prob

        w_tprec = (torch.sum(torch.multiply(q_sp, q_vl))+self.smooth)/(torch.sum(combine_tensors(q_spvp, q_slvl, q_sp))+self.smooth)
        w_tsens = (torch.sum(torch.multiply(q_sl, q_vp))+self.smooth)/(torch.sum(combine_tensors(q_slvl, q_spvp, q_sl))+self.smooth)

        # cl_m_dice_loss = - 2.0 * (w_tprec * w_tsens) / (w_tprec + w_tsens)
        cl_m_dice_loss = 1 -2.0 * (w_tprec * w_tsens) / (w_tprec + w_tsens)

        return cl_m_dice_loss

def combine_tensors(A, B, C):
    A_C = A * C
    B_C = B * C
    D = B_C.clone()
    mask_AC = (A != 0) & (B == 0)
    D[mask_AC] = A_C[mask_AC]
    return D

# 函数用于将权重应用于掩码和骨架。如果使用基础真值（' y_true '），则不考虑概率。然而，对于预测（“pred”），必须考虑概率。
def get_weights(mask_input, skel_input, dim, prob_flag=True):
    if prob_flag:
        mask_prob = mask_input
        skel_prob = skel_input

        mask = (mask_prob > 0.5).int()
        skel = (skel_prob > 0.5).int()
    else:
        mask = mask_input
        skel = skel_input

    distances = distance_transform_edt(mask).float()
    # 使用 distance_transform_edt 函数（这里假设它是一个已定义好的用于计算距离变换的函数，
    # 比如可能是基于欧几里得距离等计算每个像素或体素到掩码中最近零元素（例如背景到前景边界的距离等情况）的距离值）对二值化后的 mask 进行距离变换计算，
    # 并将结果的数据类型转换为 float 类型，得到每个位置到掩码中特定元素（比如前景边界等）的距离值存储在 distances 张量中。

    smooth = 1e-7
    #  定义了一个很小的平滑因子，这个平滑因子在后续一些涉及除法的计算中（比如归一化等操作防止分母为 0）可能会用到，起到稳定计算结果的作用。
    distances[mask == 0] = 0
    # 将 distances 中对应 mask 为 0（即背景等不需要关注的区域）的位置的距离值设置为 0，因为这些位置的距离对于后续基于前景或骨架的权重计算等操作通常是无关的，
    # 只关注前景或骨架所在区域的距离信息

    skel_radius = torch.zeros_like(distances, dtype=torch.float32)
    #  创建一个与 distances 形状相同、数据类型为 float32 的全 0 张量 skel_radius，用于存储骨架相关的半径信息（后续会根据骨架位置和距离值来更新其具体的值）
    skel_radius[skel == 1] = distances[skel == 1]
    # 将 skel 中值为 1（表示是骨架部分）的位置对应的 distances 中的距离值赋给 skel_radius，这样 skel_radius 中就记录了骨架位置到掩码中特定元素（如前景边界等）的距离值，也就是可以看作是骨架部分的某种半径度量（从边界到骨架的距离概念）。

    dist_map_norm = torch.zeros_like(distances, dtype=torch.float32)
    skel_R_norm = torch.zeros_like(skel_radius, dtype=torch.float32)
    I_norm = torch.zeros_like(mask, dtype=torch.float32)
    # 分别创建了 dist_map_norm、skel_R_norm、I_norm 这三个与 distances、skel_radius、mask 形状相同且数据类型为 float32 的全 0 张量，
    # 它们将用于后续存储归一化后的距离图、归一化后的骨架半径以及一种逆权重（根据不同维度有不同计算方式，下面循环中会详细说明）等信息，为进一步的权重计算做好准备。
    for i in range(skel_radius.shape[0]):
        distances_i = distances[i]
        skel_i = skel_radius[i]
        skel_radius_max = max(skel_i.max(), 1)
        skel_radius_min = max(skel_i.min(), 1)
    
        distances_i[distances_i > skel_radius_max] = skel_radius_max
        dist_map_norm[i] = distances_i / skel_radius_max
        skel_R_norm[i] = skel_i / skel_radius_max

        # subtraction-based inverse (linear)：
        if dim == 2:
            I_norm[i] = (skel_radius_max - skel_i + skel_radius_min) / skel_radius_max
        else:
            I_norm[i] = ((skel_radius_max - skel_i + skel_radius_min) / skel_radius_max) ** 2

        # division-based inverse (nonlinear):
        # if dim == 2:
        #     I_norm[i] = (1 + smooth) / (skel_R_norm[i] + smooth) # weight for skel
        # else:
        #     I_norm[i] = (1 + smooth) / (skel_R_norm[i] ** 2 + smooth)

    # 循环遍历数据的第一维（通常对应批次维度等情况）：
    # 通过 for i in range(skel_radius.shape[0]) 循环遍历 skel_radius 的第一维（假设数据按照批次、其他维度（如通道、高度、宽度等）的顺序组织），这样可以对每个批次的数据或者每个单独的数据样本进行以下的处理操作，以分别计算它们各自对应的归一化和权重相关的值。
    # 提取当前批次或样本的相关数据及计算最大最小半径值：
    # distances_i = distances[i] 和 skel_i = skel_radius[i] 分别提取当前批次或样本对应的距离值张量 distances_i 和骨架半径张量 skel_i，以便在当前数据样本内进行后续的局部处理。
    # skel_radius_max = max(skel_i.max(), 1) 和 skel_radius_min = max(skel_i.min(), 1) 计算当前样本中骨架半径的最大值和最小值，并且通过与 1 取最大值的操作（主要是为了防止出现半径值为 0 的情况，避免后续除法等计算出现问题），得到可靠的最大半径值 skel_radius_max 和最小半径值 skel_radius_min，这些值将用于后续的归一化和权重计算等操作。
    # 距离图归一化及骨架半径归一化：
    # distances_i[distances_i > skel_radius_max] = skel_radius_max 将当前样本的距离值张量 distances_i 中大于最大半径值 skel_radius_max 的距离值都设置为 skel_radius_max，进行一种截断处理，避免出现过大的距离值对后续归一化等操作产生异常影响。
    # dist_map_norm[i] = distances_i / skel_radius_max 将处理后的距离值张量 distances_i 中的每个元素除以最大半径值 skel_radius_max，实现距离图的归一化操作，将距离值映射到 0 到 1 的范围（或者接近这个范围，取决于具体的距离值分布和最大半径值情况），得到归一化后的距离图 dist_map_norm 中对应当前批次或样本的部分。
    # skel_R_norm[i] = skel_i / skel_radius_max 同样将当前样本的骨架半径张量 skel_i 中的每个元素除以最大半径值 skel_radius_max，对骨架半径进行归一化操作，得到归一化后的骨架半径 skel_R_norm 中对应当前批次或样本的部分，方便后续基于归一化后的骨架半径进行权重等相关计算。
    # 根据空间维度计算逆权重（线性方式示例）：
    # 当 dim == 2（即处理二维数据情况）时：
    # I_norm[i] = (skel_radius_max - skel_i + skel_radius_min) / skel_radius_max 通过一种基于减法的线性计算方式，利用最大半径值、当前骨架半径值以及最小半径值来计算当前样本对应的逆权重值，存储在 I_norm 中对应当前批次或样本的部分，具体这种线性计算方式的含义可能与想要突出或抑制不同半径区域的权重有关，比如对于半径较小的骨架部分给予相对较大的权重等（取决于具体的任务设计和需求）。
    # 当 dim!= 2（即处理三维等其他维度数据情况，这里以三维为例）时：
    # I_norm[i] = ((skel_radius_max - skel_i + skel_radius_min) / skel_radius_max) ** 2 采用类似的基于减法的计算方式，但进行了平方操作，实现一种非线性的逆权重计算，同样是为了根据骨架半径等信息来调整不同位置的权重，不过在三维情况下通过平方操作可能会对权重的调整效果产生不同的影响，比如更加强化或弱化某些半径区域的权重贡献等，具体效果取决于实际的数据和任务特点。
    # 另外，代码中还注释掉了一种基于除法的非线性逆权重计算方式（division-based inverse (nonlinear) 部分），这种方式通过 (1 + smooth) / (skel_R_norm[i] + smooth)（二维情况）或 (1 + smooth) / (skel_R_norm[i] ** 2 + smooth)（三维情况）来计算权重，也是利用了平滑因子和归一化后的骨架半径，同样是为了给不同的骨架相关区域赋予合适的权重，但与前面的线性和非线性（基于减法的）方式有所不同，具体使用哪种方式可以根据实际的实验和任务需求来选择。
    I_norm[skel == 0] = 0 # 0 for non-skeleton pixels

    if prob_flag:
        return dist_map_norm * mask_prob, skel_R_norm * mask_prob, I_norm * skel_prob
    else:
        return dist_map_norm * mask, skel_R_norm * mask, I_norm * skel
