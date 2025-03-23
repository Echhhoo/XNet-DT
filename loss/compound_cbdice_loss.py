import torch
from loss.dice import SoftDiceLoss
from nnunetv2.training.loss.compound_losses import RobustCrossEntropyLoss
from loss.cbdice_loss import SoftcbDiceLoss, SoftclMDiceLoss
from loss.cldice_loss import SoftclDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
from loss import losses
from torch.nn.modules.loss import CrossEntropyLoss
import math

class DC_and_CE_and_CBDC_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, cbdc_kwargs, weight_ce=1, weight_dice=1, weight_cbdice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param ti_kwargs:
        :param weight_ce:
        :param weight_dice:
        :param weight_ti:
        """
        super(DC_and_CE_and_CBDC_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_cbdice = weight_cbdice
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.cbdice = SoftcbDiceLoss(**cbdc_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, t_skeletonize_flage=False):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        cbdice_loss = self.cbdice(net_output, target, t_skeletonize_flage=t_skeletonize_flage) if self.weight_cbdice != 0 else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_cbdice * cbdice_loss
        return result



class DC_CE_CBDC_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, cbdc_kwargs, num_classes=2, weight_ce=1, weight_dice=1, weight_cbdice=1, ignore_label=None,
                 dice_class=SoftDiceLoss,num_epochs=200):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param ti_kwargs:
        :param weight_ce:
        :param weight_dice:
        :param weight_ti:
        """
        super(DC_CE_CBDC_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_cbdice = weight_cbdice
        self.ignore_label = ignore_label
        
        self.num_epochs=num_epochs
        
        self.ce = CrossEntropyLoss()
        self.dc = losses.DiceLoss(num_classes).cuda()
        self.cbdice = SoftcbDiceLoss(**cbdc_kwargs)
        # self.cbdice=SoftclDiceLoss(**cbdc_kwargs)
        # self.cbdice = SoftclMDiceLoss(**cbdc_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, epoch=300, t_skeletonize_flage=False):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            # 断言目标变量的形状是否符合要求，这里要求目标变量（通常在图像分割等任务中按批次、类别、高度、宽度等维度组织。因为对于已经是独热编码形式的目标变量，当前代码并没有实现对忽略标签的处理机制
            # （针对名为DC_and_CE_loss的相关损失函数而言），确保传入的目标变量不是独热编码形式，而是类似标签图（每个像素位置只有一个类别标签值）
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                 '(DC_and_CE_loss)'
            # 创建一个布尔类型的掩码张量mask，用于标记哪些像素对应的标签不是忽略标签。通过比较目标变量target张量中的每个元素与self.ignore_label的值是否不相等，
            # 得到一个与target形状相同、元素为布尔值（True表示对应像素不是忽略标签，False表示是忽略标签）的临时张量，
            # 然后通过.bool()方法将其转换为明确的布尔类型张量（如果之前是其他可以表示布尔值的类型，如整数类型的0和1表示False和True，进行规范化转换）
            mask = (target!= self.ignore_label).bool()
            # 克隆目标变量target，创建一个与它完全相同的新张量target_dice，目的是在不改变原始target张量的情况下，
            # 对用于计算骰子系数相关操作的目标变量进行特定处理，避免对原始数据造成意外修改
            target_dice = torch.clone(target)
            # 通过布尔索引找到target_dice中与self.ignore_label相等的元素位置（通过target == self.ignore_label得到一个布尔掩码，用于筛选出这些位置），
            # 并将这些位置上的元素值修改为0。这么做的原因是，在后续计算骰子系数等操作时，那些被标记为忽略标签的像素已经通过掩码等方式排除了其对损失计算的影响，
            # 但需要将其值设置为一个合理的已知值（这里选择0，因为通常在分类标签中0也是一个合理的类别表示，并且不影响计算，毕竟已经通过掩码控制其是否参与计算了），
            # 避免在计算过程中出现异常情况
            target_dice[target == self.ignore_label] = 0
            # 计算掩码mask中值为True的元素个数，也就是统计那些不是忽略标签的有效像素的数量。这个统计值在一些情况下可能会有用，
            # 例如在计算平均损失或者进行一些基于有效像素数量的归一化操作时，可以作为分母等参与计算，不过具体是否使用以及如何使用取决于包含这段代码的整个损失函数等相关逻辑的具体需求
            num_fg = mask.sum()
        else:
            # 当self.ignore_label为None，即不存在需要忽略的标签时，直接将原始目标变量target赋值给target_dice，
            # 表示不需要对目标变量进行特殊处理，直接使用原始的目标变量进行后续计算
            target_dice = target
            # 将掩码mask设置为None，因为没有忽略标签也就不需要掩码来筛选像素了，后续相关代码在使用mask时可以通过判断其是否为None来区分是否存在忽略标签的情况，
            # 进而执行不同的操作逻辑
            mask = None
        # if epoch<200:
        #     a=0
        # else:
        #     a= (epoch-200) / self.num_epochs
        a= epoch * epoch / (self.num_epochs * self.num_epochs)
        # a= math.sinh(epoch/self.num_epochs)
        # a=2**(epoch/self.num_epochs)-1
        weight_cbdice = self.weight_cbdice * a
        weight_ce = self.weight_ce
        weight_dice = self.weight_dice *(2-a)
        # print(epoch,a,weight_cbdice,weight_ce)

        dc_loss = self.dc(net_output, target_dice) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        cbdice_loss = self.cbdice(net_output, target, t_skeletonize_flage=t_skeletonize_flage) if self.weight_cbdice != 0 else 0

        result = weight_ce * ce_loss + weight_dice * dc_loss + weight_cbdice * cbdice_loss
        return result

class DC_and_CE_and_CL_M_DC_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, cbdc_kwargs, weight_ce=1, weight_dice=1, weight_clMdice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param ti_kwargs:
        :param weight_ce:
        :param weight_dice:
        :param weight_ti:
        """
        super(DC_and_CE_and_CL_M_DC_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_clMdice = weight_clMdice
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.clMdice = SoftclMDiceLoss(**cbdc_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, t_skeletonize_flage=False):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        clMdice_loss = self.clMdice(net_output, target, t_skeletonize_flage=t_skeletonize_flage) if self.weight_clMdice != 0 else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_clMdice * clMdice_loss
        return result
