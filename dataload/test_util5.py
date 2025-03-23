import math

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from skimage.measure import label
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize
from tqdm import tqdm
from dataload.cldice import clDice
import re


def test_single_case(net, image1,image2,image3,image4, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image1.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image1 = np.pad(image1, [(wl_pad, wr_pad), (hl_pad, hr_pad),(dl_pad, dr_pad)], mode='constant', constant_values=0)
        image2 = np.pad(image2, [(wl_pad, wr_pad), (hl_pad, hr_pad),(dl_pad, dr_pad)], mode='constant', constant_values=0)
        image3 = np.pad(image3, [(wl_pad, wr_pad), (hl_pad, hr_pad),(dl_pad, dr_pad)], mode='constant', constant_values=0)
        image4 = np.pad(image4, [(wl_pad, wr_pad), (hl_pad, hr_pad),(dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image1.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map1 = np.zeros((num_classes, ) + image1.shape).astype(np.float32)
    score_map2 = np.zeros((num_classes, ) + image2.shape).astype(np.float32)
    score_map3 = np.zeros((num_classes, ) + image3.shape).astype(np.float32)
    cnt = np.zeros(image1.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch1 = image1[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch1 = np.expand_dims(np.expand_dims(
                    test_patch1, axis=0), axis=0).astype(np.float32)
                test_patch1 = torch.from_numpy(test_patch1).cuda()
                
                test_patch2 = image2[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch2 = np.expand_dims(np.expand_dims(
                    test_patch2, axis=0), axis=0).astype(np.float32)
                test_patch2 = torch.from_numpy(test_patch2).cuda()
                
                test_patch3 = image3[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch3 = np.expand_dims(np.expand_dims(
                    test_patch3, axis=0), axis=0).astype(np.float32)
                test_patch3 = torch.from_numpy(test_patch3).cuda()
                
                test_patch4 = image4[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch4 = np.expand_dims(np.expand_dims(
                    test_patch4, axis=0), axis=0).astype(np.float32)
                test_patch4 = torch.from_numpy(test_patch4).cuda()


                with torch.no_grad():
                    y1,y2,y3 = net(test_patch1,test_patch2,test_patch3,test_patch4)
                    # ensemble
                    y1 = torch.softmax(y1, dim=1)
                    y2 = torch.softmax(y2, dim=1)
                    y3 = torch.softmax(y3, dim=1)
                    # y1 = net(test_patch)
                    # # ensemble
                    # y = torch.softmax(y1, dim=1)
                y1 = y1.cpu().data.numpy()
                y1 = y1[0, :, :, :, :]
                y2 = y2.cpu().data.numpy()
                y2 = y2[0, :, :, :, :]
                y3 = y3.cpu().data.numpy()
                y3 = y3[0, :, :, :, :]
                score_map1[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map1[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y1
                score_map2[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map2[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y2
                score_map3[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map3[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y3
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map1 = score_map1/np.expand_dims(cnt, axis=0)
    score_map2 = score_map2/np.expand_dims(cnt, axis=0)
    score_map3 = score_map3/np.expand_dims(cnt, axis=0)
    label_map1 = np.argmax(score_map1, axis=0)
    label_map2 = np.argmax(score_map2, axis=0)
    label_map3 = np.argmax(score_map3, axis=0)

    if add_pad:
        label_map1 = label_map1[wl_pad:wl_pad+w,hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map1 = score_map1[:, wl_pad:wl_pad + w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        label_map2 = label_map2[wl_pad:wl_pad+w,hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map2 = score_map2[:, wl_pad:wl_pad +w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        label_map3 = label_map3[wl_pad:wl_pad+w,hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map3 = score_map3[:, wl_pad:wl_pad +w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    # score_map1=torch.from_numpy(score_map1).cuda()
    # score_map2=torch.from_numpy(score_map2).cuda()
    return label_map1,label_map2,label_map3


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        ravd = abs(metric.binary.ravd(pred, gt))
        hd = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        pre = metric.binary.precision(pred, gt)
        recall = metric.binary.recall(pred, gt)
        # Iou = metric.binary.IOU
        JI=metric.binary.jc(pred, gt)
        spec=metric.binary.specificity(pred,gt)
        cldice=clDice(pred,gt)
        return np.array([dice, ravd, hd, asd, pre, recall, JI,spec,cldice])
    else:
        return np.zeros(9)


def test_all_case(net, base_dir, method="unet_3D", test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24, test_save_path=None):
    with open(test_list, 'r') as f:
        image_list = re.findall(r'[\'](.*?)[\']', str(f.readlines()))  # 返回单引号内的内容
    # image_list = [base_dir + "/data/{}.h5".format(
    #     item.replace('\n', '').split(",")[0]) for item in image_list]
    # L_alpha = 0.0
    # H_beta = 0.2
    image_len=len(image_list)
    image_list = [base_dir + "/" + item.replace('\n', '') for item in image_list]
    print("Testing begin")
    total_metric1 = np.zeros((num_classes - 1, 9))
    total_metric2 = np.zeros((num_classes - 1, 9))
    total_metric3 = np.zeros((num_classes - 1, 9))
    metric = np.zeros((num_classes - 1, 9))
    ith = 0
    total_metric=[]
    with open(test_save_path + "/{}.txt".format(method), "a") as f:# 这里新建了一个文档但是没有写入
        for image_path in tqdm(image_list):
            metric1 = np.zeros((num_classes - 1, 9))
            metric2 = np.zeros((num_classes - 1, 9))
            metric3 = np.zeros((num_classes - 1, 9))
            ids = image_path.split("/")[-1].replace(".h5", "")
            h5f = h5py.File(image_path, 'r')
            image1 = h5f['image1'][:]
            image2 = h5f['image2'][:]
            image3 = h5f['image3'][:]
            image4 = h5f['image4'][:]
            label = h5f['label'][:]
            # L = image2 + L_alpha * image3
            # # L = (L - L.min()) / (L.max() - L.min()) * 255
            # L = (L - np.mean(L)) / np.std(L)

            # H =image3 + H_beta * image2
            # # H = (H - H.min()) / (H.max() - H.min()) * 255
            # H = (H - np.mean(H)) / np.std(H)
            # image2 = L
            # image3 = H
            prediction1,prediction2,prediction3 = test_single_case(net, image1,image2,image3,image4,stride_xy, stride_z, patch_size, num_classes=num_classes)
            for i in range(1, num_classes):
                metric1[i - 1, :] = cal_metric(prediction1 == i, label == i)
                metric2[i - 1, :]= cal_metric(prediction2 == i, label == i)
                metric3[i - 1, :]= cal_metric(prediction3 == i, label == i)
                total_metric1[i - 1, :] += metric1[i - 1, :]
                total_metric2[i - 1, :] += metric2[i - 1, :]
                total_metric3[i - 1, :] += metric3[i - 1, :]
                
            if metric1[:, 0].mean()>metric2[:, 0].mean() and metric1[:, 0].mean()>metric3[:, 0].mean() :
                total_metric.append(metric1)
                metric[i-1,:]+=metric1[i - 1, :] 
                prediction=prediction1
            elif metric2[:, 0].mean()>metric1[:, 0].mean() and metric2[:, 0].mean()>metric3[:, 0].mean():
                metric[i-1,:]+=metric2[i - 1, :] 
                total_metric.append(metric2)
                prediction=prediction2
            else:
                metric[i-1,:]+=metric3[i - 1, :] 
                total_metric.append(metric3)
                prediction=prediction3
                
            pred_itk = sitk.GetImageFromArray(prediction.transpose(2, 1, 0))
            pred_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(pred_itk, test_save_path +
                            "/{}_pred.nii.gz".format(ids))

            img_itk = sitk.GetImageFromArray(image1.transpose(2, 1, 0))
            img_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(img_itk, test_save_path +
                            "/{}_img.nii.gz".format(ids))

            lab_itk = sitk.GetImageFromArray(label.astype(np.uint8).transpose(2, 1, 0))
            lab_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(lab_itk, test_save_path +
                            "/{}_lab.nii.gz".format(ids))
            f.writelines("1st {} metric is:{}, {}, {}, {},{}, {}, {}, {},{}\n".format(ids,total_metric[ith][0,0], total_metric[ith][0,1], total_metric[ith][0,2], total_metric[ith][0,3],total_metric[ith][0,4],total_metric[ith][0,5],total_metric[ith][0,6],total_metric[ith][0,7],total_metric[ith][0,8]))
            ith += 1
        f.writelines("average metric is:{}, {}, {}, {},{}, {}, {}, {},{}\n".format(metric[:, 0].mean() /image_len, metric[:, 1].mean() /image_len, metric[:, 2].mean() / image_len, metric[:, 3].mean() /image_len, metric[:, 4].mean() /image_len, metric[:, 5].mean() /image_len, metric[:, 6].mean() /image_len, metric[:, 7].mean() /image_len, metric[:,8].mean() /image_len))
        avg_metric = metric / image_len
        
        f.writelines("sup1 average metric is:{}, {}, {}, {},{}, {}, {}, {},{}\n".format(total_metric1[:, 0].mean() /image_len, 
                    total_metric1[:, 1].mean() /image_len,total_metric1[:, 2].mean() / image_len,total_metric1[:, 3].mean() /image_len,total_metric1[:, 4].mean() /image_len,total_metric1[:, 5].mean() /image_len,total_metric1[:, 6].mean() /image_len,total_metric1[:, 7].mean() /image_len,total_metric1[:, 8].mean() /image_len))
        f.writelines("sup2 average metric is:{}, {}, {}, {},{}, {}, {}, {},{}\n".format(total_metric2[:, 0].mean() /image_len, 
                    total_metric2[:, 1].mean() /image_len,total_metric2[:, 2].mean() / image_len,total_metric2[:, 3].mean() /image_len,total_metric2[:, 4].mean() /image_len,total_metric2[:, 5].mean() /image_len,total_metric2[:, 6].mean() /image_len,total_metric2[:, 7].mean() /image_len,total_metric2[:, 8].mean() /image_len))
        f.writelines("sup3 average metric is:{}, {}, {}, {},{}, {}, {}, {},{}\n".format(total_metric3[:, 0].mean() /image_len, 
                    total_metric3[:, 1].mean() /image_len,total_metric3[:, 2].mean() / image_len,total_metric3[:, 3].mean() /image_len,total_metric3[:, 4].mean() /image_len,total_metric3[:, 5].mean() /image_len,total_metric3[:, 6].mean() /image_len,total_metric3[:, 7].mean() /image_len,total_metric3[:, 8].mean() /image_len))
    f.close()
    print("Testing end")
    return avg_metric



