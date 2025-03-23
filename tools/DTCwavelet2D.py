import numpy as np
from PIL import Image
import pywt
import argparse
import os
import torch
import dtcwt
from pytorch_wavelets import DTCWTForward, DTCWTInverse
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--image_path', default='//10.0.5.233/shared_data/XNet/dataset/CREMI/train_unsup_80/image')
    # parser.add_argument('--L_path', default='//10.0.5.233/shared_data/XNet/dataset/CREMI/train_unsup_80/L')
    # parser.add_argument('--H_path', default='//10.0.5.233/shared_data/XNet/dataset/CREMI/train_unsup_80/H')
    parser.add_argument('--image_path', default='/home/ac/datb/wch/ASOCA2d/img')
    parser.add_argument('--L_path', default='/home/ac/datb/wch/ASOCA2d/DTCL')
    parser.add_argument('--H_path', default='/home/ac/datb/wch/ASOCA2d/DTCH')
    parser.add_argument('--wavelet_type', default='db2', help='haar, db2, bior1.5, bior2.4, coif1, dmey')
    parser.add_argument('--if_RGB', default=True)
    args = parser.parse_args()

    if not os.path.exists(args.L_path):
        os.mkdir(args.L_path)
    if not os.path.exists(args.H_path):
        os.mkdir(args.H_path)

    # for i in os.listdir(args.image_path):
    i="10694456_slice_200.png"
    image_path = os.path.join(args.image_path, i)
    L_path = os.path.join(args.L_path, i)
    H_path = os.path.join(args.H_path, i[:-4])

    if args.if_RGB:
        image = Image.open(image_path).convert('L')
    else:
        image = Image.open(image_path)
    image = np.array(image)
    # xfm = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
    # X=torch.from_numpy(image)
    # X=X[:].unsqueeze(0)
    # X=X[:].unsqueeze(0)
    # print(X.shape)
    # Yl, Yh = xfm(X.float)
    
    transform = dtcwt.Transform2d(biort='near_sym_b', qshift='qshift_b')
    mandrill_t = transform.forward(image, nlevels=1) 
    mandrill_t.highpasses
    low=mandrill_t.lowpass
    low = (low- low.min()) / (low.max() - low.min()) * 255
    low=Image.fromarray(low.astype(np.uint8))
    low.save(os.path.join(args.L_path, "ll.png"))
    
    high0=np.abs(mandrill_t.highpasses[0][:,:,0])
    high0=(high0-high0.min())/(high0.max()-high0.min())*255
    
    high1=np.abs(mandrill_t.highpasses[0][:,:,1])
    high1=(high1-high1.min())/(high1.max()-high1.min())*255
    
    high2=np.abs(mandrill_t.highpasses[0][:,:,2])
    high2=(high2-high2.min())/(high2.max()-high2.min())*255
    
    high3=np.abs(mandrill_t.highpasses[0][:,:,3])
    high3=(high3-high3.min())/(high3.max()-high3.min())*255
    
    high4=np.abs(mandrill_t.highpasses[0][:,:,4])
    high4=(high4-high4.min())/(high4.max()-high4.min())*255
   
    high5=np.abs(mandrill_t.highpasses[0][:,:,5])
    high5=(high1-high5.min())/(high5.max()-high5.min())*255
    
    
    merge1=high0+high1+high2+high3+high4+high5
    merge1 = (merge1-merge1.min()) / (merge1.max()-merge1.min()) * 255
    
    merge2=high0+high1+high2
    merge2 = (merge2-merge2.min()) / (merge2.max()-merge2.min()) * 255
    
    merge3=high3+high4+high5
    merge3 = (merge3-merge3.min()) / (merge3.max()-merge3.min()) * 255
    
    high0=Image.fromarray(high0.astype(np.uint8))
    high0.save(os.path.join(args.H_path, "high0.png"))
    
    high1=Image.fromarray(high1.astype(np.uint8))
    high1.save(os.path.join(args.H_path, "high1.png"))
    
    high2=Image.fromarray(high2.astype(np.uint8))
    high2.save(os.path.join(args.H_path, "high2.png"))
    
    high3=Image.fromarray(high3.astype(np.uint8))
    high3.save(os.path.join(args.H_path, "high3.png"))
    
    high4=Image.fromarray(high4.astype(np.uint8))
    high4.save(os.path.join(args.H_path, "high4.png"))
    
    high5=Image.fromarray(high5.astype(np.uint8))
    high5.save(os.path.join(args.H_path, "high5.png"))
    
    merge1 = Image.fromarray(merge1.astype(np.uint8))
    merge1.save(os.path.join(args.H_path, "merge1.png"))
    
    
    merge2 = Image.fromarray(merge2.astype(np.uint8))
    merge2.save(os.path.join(args.H_path, "merge2.png"))
    
    merge3 = Image.fromarray(merge3.astype(np.uint8))
    merge3.save(os.path.join(args.H_path, "merge3.png"))
    
    
    LL, (LH, HL, HH) = pywt.dwt2(image, args.wavelet_type)

    LL = (LL - LL.min()) / (LL.max() - LL.min()) * 255

    LL = Image.fromarray(LL.astype(np.uint8))
    LL.save(L_path)
    
    

    LH = (LH - LH.min()) / (LH.max() - LH.min()) * 255
    HL = (HL - HL.min()) / (HL.max() - HL.min()) * 255
    HH = (HH - HH.min()) / (HH.max() - HH.min()) * 255

    merge4 = HH + HL + LH
    
    LH = (LH - LH.min()) / (LH.max() - LH.min()) * 255

    LH = Image.fromarray(LH.astype(np.uint8))
    LL.save(os.path.join(args.H_path, "LH.png"))
    
    HL = (HL - HL.min()) / (HL.max() -HL.min()) * 255

    HL = Image.fromarray(HL.astype(np.uint8))
    HL.save(os.path.join(args.H_path, "HL.png"))
    
    HH= (HH - HH.min()) / (HH.max() - HH.min()) * 255

    HH = Image.fromarray(HH.astype(np.uint8))
    HH.save(os.path.join(args.H_path, "HH.png"))
    merge4 = (merge4-merge4.min()) / (merge4.max()-merge4.min()) * 255

    merge4 = Image.fromarray(merge4.astype(np.uint8))
    merge4.save(os.path.join(args.H_path, "H.png"))

