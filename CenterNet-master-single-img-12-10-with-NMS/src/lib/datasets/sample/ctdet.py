# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from lib.utils.image import flip, color_aug
from lib.utils.image import get_affine_transform, affine_transform
from lib.utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from lib.utils.image import draw_dense_reg
import math

class CTDetDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):  # 转换ｂｂｏｘ的格式
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox  # bbox:[ltx,lty,rbx,rby]  ;  coco_box:[ltx,lty,w,h]

  def _get_border(self, border, size):  # 128, img.shape[1]
    i = 1
    while size - border // i <= border // i:  # " / "表示 浮点数除法，返回浮点结果;" // "表示整数除法
        i *= 2
    return border // i   # (border // i) == 128

  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    # print("55555:", img_path)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])   # getAnnIds:通过输入图片的id来得到图片的anno_id
    anns = self.coco.loadAnns(ids=ann_ids)           # loadAnns:通过anno_id，得到图片对应的详细anno信息
    # print(111111, anns)
    num_objs = min(len(anns), self.max_objs)

    img = cv2.imread(img_path)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res:
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w
    
    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:  # (not self.opt.not_rand_crop) = True
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))  # 从0.6-1.4中随机选取一个数字（步长为0.1）
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        sf = self.opt.scale   # 0.4 # when not using random crop apply scale augmentation.
        cf = self.opt.shift   # 0.1 # when not using random crop apply shift augmentation.
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      
      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1


    trans_input = get_affine_transform(c, s, 0, [input_w, input_h])  # 由三对点计算仿射变换
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)  # 对图像做仿射变换
    inp = (inp.astype(np.float32) / 255.)   # 归一化
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std                                         # ！！！！！！！ modified by zy
    inp = inp.transpose(2, 0, 1)

    # add by zy
    # inp = np.transpose(inp, [1, 2, 0])
    # cv2.imshow('input', inp)
    # cv2.waitKey(0)

    output_h = input_h // self.opt.down_ratio  # 网络输出的预测结果特征图是128*128,这里要将GTbox也缩小为128*128来计算loss
    output_w = input_w // self.opt.down_ratio  #
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    ori_wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    cxcy = np.zeros((self.max_objs, 2), dtype=np.float32)
    ori_cxcy = np.zeros((self.max_objs, 2), dtype=np.float32)
    cls_idx = np.zeros((self.max_objs), dtype=np.int64)
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
    
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      ori_h, ori_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      ori_cx, ori_cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
      cls_id = int(self.cat_ids[ann['category_id']])
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      # print(111111, bbox)
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      # print(222222, bbox)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
      # print(333333, h,w)
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[cls_id], ct_int, radius)
        wh[k] = 1. * w, 1. * h
        ori_wh[k] = 1. * ori_w, 1. * ori_h
        cxcy[k] = 1. * cx, 1. * cy
        ori_cxcy[k] = 1. * ori_cx, 1. * ori_cy
        cls_idx[k] = cls_id
        ind[k] = ct_int[1] * output_w + ct_int[0]     # ind这个参数是用来？？？ ind = int_cy*output_w + int_cx
        reg[k] = ct - ct_int                          # reg是用来回归精确小数与整数之间的误差的？？？
        reg_mask[k] = 1
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        if self.opt.dense_wh:
          draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])   # [tlx,tly,brx,bry,1,cls_id]
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'ori_wh': ori_wh, 'cxcy': cxcy, 'ori_cxcy': ori_cxcy, 'cls_idx': cls_idx}
    if self.opt.dense_wh:   # FALSE , 'apply weighted regression near center or just apply regression on center point.'
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:   # flase , 'category specific bounding box size.'
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:   # true
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta

    # print(ret['hm'].shape, )
    # cx, cy = int(ret['cxcy'][0][0]), int(ret['cxcy'][0][1])
    # w, h = int(ret['wh'][0][0]), int(ret['wh'][0][1])
    # print(cx,cy,w,h, ret['hm'][0][cy][cx])
    # print(1111111111111, ret['input'].shape, ret['hm'].shape, ret['wh'].shape, ret['reg'].shape, ret['reg_mask'].shape, ret['ind'].shape)
    # input_img = np.transpose([ret['hm'][cls_id]], [1,2,0])
    # crop_img = input_img[(cy - h // 2):(cy + h // 2), (cx - w // 2):(cx + w // 2)]
    # cv2.rectangle(input_img, (cx - w // 2, cy - h // 2),
    #               (cx + w // 2, cy + h // 2), (255, 250, 250),  2)
    # cv2.imshow('input', crop_img)
    # cv2.waitKey(0)
    # print(2222222222222, type(ret['input']), type(ret['hm']), type(ret['hm']), type(ret['reg']), type(ret['reg_mask']), type(ret['ind']))
    # print(ret.keys())   # dict_keys(['ind', 'reg_mask', 'hm', 'input', 'reg', 'wh'])
    return ret