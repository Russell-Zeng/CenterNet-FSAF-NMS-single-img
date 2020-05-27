from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import cv2
import os
import traceback

from src.lib.models.losses import FocalLoss
from src.lib.models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from src.lib.models.decode import ctdet_decode
from src.lib.models.utils import _sigmoid
from src.lib.utils.debugger import Debugger
from src.lib.utils.post_process import ctdet_post_process
from src.lib.utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer

class CtdetLoss(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    # print(self.crit, self.crit_reg, self.crit_wh)   # FocalLoss() RegL1Loss() RegL1Loss()
    self.opt = opt

  def forward(self, all_outputs, batch):

    # for key in batch.keys():
      # print(111111111111, key)
      # print(2222222,  batch[key].size(), batch[key][0].size())
    opt = self.opt
    # batch_size = opt.batch_size
    batch_size = len(batch['input'])
    batch_loss = 0
    batch_hm_loss = 0
    batch_wh_loss = 0
    batch_off_loss = 0
    for i in range(batch_size):
      try:
        batch_img = {}   # 对batch中的每张图片，都重置一次batch_img
        try :
          for key in batch.keys():
            if key != 'meta':
              batch_img[key] = batch[key][i].unsqueeze(0)    # 要将字典中的值包装为4维张量
            else:
              pass
            # raise KeyError
        except Exception as e:  # 如果发生异常，那就返回预设的loss值
          print('Error1!!!', e)
          print(traceback.format_exc())
          continue
        # for key in batch_img.keys():
        #   print(222, key, batch_img[key].size(), batch_img[key][0].size())
        real_obj_num = list(batch_img['reg_mask'][0]).count(1)  #  batch-img中有效obj的数量, 一般是3,4,5..
        obj_num = int(str(batch_img['reg_mask'][0].size()).split('[')[-1].split(']')[0])  # obj_num = 128

        img_loss = 0
        img_hm_loss = 0
        img_wh_loss = 0
        img_off_loss = 0
        # print('total obj num:', real_obj_num)
        for obj_idx in range(real_obj_num):
          # print('obj_idx:', obj_idx)
          cxcy = list(batch_img['cxcy'][0][obj_idx].cpu().numpy())  # 将tensor数据全部转换成numpy数组，方便后面读取
          cx,cy = int(cxcy[0]), int(cxcy[1])
          wh = list(batch_img['wh'][0][obj_idx].cpu().numpy())
          w, h = int(wh[0]), int(wh[1])                           # 原始的obj的w，h
          # w, h = int(w*1.3), int(h*1.3)                         # 扩大后的obj的w和h，用来扩大掩码mask的范围！！提升large物体的检测效果
          w, h = int(w * 0.5), int(h * 0.5)                       # FSAF只用了0.2倍的area
          cls_idx = int(batch_img['cls_idx'][0][obj_idx].cpu().numpy())
          # print('obj_info:', cxcy, wh, cls_idx)
          hm_mask = torch.zeros_like(batch_img['hm']).cuda()
          hm_h, hm_w = hm_mask.size()[2], hm_mask.size()[3]
          y0, y1, x0, x1 = cy-(h//2), cy+(h//2), cx-(w//2), cx+(w//2)
          y0, y1, x0, x1 = max(1, y0-1), min(hm_h-1, y1+1), max(1, x0-1), min(hm_w-1, x1+1) # 为掩码范围限定距离
          hm_mask[0][cls_idx][y0:y1,x0:x1] = 1
          obj_hm = batch_img['hm'] * hm_mask   # !!!!!!!!单个物体的heatmap！！！！！！
          # obj_hm = obj_hm[:,:,y0:y1,x0:x1]    # [1, 20, 101, 59]--->[batch_size, channels, obj_h, obj_w]
          # print(1111, x0, y0, x1, y1, obj_hm.size())

          wh_mask = torch.zeros_like(batch_img['wh']).cuda()
          wh_mask[0][obj_idx,:] = 1
          obj_wh = batch_img['wh'] * wh_mask  # !!!!!!!!单个物体的wh！！！！！！

          reg_mask = torch.zeros_like(batch_img['reg']).cuda()
          reg_mask[0][obj_idx, :] = 1
          obj_reg = batch_img['reg'] * reg_mask  # !!!!!!!!单个物体的wh！！！！！！

          ind_mask = torch.zeros_like(batch_img['ind']).cuda()
          # print(1111, ind_mask.size(), batch_img['ind'])
          ind_mask[0][obj_idx] = 1
          obj_ind = batch_img['ind'] * ind_mask  # !!!!!!!!单个物体的ind！！！！！！
          # print(22222, obj_ind.size(), obj_ind)

          reg_mask_mask = torch.zeros_like(batch_img['reg_mask']).cuda()
          # print(1111, reg_mask_mask.size(), batch_img['reg_mask'])
          reg_mask_mask[0][obj_idx] = 1
          obj_reg_mask = batch_img['reg_mask'] * reg_mask_mask  # !!!!!!!!单个物体的reg_mask！！！！！！
          # print(22222, reg_mask_mask.size(), reg_mask_mask)

          output_loss = [] # 用来存储当前obj在各个特征图上计算的loss
          output_hm_loss = []
          output_wh_loss = []
          output_off_loss = []
          for key, outputs in enumerate(all_outputs):
            hm_loss, wh_loss, off_loss = 0, 0, 0  # add by zy
            for s in range(opt.num_stacks):  # opt.num_stacks = 2 if opt.arch == 'hourglass' else 1
              output = [outputs][s]  # output = outputs[0]

              # 原本output['hm']在这里经过激活函数sigmoid，但是由于for循环，sigmoid放在这里会运行很多次
              # 所以将sigmoid激活函数直接放在网络结构的最后一层实现，也就没有判断opt.mse_loss参数的值！！！
              # if not opt.mse_loss:  #  not opt.mse_loss = True
              #   output['hm'] = _sigmoid(output['hm'])   # !!!在网络结构中没经过激活函数，在这经过了。。

              # print(batch_img['input'].size(), batch_img['hm'].size(), hm_mask.size(), obj_hm.size())

              # ===============================
              # input_img0 = np.transpose(batch_img['input'][0].cpu().numpy(), [1, 2, 0])    # 原始图像
              # input_img1 = np.transpose(hm_mask[0][cls_idx].unsqueeze(0).cpu().numpy(), [1,2,0])  # heatmap掩码图像
              # input_img2 = np.transpose(batch_img['hm'][0][cls_idx].unsqueeze(0).cpu().numpy(), [1, 2, 0])  # 输入图像的heatmap图像
              # input_img3 = np.transpose(obj_hm[0][cls_idx].unsqueeze(0).cpu().numpy(), [1, 2, 0])       #　裁切前的单个obj的GT heatmap图像（128*128）
              # input_img3 = np.transpose(obj_hm[0][cls_idx].unsqueeze(0)[:,y0:y1,x0:x1].cpu().numpy(), [1, 2, 0])  #　裁切后的单个obj的GT heatmap图像（h*w）
              # input_img4 = np.transpose(output['hm'][i][cls_idx].unsqueeze(0).cpu().detach().numpy(),[1, 2, 0])   #　裁切前的单个obj的pred heatmap图像（128*128）
              # input_img4 = np.transpose(output[ 'hm'][i][cls_idx].unsqueeze(0)[:,y0:y1,x0:x1].cpu().detach().numpy(), [1, 2, 0]) #　裁切后的单个obj的pred heatmap图像（h*w）

              # cv2.imshow('input0', input_img0)
              # cv2.imshow('input1', input_img1)
              # cv2.imshow('input2', input_img2)
              # cv2.imshow('input3', input_img3)
              # cv2.imshow('input4', input_img4)
              # cv2.waitKey(0)
              # ================================

              # if opt.eval_oracle_hm:  # help='use ground center heatmap.'
              #   output['hm'] = batch['hm']
              # if opt.eval_oracle_wh:  # help='use ground truth bounding box size.')
              #   output['wh'] = torch.from_numpy(gen_oracle_map(
              #     batch['wh'].detach().cpu().numpy(),
              #     batch['ind'].detach().cpu().numpy(),
              #     output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
              # if opt.eval_oracle_offset:  # help='use ground truth local heatmap offset.'
              #   output['reg'] = torch.from_numpy(gen_oracle_map(
              #     batch['reg'].detach().cpu().numpy(),
              #     batch['ind'].detach().cpu().numpy(),
              #     output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

              # 用focalloss计算hm-loss
              # print(hm_loss)
              # print((output['wh'][i].unsqueeze(0)).size(),  obj_wh.size(), wh_mask.size())
              # print(111111111, output['hm'][i].size(), obj_hm.size())
              hm_loss += self.crit(output['hm'][i].unsqueeze(0)[:,:,y0:y1,x0:x1], obj_hm[:,:,y0:y1,x0:x1]) / opt.num_stacks
              # print(hm_loss)
              # 用l1loss计算wh-loss
              if opt.wh_weight > 0:  # opt.wh_weight == 0.1 : 'loss weight for bounding box size.'
                if opt.dense_wh:  # False # help='apply weighted regression near center or just apply regression on center point.'
                  mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                  wh_loss += (self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                           batch['dense_wh'] * batch['dense_wh_mask']) /
                              mask_weight) / opt.num_stacks
                elif opt.cat_spec_wh:  # False # help='category specific bounding box size.'
                  wh_loss += self.crit_wh(
                    output['wh'], batch['cat_spec_mask'], batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
                else:
                  # wh_loss += self.crit_reg(output['wh'][i].unsqueeze(0), batch['reg_mask'][i].unsqueeze(0), batch['ind'][i].unsqueeze(0), obj_wh) / opt.num_stacks
                  wh_loss += self.crit_reg(output['wh'][i].unsqueeze(0), obj_reg_mask, obj_ind, obj_wh) / opt.num_stacks
              # 用l1loss计算reg-loss
              if opt.reg_offset and opt.off_weight > 0:  # reg_offset = True , off_weight = 1
                # off_loss += self.crit_reg(output['reg'][i].unsqueeze(0), batch['reg_mask'][i].unsqueeze(0), batch['ind'][i].unsqueeze(0), obj_reg) / opt.num_stacks
                off_loss += self.crit_reg(output['reg'][i].unsqueeze(0), obj_reg_mask, obj_ind, obj_reg) / opt.num_stacks
              # 计算总loss
              loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
            output_loss.append(loss)   # output_loss是列表，保存当前obj在每个特征图下计算的loss
            output_hm_loss.append(hm_loss)
            output_wh_loss.append(wh_loss)
            output_off_loss.append(off_loss)
          # print(output_loss)
          # print('an obj done!!!!!!!!!!!')
          min_output_loss = min(output_loss)  # min_output_loss是该obj在所有特征图中回归的最小loss，该物体就归min_output_loss_idx层特征图回归
          min_output_loss_idx = output_loss.index(min_output_loss)
          # print('obj_info:', cxcy, wh, cls_idx, 'ouput_choice_idx:', min_output_loss_idx)
          min_output_hm_loss = output_hm_loss[min_output_loss_idx]
          min_output_wh_loss = output_wh_loss[min_output_loss_idx]
          min_output_off_loss = output_off_loss[min_output_loss_idx]

          #===============add by zy
          # if i==1:  # 为了减小txt文件体积，只记录每个batch的第一张
          #   output_choice_log = '/home/zy/zy/2new_network/CenterNet-master/output_choice.log'
          #   log_txt = 'obj_info: %s, %s, %d'%(cxcy, wh, cls_idx) + '   ;   ouput_choice_idx: %d'%min_output_loss_idx
          #   log = open(output_choice_log, 'a')
          #   log.write(log_txt + '\n')
          #   log.flush()
          #   log.close()
          #=================

          img_loss += min_output_loss   # img_loss是一张图像上所有obj的最小loss总和
          img_hm_loss += min_output_hm_loss
          img_wh_loss += min_output_wh_loss
          img_off_loss += min_output_off_loss
        if real_obj_num > 0:
          batch_loss += (img_loss)   # batch_loss是一个batch中上所有img的loss总和
          batch_hm_loss += (img_hm_loss)
          batch_wh_loss += (img_wh_loss)
          batch_off_loss += (img_off_loss)
        else:
          batch_loss += img_loss  # batch_loss是一个batch中上所有img的loss总和
          batch_hm_loss += img_hm_loss
          batch_wh_loss += img_wh_loss
          batch_off_loss += img_off_loss
        # raise KeyError
      except Exception as e:   # 如果发生异常，那就返回预设的loss值
        print('Error_all!!!', e)
        print(traceback.format_exc())
        continue
    final_loss = batch_loss / batch_size  # final_loss是一个batch上的loss平均值
    final_hm_loss = batch_hm_loss / batch_size
    final_wh_loss = batch_wh_loss / batch_size
    final_off_loss = batch_off_loss / batch_size
    loss_stats = {'loss': final_loss, 'hm_loss': final_hm_loss,
                  'wh_loss': final_wh_loss, 'off_loss': final_off_loss}
    # print(final_loss)
    return final_loss, loss_stats


        # print(batch_img['input'].size(), batch_img['hm'].size(), obj_mask.size(), obj_hm.size())
        #
        # input_img = np.transpose(obj_mask[0][cls_idx].unsqueeze(0).cpu().numpy(), [1,2,0])
        # input_img2 = np.transpose(batch_img['hm'][0][cls_idx].unsqueeze(0).cpu().numpy(), [1, 2, 0])
        # # input_img3 = np.transpose(batch_img['input'][0].cpu().numpy(), [1, 2, 0])
        # input_img3 = np.transpose(obj_hm[0][cls_idx].unsqueeze(0).cpu().numpy(), [1, 2, 0])
        #
        # cv2.imshow('input', input_img)
        # cv2.imshow('input2', input_img2)
        # cv2.imshow('input3', input_img3)
        # cv2.waitKey(0)



    #
    # opt = self.opt
    # hm_loss, wh_loss, off_loss = 0, 0, 0
    # for s in range(opt.num_stacks):  # opt.num_stacks = 2 if opt.arch == 'hourglass' else 1
    #   output = [all_outputs[0]][s]  # output = outputs[0]
    #   if not opt.mse_loss:  # not opt.mse_loss = True
    #     output['hm'] = _sigmoid(output['hm'])  # !!!在网络结构中没经过激活函数，在这经过了。。
    #   '''
    #   if opt.eval_oracle_hm:  # help='use ground center heatmap.'
    #     output['hm'] = batch['hm']
    #   if opt.eval_oracle_wh:  # help='use ground truth bounding box size.')
    #     output['wh'] = torch.from_numpy(gen_oracle_map(
    #       batch['wh'].detach().cpu().numpy(),
    #       batch['ind'].detach().cpu().numpy(),
    #       output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
    #   if opt.eval_oracle_offset:  # help='use ground truth local heatmap offset.'
    #     output['reg'] = torch.from_numpy(gen_oracle_map(
    #       batch['reg'].detach().cpu().numpy(),
    #       batch['ind'].detach().cpu().numpy(),
    #       output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)
    #   '''
    #   # print(1111111111, output['hm'].size(), batch['hm'].size())
    #   # hm_np = batch['hm'][0].cpu().data.numpy()
    #   # hm_img = np.transpose(hm_np, [1,2,0])
    #   # cv2.imshow('input', hm_img)
    #   # cv2.waitKey(0)
    #   # print(11111111111, batch['wh'][0][0])
    #   hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks  # 用focalloss计算hm-loss
    #
    #   if opt.wh_weight > 0:  # opt.wh_weight == 0.1 : 'loss weight for bounding box size.'
    #     if opt.dense_wh:  # False # help='apply weighted regression near center or just apply regression on center point.'
    #       mask_weight = batch['dense_wh_mask'].sum() + 1e-4
    #       wh_loss += (self.crit_wh(output['wh'] * batch['dense_wh_mask'], batch['dense_wh'] * batch['dense_wh_mask']) /
    #                   mask_weight) / opt.num_stacks
    #     elif opt.cat_spec_wh:  # False # help='category specific bounding box size.'
    #       wh_loss += self.crit_wh(
    #         output['wh'], batch['cat_spec_mask'], batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
    #     else:
    #       wh_loss += self.crit_reg(output['wh'], batch['reg_mask'], batch['ind'], batch['wh']) / opt.num_stacks
    #   if opt.reg_offset and opt.off_weight > 0:  # reg_offset = True , off_weight = 1
    #     off_loss += self.crit_reg(output['reg'], batch['reg_mask'], batch['ind'], batch['reg']) / opt.num_stacks
    #
    # loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
    #        opt.off_weight * off_loss
    # loss_stats = {'loss': loss, 'hm_loss': hm_loss,
    #               'wh_loss': wh_loss, 'off_loss': off_loss}
    # return loss, loss_stats

  # def forward1(self, all_outputs, batch):
  #
  #   # for key in batch.keys():
  #   #   print(key, batch[key].size(), batch[key][0].size())
  #
  #   opt = self.opt
  #   batch_size = opt.batch_size
  #   # hm_loss, wh_loss, off_loss = 0, 0, 0
  #   all_loss_res = []
  #   all_hm_loss = []
  #   all_wh_loss = []
  #   all_off_loss = []
  #   for i in range(batch_size):
  #     batch_img = {}   # 对batch中的每张图片，都重置一次batch_img
  #     for key in batch.keys():
  #       batch_img[key] = [ batch[key][i] ]   # 要将字典中的值包装为4维张量
  #     obj_num = list(batch_img['reg_mask'][0]).count(1)  #  batch-img中有效obj的数量
  #
  #
  #     batch_img = [batch_img]
  #     obj_num = batch_img['reg_mask']
  #     for key,obj in enumerate(len())
  #     for outputs in all_outputs:
  #       hm_loss, wh_loss, off_loss = 0, 0, 0  # add by zy
  #       for s in range(opt.num_stacks):  # opt.num_stacks = 2 if opt.arch == 'hourglass' else 1
  #         output = [outputs][s]  # output = outputs[0]
  #         if not opt.mse_loss:  #  not opt.mse_loss = True
  #           output['hm'] = _sigmoid(output['hm'])   # !!!在网络结构中没经过激活函数，在这经过了。。
  #         '''
  #         if opt.eval_oracle_hm:   # help='use ground center heatmap.'
  #           output['hm'] = batch['hm']
  #         if opt.eval_oracle_wh:   # help='use ground truth bounding box size.')
  #           output['wh'] = torch.from_numpy(gen_oracle_map(
  #             batch['wh'].detach().cpu().numpy(),
  #             batch['ind'].detach().cpu().numpy(),
  #             output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
  #         if opt.eval_oracle_offset:  # help='use ground truth local heatmap offset.'
  #           output['reg'] = torch.from_numpy(gen_oracle_map(
  #             batch['reg'].detach().cpu().numpy(),
  #             batch['ind'].detach().cpu().numpy(),
  #             output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)
  #         '''
  #         # print(1111111111, output['hm'].size(), batch['hm'].size())
  #         # hm_np = batch['hm'][0].cpu().data.numpy()
  #         # hm_img = np.transpose(hm_np, [1,2,0])
  #         # cv2.imshow('input', hm_img)
  #         # cv2.waitKey(0)
  #         # print(11111111111, batch['wh'][0][0])
  #
  #         hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks  # 用focalloss计算hm-loss
  #         # print(hm_loss)
  #         if opt.wh_weight > 0:  # opt.wh_weight == 0.1 : 'loss weight for bounding box size.'
  #           if opt.dense_wh:   # False # help='apply weighted regression near center or just apply regression on center point.'
  #             mask_weight = batch['dense_wh_mask'].sum() + 1e-4
  #             wh_loss += (self.crit_wh(output['wh'] * batch['dense_wh_mask'], batch['dense_wh'] * batch['dense_wh_mask']) /
  #               mask_weight) / opt.num_stacks
  #           elif opt.cat_spec_wh: # False # help='category specific bounding box size.'
  #             wh_loss += self.crit_wh(
  #               output['wh'], batch['cat_spec_mask'],batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
  #           else:
  #             wh_loss += self.crit_reg(output['wh'], batch['reg_mask'], batch['ind'], batch['wh']) / opt.num_stacks
  #
  #         if opt.reg_offset and opt.off_weight > 0:   # reg_offset = True , off_weight = 1
  #           off_loss += self.crit_reg(output['reg'], batch['reg_mask'], batch['ind'], batch['reg']) / opt.num_stacks
  #
  #       loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
  #              opt.off_weight * off_loss
  #       all_loss_res.append(loss)
  #       all_hm_loss.append(hm_loss)
  #       all_wh_loss.append(wh_loss)
  #       all_off_loss.append(off_loss)
  #
  #
  #
  #   print('all_loss_res:', len(all_loss_res), all_loss_res[0].data, all_loss_res[1].data, all_loss_res[2].data, all_loss_res[3].data, all_loss_res[4].data,
  #         all_loss_res[5].data, all_loss_res[6].data)
  #   # loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
  #   #        opt.off_weight * off_loss
  #   # all_loss_res = all_loss_res[1:]
  #   min_idx = all_loss_res.index(min(all_loss_res))
  #   print(min_idx, min(all_loss_res))
  #   loss_stats = {'loss': all_loss_res[min_idx], 'hm_loss': all_hm_loss[min_idx],
  #                 'wh_loss': all_wh_loss[min_idx], 'off_loss': all_off_loss[min_idx]}
  #   return min(all_loss_res), loss_stats

    # hm_loss, wh_loss, off_loss = 0, 0, 0
    # for s in range(opt.num_stacks):  # opt.num_stacks = 2 if opt.arch == 'hourglass' else 1
    #   output = [all_outputs[0]][s]  # output = outputs[0]
    #   if not opt.mse_loss:  # not opt.mse_loss = True
    #     output['hm'] = _sigmoid(output['hm'])  # !!!在网络结构中没经过激活函数，在这经过了。。
    #
    #   if opt.eval_oracle_hm:  # help='use ground center heatmap.'
    #     output['hm'] = batch['hm']
    #   if opt.eval_oracle_wh:  # help='use ground truth bounding box size.')
    #     output['wh'] = torch.from_numpy(gen_oracle_map(
    #       batch['wh'].detach().cpu().numpy(),
    #       batch['ind'].detach().cpu().numpy(),
    #       output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
    #   if opt.eval_oracle_offset:  # help='use ground truth local heatmap offset.'
    #     output['reg'] = torch.from_numpy(gen_oracle_map(
    #       batch['reg'].detach().cpu().numpy(),
    #       batch['ind'].detach().cpu().numpy(),
    #       output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)
    #
    #   # print(1111111111, output['hm'].size(), batch['hm'].size())
    #   # hm_np = batch['hm'][0].cpu().data.numpy()
    #   # hm_img = np.transpose(hm_np, [1,2,0])
    #   # cv2.imshow('input', hm_img)
    #   # cv2.waitKey(0)
    #   # print(11111111111, batch['wh'][0][0])
    #   hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks  # 用focalloss计算hm-loss
    #
    #   if opt.wh_weight > 0:  # opt.wh_weight == 0.1 : 'loss weight for bounding box size.'
    #     if opt.dense_wh:  # False # help='apply weighted regression near center or just apply regression on center point.'
    #       mask_weight = batch['dense_wh_mask'].sum() + 1e-4
    #       wh_loss += (self.crit_wh(output['wh'] * batch['dense_wh_mask'], batch['dense_wh'] * batch['dense_wh_mask']) /
    #                   mask_weight) / opt.num_stacks
    #     elif opt.cat_spec_wh:  # False # help='category specific bounding box size.'
    #       wh_loss += self.crit_wh(
    #         output['wh'], batch['cat_spec_mask'], batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
    #     else:
    #       wh_loss += self.crit_reg(output['wh'], batch['reg_mask'], batch['ind'], batch['wh']) / opt.num_stacks
    #   if opt.reg_offset and opt.off_weight > 0:  # reg_offset = True , off_weight = 1
    #     off_loss += self.crit_reg(output['reg'], batch['reg_mask'], batch['ind'], batch['reg']) / opt.num_stacks
    #
    # loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
    #        opt.off_weight * off_loss
    # loss_stats = {'loss': loss, 'hm_loss': hm_loss,
    #               'wh_loss': wh_loss, 'off_loss': off_loss}
    # return loss, loss_stats

class CtdetTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
    loss = CtdetLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    dets = ctdet_decode(output['hm'], output['wh'], reg=reg, cat_spec_wh=opt.cat_spec_wh, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.down_ratio
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')
      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.center_thresh:
          #        add_coco_bbox(bbox, cat, conf=1, show_txt=True, img_id='default'):
          debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1], dets[i, k, 4], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          #        add_coco_bbox(bbox, cat, conf=1, show_txt=True, img_id='default'):
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1], dets_gt[i, k, 4], img_id='out_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = ctdet_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
