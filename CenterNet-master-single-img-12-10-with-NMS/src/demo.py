# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def draw_img(dropper_list, img):
    im_data = cv2.imread(img)
    h, w, _ = im_data.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    # print(dropper_list)
    for item in dropper_list:
        # objName = item[0]
        objName = "ring"
        scores = item[-1]
        # box = item[2]
        show_tlx = int(item[0]) #
        show_tly = int(item[1]) #
        show_brx = int(item[2]) #
        show_bry = int(item[3]) #
        thick = int((h + w) / 800)
        # cv2.rectangle(im_data,
        #               (box[0], box[1]), (box[2], box[3]),
        #               colors, thick)
        # cv2.rectangle(im_data, (show_x - show_w // 2, show_y - show_h // 2),
        #               (show_x + show_w // 2, show_y + show_h // 2), (0, 255, 0), thick)
        cv2.rectangle(im_data, (show_tlx, show_tly),
                      (show_brx, show_bry), (0, 255, 0), thick)
        mess = '%s: %.3f' % (objName, scores)

        # cv2.putText(im_data, mess, (show_x - show_w // 2, show_y - show_h // 2 - 12),
        #             0, 1e-3 * h, (0, 0, 255), thick // 2)
        cv2.putText(im_data, mess, (show_tlx, show_tly - 4),
                    font, 0.4, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

    # cv2.imshow("test", im_data)
    # cv2.waitKey(0)
    return im_data

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  # opt.debug = max(opt.debug, 1)  # 这一句将debug从初始的0设置为1

  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    while True:
        _, img = cam.read()
        cv2.imshow('input', img)
        ret = detector.run(img)  # 这里的run函数是调用的src/lib/detectors/base_detector.py中的run函数
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:  # 检测文件夹中的图片
    if os.path.isdir(opt.demo):  # 把文件夹里的图片名加入到列表中
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]# 检测单张图片

    for (image_name) in image_names:
      # print("9999999", type(image_name))
      ret = detector.run(image_name)
      #ret = {'results': results, 'tot': tot_time, 'load': load_time,
            # 'pre': pre_time, 'net': net_time, 'dec': dec_time,
            # 'post': post_time, 'merge': merge_time}
      time_str = ''
      for stat in time_stats:  # time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)

      # ==================================
      # add by zengyuan (for vis result)
      res_dict = ret['results']
      res_list = res_dict[14].tolist()  # array数据转换成列表
      bbox_list = []
      for bbox in res_list:
        aspect_ratio = (bbox[3]-bbox[1])/(bbox[2]-bbox[0]) # 得到检测框的横纵比(h/w)
        if bbox[-1] > 0.1 and (aspect_ratio < 5): # 检测框的h/w不能大于5，也就是检测框不能太细
          for i in range(len(bbox) - 1):  # this for-loop transfer coord of bbox from float to int
            bbox[i] = int(bbox[i])
            bbox_list.append(bbox)
      # print(bbox_list, len(bbox_list))
      im_data = draw_img(bbox_list, image_name)
      save_path = os.path.join('/home/zy/zy/CenterNet-master/testimg_result', image_name.split('/')[-1])
      cv2.imwrite(save_path, im_data)
  return ret
if __name__ == '__main__':
  opt = opts().init()
  ret = demo(opt)

