from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json

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

def draw_img_with_chinese(dropper_list, img, idx):
    im_data = cv2.imread(img)
    h, w, _ = im_data.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    class_name = ['破洞', '水渍_油渍_污渍', '三丝', '结头', '花板跳', '百脚', '毛粒', '粗经', '松经', '断经', '吊经', '粗维', '纬缩', '浆斑', '整经结',
                  '星跳_跳花', '断氨纶', '稀密档_浪纹档_色差档', '磨痕_轧痕_修痕_烧毛痕', '死皱_云织_双维_双经_跳纱_筘路_纬纱不良']
    # print(dropper_list)
    for item in dropper_list:
        # objName = item[0]
        objName = class_name[idx-1]
        scores = item[-1]
        # box = item[2]
        show_tlx = int(item[0]) #
        show_tly = int(item[1]) #
        show_brx = int(item[2]) #
        show_bry = int(item[3]) #
        thick = int((h + w) / 800)
        cv2.rectangle(im_data, (show_tlx, show_tly),
                      (show_brx, show_bry), (0, 255, 0), thick)
        mess = '%s: %.3f' % (objName, scores)
        # ======================
        cv2img = cv2.cvtColor(im_data, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
        pilimg = Image.fromarray(cv2img)
        # PIL图片上打印汉字
        draw = ImageDraw.Draw(pilimg)  # 图片上打印
        font = ImageFont.truetype("/usr/share/fonts/truetype/arphic/uming.ttc", 30, encoding="utf-8")
        draw.text((show_tlx, show_tly-35), mess, (255, 255, 255), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
        # PIL图片转cv2 图片
        im_data = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
        # =======================
        # cv2.putText(im_data, mess, (show_x - show_w // 2, show_y - show_h // 2 - 12),
        #             0, 1e-3 * h, (0, 0, 255), thick // 2)
        # cv2.putText(im_data, mess, (show_tlx, show_tly - 4),
        #             font, 0.4, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

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

        cnt_img = 0
        result = [] # 保存最后的json文件的列表
        for (image_name) in image_names:
            cnt_img += 1
            print('cnt:', cnt_img, "image_name:", image_name)
            ret = detector.run(image_name)
            # print(ret['results'])
            #ret = {'results': results, 'tot': tot_time, 'load': load_time,
                # 'pre': pre_time, 'net': net_time, 'dec': dec_time,
                # 'post': post_time, 'merge': merge_time}
            time_str = ''
            for stat in time_stats:  # time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)

            # ==================================
            # add by zengyuan (for vis result)
            class_num = 20
            for idx in range(1,class_num+1):
                res_dict = ret['results']
                # print(res_dict)
                res_list = res_dict[idx].tolist()  # array数据转换成列表  # 改变识别类别！！！！！！！！！
                bbox_list = []
                for bbox in res_list:
                    # aspect_ratio = (bbox[3]-bbox[1])/(bbox[2]-bbox[0]) # 得到检测框的横纵比(h/w)
                    if bbox[-1] > 0.2: # 检测框的h/w不能大于5，也就是检测框不能太细
                        for i in range(len(bbox) - 1):  # this for-loop transfer coord of bbox from float to int
                            bbox[i] = round(max(bbox[i],1), 2)
                        print(bbox)
                        bbox_list.append(bbox)
                        result.append({'name': image_name.split('/')[-1], 'category': idx,
                                       'bbox': bbox[:-1], 'score': bbox[-1]})
                    # if idx == 1 :
                    #     im_data = draw_img_with_chinese(bbox_list, image_name, idx)
                    #     save_path = os.path.join('/home/xidian/zy/7tianchi/CenterNet_for_tianchi/testimg_result',
                    #                             image_name.split('/')[-1])
                    #     cv2.imwrite(save_path, im_data)
                    # else:
                    #     im_data = draw_img_with_chinese(bbox_list, save_path, idx)
                    #     save_path = os.path.join('/home/xidian/zy/7tianchi/CenterNet_for_tianchi/testimg_result',
                    #                             image_name.split('/')[-1])
                    #     cv2.imwrite(save_path, im_data)

                # print(bbox_list, len(bbox_list))
                # ===================================
                # for bbox in res_list:
                #   aspect_ratio = (bbox[3]-bbox[1])/(bbox[2]-bbox[0]) # 得到检测框的横纵比(h/w)
                #   if bbox[-1] > 0.1 and (aspect_ratio < 5): # 检测框的h/w不能大于5，也就是检测框不能太细
                #     for i in range(len(bbox) - 1):  # this for-loop transfer coord of bbox from float to int
                #       bbox[i] = int(bbox[i])
                #       bbox_list.append(bbox)
                # print(bbox_list, len(bbox_list))
                # im_data = draw_img_with_chinese(bbox_list, image_name, idx)
                # ===================================
                # save_path = os.path.join('/home/xidian/zy/7tianchi/CenterNet_for_tianchi/testimg_result', image_name.split('/')[-1])
                # cv2.imwrite(save_path, im_data)
        with open('result.json', 'w') as fp:
            json.dump(result, fp, indent=4, separators=(',', ': '))
        return ret
if __name__ == '__main__':
    opt = opts().init()
    ret = demo(opt)

