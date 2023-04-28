#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = '古溪'
import os
import numpy as np
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
from util.io import read_lines
import cv2
from util.augmentation import *

class CustomText(TextDataset):
    def __init__(self, data_root, cfg=None, is_training=True, load_memory=False, transform=None, ignore_list=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.load_memory = load_memory

        self.image_root = os.path.join(data_root, 'train' if is_training else 'test')
        self.back_root =  os.path.join(data_root, 'train_back' if is_training else 'test_back')
        # self.annotation_root = os.path.join(data_root, 'train' if is_training else 'test', "text_label_circum")
        self.image_list = os.listdir(self.image_root)
        self.back_image_list = os.listdir(self.back_root)
        self.annotation_list = ['{}'.format(img_name.replace('.jpg', '')) for img_name in self.image_list]
        # self.augmentation = Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)

        if self.load_memory:
            self.datas = list()
            for item in range(len(self.image_list)):
                self.datas.append(self.load_img(item))


    def load_img(self, img_root, image_id):
        image_path = os.path.join(img_root, image_id)

        # Read image data
        image = Image.open(image_path)
        image = image.convert('RGBA')
        data = dict()
        data["image"] = image
        # data["polygons"] = polygons
        data["image_id"] = image_id.split("/")[-1]
        data["image_path"] = image_path

        return data

    def __getitem__(self, item):

        if self.load_memory:
            data = self.datas[item]
        else:
            image_id = self.image_list[item]
            data = self.load_img(self.image_root, image_id)
            
            random_id = random.randint(0, len(self.back_image_list))
            random_id = self.back_image_list[random_id]
            back_data = self.load_img(self.back_root, random_id)
            
        image, polygons = perform_operation(data['image'], back_data['image'], magnitude=0.1)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        polygons = self.polygons_converter(polygons, num_poly=4)
        
        if self.is_training:
            return self.get_training_data(image, polygons, 
                                          image_id=data["image_id"], image_path=data["image_path"])
            
            # return self.get_training_data(data["image"], data["polygons"],
            #                               image_id=data["image_id"], image_path=data["image_path"])
        else:
            image, meta = self.get_test_data_only_image(image, polygons, image_id=data["image_id"], image_path=data["image_path"])
            return image, meta

    def __len__(self):
        return len(self.image_list)
    
    def polygons_converter(self, polygon, num_poly=None):
        
        polygons = []
        new = []
        for index, (x,y) in enumerate(polygon):
            pts = np.array([x, y]).astype(np.int32)
            new.append(pts)
            
            # if num_poly and index > 0 :
            #     for ms in range(2, num_poly+1):
                    
            #         clone_x = int((oldx + x) // ms)
            #         clone_y = int((oldy + y) // ms)
            #         for m in range(1, ms):
            #         clone_pts = np.array([clone_x, clone_y]).astype(np.int32)
            #         new.append(clone_pts)
                    
            # oldx, oldy = x, y 
            
        polygons.append(TextInstance(new, 'c', "**"))
     
        return polygons

    def polygon_clone(self, polygon):
    
        return polygon
    
if __name__ == '__main__':
    from util.augmentation import Augmentation
    from util.misc import regularize_sin_cos
    from util.pbox import bbox_transfor_inv, minConnectPath
    from util import canvas as cav
    import time

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=640, mean=means, std=stds
    )

    trainset = Ctw1500Text(
        data_root='../data/ctw1500',
        is_training=True,
        transform=transform
    )

    # img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[944]
    for idx in range(0, len(trainset)):
        t0 = time.time()
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi = trainset[idx]
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi \
            = map(lambda x: x.cpu().numpy(), (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi))

        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)
        print(idx, img.shape)
        top_map = radius_map[:, :, 0]
        bot_map = radius_map[:, :, 1]

        print(radius_map.shape)

        sin_map, cos_map = regularize_sin_cos(sin_map, cos_map)
        ret, labels = cv2.connectedComponents(tcl_mask[:, :, 0].astype(np.uint8), connectivity=8)
        cv2.imshow("labels0", cav.heatmap(np.array(labels * 255 / np.max(labels), dtype=np.uint8)))
        print(np.sum(tcl_mask[:, :, 1]))

        t0 = time.time()
        for bbox_idx in range(1, ret):
            bbox_mask = labels == bbox_idx
            text_map = tcl_mask[:, :, 0] * bbox_mask

            boxes = bbox_transfor_inv(radius_map, sin_map, cos_map, text_map, wclip=(2, 8))
            # nms
            boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), 0.25)
            boxes = boxes[:, :8].reshape((-1, 4, 2)).astype(np.int32)
            if boxes.shape[0] > 1:
                center = np.mean(boxes, axis=1).astype(np.int32).tolist()
                paths, routes_path = minConnectPath(center)
                boxes = boxes[routes_path]
                top = np.mean(boxes[:, 0:2, :], axis=1).astype(np.int32).tolist()
                bot = np.mean(boxes[:, 2:4, :], axis=1).astype(np.int32).tolist()

                boundary_point = top + bot[::-1]
                # for index in routes:

                for ip, pp in enumerate(top):
                    if ip == 0:
                        color = (0, 255, 255)
                    elif ip == len(top) - 1:
                        color = (255, 255, 0)
                    else:
                        color = (0, 0, 255)
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, color, -1)
                for ip, pp in enumerate(bot):
                    if ip == 0:
                        color = (0, 255, 255)
                    elif ip == len(top) - 1:
                        color = (255, 255, 0)
                    else:
                        color = (0, 255, 0)
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, color, -1)
                cv2.drawContours(img, [np.array(boundary_point)], -1, (0, 255, 255), 1)
        # print("nms time: {}".format(time.time() - t0))
        # # cv2.imshow("", img)
        # # cv2.waitKey(0)

        # print(meta["image_id"])
        cv2.imshow('imgs', img)
        cv2.imshow("", cav.heatmap(np.array(labels * 255 / np.max(labels), dtype=np.uint8)))
        cv2.imshow("tr_mask", cav.heatmap(np.array(tr_mask * 255 / np.max(tr_mask), dtype=np.uint8)))
        cv2.imshow("tcl_mask",
                   cav.heatmap(np.array(tcl_mask[:, :, 1] * 255 / np.max(tcl_mask[:, :, 1]), dtype=np.uint8)))
        # cv2.imshow("top_map", cav.heatmap(np.array(top_map * 255 / np.max(top_map), dtype=np.uint8)))
        # cv2.imshow("bot_map", cav.heatmap(np.array(bot_map * 255 / np.max(bot_map), dtype=np.uint8)))
        cv2.waitKey(0)