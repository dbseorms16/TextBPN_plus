import os
import time
import cv2
import numpy as np
import json
from shapely.geometry import *
import torch
import subprocess
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from dataset import TotalText, Ctw1500Text, Icdar15Text, Mlt2017Text, TD500Text, \
    ArtText, ArtTextJson, Mlt2019Text, Ctw1500Text_New, TotalText_New, CustomText
from network.textnet import TextNet
from cfglib.config import config as cfg, update_config, print_config
from cfglib.option import BaseOptions
from util.augmentation import BaseTransform
from util.visualize import visualize_detection, visualize_gt
from util.misc import to_device, mkdirs,rescale_result
from util.eval import deal_eval_total_text, deal_eval_ctw1500, deal_eval_icdar15, \
    deal_eval_TD500, data_transfer_ICDAR, data_transfer_TD500, data_transfer_Custom, data_transfer_MLT2017, deal_eval_custom

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)


def osmkdir(out_dir):
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)


def write_to_file(contours, file_path, gt=False):
    """
    :param contours: [[x1, y1], [x2, y2]... [xn, yn]]
    :param file_path: target file path
    """
    # according to total-text evaluation method, output file shoud be formatted to: y0,x0, ..... yn,xn
    with open(file_path, 'w') as f:
        for cont in contours:
            cont = np.stack([cont[:, 0], cont[:, 1]], 1)
            if cv2.contourArea(cont) <= 0:
                continue
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            if gt:
                f.write(cont +',text'+'\n')
            else:
                f.write(cont +'\n')
                

def inference(model, test_loader, output_dir):

    total_time = 0.
    if cfg.exp_name != "MLT2017" and cfg.exp_name != "ArT":
        osmkdir(output_dir)
    else:
        if not os.path.exists(output_dir):
            mkdirs(output_dir)
        if cfg.exp_name == "MLT2017":
            out_dir = os.path.join(output_dir, "{}_{}_{}_{}_{}".
                                   format(str(cfg.checkepoch), cfg.test_size[0],
                                          cfg.test_size[1], cfg.dis_threshold, cfg.cls_threshold))
            if not os.path.exists(out_dir):
                mkdirs(out_dir)

    art_results = dict()
    for i, (image, meta) in enumerate(test_loader):
        input_dict = dict()
        idx = 0  # test mode can only run with batch_size == 1
        H, W = meta['Height'][idx].item(), meta['Width'][idx].item()
        print(meta['image_id'], (H, W))

        input_dict['img'] = to_device(image)

        # get detection result
        start = time.time()
        output_dict = model(input_dict)
        torch.cuda.synchronize()
        end = time.time()
        if i > 0:
            total_time += end - start
            fps = (i + 1) / total_time
        else:
            fps = 0.0

        print('detect {} / {} images: {}. ({:.2f} fps)'.
              format(i + 1, len(test_loader), meta['image_id'][idx], fps))

        # visualization
        img_show = image[idx].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

        if cfg.viz:
        # if True:
            gt_contour = []
            label_tag = meta['label_tag'][idx].int().cpu().numpy()
            for annot, n_annot in zip(meta['annotation'][idx], meta['n_annotation'][idx]):
                if n_annot.item() > 0:
                    gt_contour.append(annot[:n_annot].int().cpu().numpy())

            gt_vis = visualize_gt(img_show, gt_contour, label_tag)
            show_boundary, heat_map = visualize_detection(img_show, output_dict, meta=meta)
            # heat_map[:,:320,1]
            # print(heat_map.min(), heat_map.max())
            # print(heat_map)
            heat_map = np.where(heat_map > 127.5, 255, 0)
            show_boundary[:,:320,0] = show_boundary[:,:320,0] + heat_map[:,:320,1]
            show_boundary[:,:320,1] = show_boundary[:,:320,1] + heat_map[:,:320,1]
            show_boundary[:,:320,2] = show_boundary[:,:320,2] + heat_map[:,:320,1]
            
            show_boundary[:,320:640,0] = show_boundary[:,320:640,0] + heat_map[:,:320,1]
            show_boundary[:,320:640,1] = show_boundary[:,320:640,1] + heat_map[:,:320,1]
            show_boundary[:,320:640,2] = show_boundary[:,320:640,2] + heat_map[:,:320,1]
            
            
            show_boundary[:,640:,0] = show_boundary[:,640:,0] + heat_map[:,:320,1]
            show_boundary[:,640:,1] = show_boundary[:,640:,1] + heat_map[:,:320,1]
            show_boundary[:,640:,2] = show_boundary[:,640:,2] + heat_map[:,:320,1]
            # print(heat_map[:,:,0].shape)
            # print(heat_map[:,:320,0].shape)
            show_map = np.concatenate([heat_map, gt_vis], axis=1)
            show_map = cv2.resize(show_map, (320 * 3, 320))
            im_vis = np.concatenate([show_map, show_boundary], axis=0)
            path = os.path.join(cfg.vis_dir, '{}_{}_{}_test'.format(cfg.iter, cfg.exp_name, cfg.num_poly), meta['image_id'][idx].split(".")[0]+".jpg")
            m_path = os.path.join(cfg.vis_dir, '{}_{}_{}_test'.format(cfg.iter, cfg.exp_name, cfg.num_poly), meta['image_id'][idx].split(".")[0]+"_m.jpg")
            cv2.imwrite(path, im_vis)
            cv2.imwrite(m_path, heat_map[:,:320,1])

        contours = output_dict["py_preds"][-1].int().cpu().numpy()
        img_show, contours = rescale_result(img_show, contours, H, W)
        _, gt_contour = rescale_result(img_show, gt_contour, H, W)
        ##여기도 해야하나
        
        # path = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name), meta['image_id'][idx].split(".")[0] + ".jpg")
        # im_show = img_show.copy()
        # im_show = np.ascontiguousarray(im_show[:, :, ::-1])
        # cv2.drawContours(im_show, [gt_contour[i] for i, tag in enumerate(label_tag) if tag >0], -1, (0, 255, 0), 4)
        # cv2.drawContours(im_show, contours, -1, (0, 0, 255), 2)
        # cv2.imwrite(path, im_show)

        # empty GPU  cache
        torch.cuda.empty_cache()

        # write to file
        if cfg.exp_name == "Icdar2015":
            fname = "res_" + meta['image_id'][idx].replace('jpg', 'txt')
            contours = data_transfer_ICDAR(contours)
            write_to_file(contours, os.path.join(output_dir, fname))
        elif cfg.exp_name == "MLT2017":
            fname = meta['image_id'][idx].split("/")[-1].replace('ts', 'res')
            fname = fname.split(".")[0] + ".txt"
            data_transfer_MLT2017(contours, os.path.join(out_dir, fname))
        elif cfg.exp_name == "TD500":
            fname = "res_" + meta['image_id'][idx].split(".")[0]+".txt"
            data_transfer_TD500(contours, os.path.join(output_dir, fname))
        elif cfg.exp_name == "Custom":
            
            p_contours = np.array(contours, np.int32)
            p_gt_contour = np.array(gt_contour, np.int32)
            
            fname = "res_img_" + meta['image_id'][idx].replace('png', 'txt')
            contours = data_transfer_Custom(contours)
            write_to_file(contours, os.path.join(output_dir, fname))
            
            if not os.path.exists(os.path.join(output_dir,'gt')):
                mkdirs(os.path.join(output_dir,'gt'))
            fname = "gt_img_" + meta['image_id'][idx].replace('png', 'txt')
            gt_contour = data_transfer_Custom(gt_contour)
            # print(gt_contour)

            # w, h, _ = img_show.shape            
            # tr_mask = np.zeros((h, w, 1), np.uint8)
            # gt_mask = np.zeros((h, w, 1), np.uint8)

            # print(p_contours)
            # print(p_gt_contour)
            # p_contours = cv2.fillPoly(tr_mask, [p_contours], (255,255,255)) // 255
            # p_gt_contour = cv2.fillPoly(gt_mask, [p_gt_contour], (255,255,255)) // 255

            # union = p_contours + p_gt_contour
            # uni = np.clip(union, 0,1)
            
            # area = np.clip(union, 1,2) - 1
            # print(uni.sum(), area.sum())

            write_to_file(gt_contour, os.path.join(output_dir,'gt', fname), gt=True)
            
            
        elif cfg.exp_name == "ArT":
            fname = meta['image_id'][idx].split(".")[0].replace('gt', 'res')
            art_result = []
            for j in range(len(contours)):
                art_res = dict()
                S = cv2.contourArea(contours[j], oriented=True)
                if S < 0:
                    art_res['points'] = contours[j].tolist()[::-1]
                else:
                    print((meta['image_id'], S))
                    continue
                art_res['confidence'] = float(output_dict['confidences'][j])
                art_result.append(art_res)
            art_results[fname] = art_result
            # print(art_results)
        else:
            fname = meta['image_id'][idx].replace('jpg', 'txt')
            write_to_file(contours, os.path.join(output_dir, fname))
    if cfg.exp_name == "ArT":
        with open(output_dir + '/art_test_{}_{}_{}_{}_{}.json'.
                format(cfg.checkepoch, cfg.test_size[0], cfg.test_size[1],
                       cfg.dis_threshold, cfg.cls_threshold), 'w') as f:
            json.dump(art_results, f)
    elif cfg.exp_name == "MLT2017":
        father_path = "{}_{}_{}_{}_{}".format(str(cfg.checkepoch), cfg.test_size[0],
                                              cfg.test_size[1], cfg.dis_threshold, cfg.cls_threshold)
        subprocess.call(['sh', './output/MLT2017/eval_zip.sh', father_path])
        pass


def main(vis_dir_path):

    osmkdir(vis_dir_path)
    if cfg.exp_name == "Totaltext":
        testset = TotalText(
            data_root='data/total-text-mat',
            ignore_list=None,
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )

    elif cfg.exp_name == "Ctw1500":
        testset = Ctw1500Text(
            data_root='data/ctw1500',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    elif cfg.exp_name == "Icdar2015":
        testset = Icdar15Text(
            data_root='data/Icdar2015',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    elif cfg.exp_name == "MLT2017":
        testset = Mlt2017Text(
            data_root='data/MLT2017',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    elif cfg.exp_name == "TD500":
        testset = TD500Text(
            data_root='data/TD500',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    elif cfg.exp_name == "ArT":
        testset = ArtTextJson(
            data_root='data/ArT',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        )
    elif cfg.exp_name == 'Custom':
        testset = CustomText(
            data_root='data/Custom_data',
            is_training=False,
            transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds),
            cfg=cfg
        )
    else:
        print("{} is not justify".format(cfg.exp_name))

    if cfg.cuda:
        cudnn.benchmark = True

    # Data
    test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

    # Model
    model = TextNet(is_training=False, iteration=cfg.iter, backbone=cfg.net)
    model_path = os.path.join(cfg.save_dir, cfg.exp_name,
                              'TextBPN_{}_{}_{}_{}.pth'.format(cfg.iter, cfg.num_poly, model.backbone_name, cfg.checkepoch))
                            #   'TextBPN_{}_{}_{}.pth'.format(cfg.num_poly, model.backbone_name, cfg.checkepoch))

    model.load_model(model_path)
    model = model.to(cfg.device)  # copy to cuda
    model.eval()
    with torch.no_grad():
        print('Start testing TextBPN++.')
        output_dir = os.path.join(cfg.output_dir, cfg.exp_name)
        inference(model, test_loader, output_dir)

    if cfg.exp_name == "Totaltext":
        deal_eval_total_text(debug=True)

    elif cfg.exp_name == "Ctw1500":
        deal_eval_ctw1500(debug=True)
    elif cfg.exp_name == "Custom":
        deal_eval_custom(debug=True)
    elif cfg.exp_name == "Icdar2015":
        deal_eval_icdar15(debug=True)
    elif cfg.exp_name == "TD500":
        deal_eval_TD500(debug=True)
    else:
        print("{} is not justify".format(cfg.exp_name))


if __name__ == "__main__":
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    vis_dir = os.path.join(cfg.vis_dir, '{}_{}_{}_test'.format(cfg.iter, cfg.exp_name, cfg.num_poly))

    if not os.path.exists(vis_dir):
        mkdirs(vis_dir)
    # main
    main(vis_dir)