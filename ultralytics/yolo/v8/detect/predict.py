# Ultralytics YOLO 🚀, GPL-3.0 license

import os
import hydra
import torch
import json
import cv2

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box, save_img


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        # save_path = str(self.save_dir / p.name)  # im.jpg

        """4-26 gai"""
        # self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        self.txt_path = "/home/su/su/yolo/yolov8_tracking-master_main/yolov8/runs/detect/total"

        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        """4-26"""
        # with open(f'{self.txt_path}.txt', 'a') as f:
        #         f.write(str(self.data_path) + '\n')
        dict1 = {}
        dict1['img_path'] = str(self.data_path) 
        dict1['label'] = {}

        for *xyxy, conf, cls in reversed(det):

            

            rate = 1.1     # 添加的置信度比例，设为0.7
            list1 = [0,32,38]      # 标签类别 [0,32,38] ,0代表人，32代表球，38代表网球拍

            # import random
            # random_num = random.randint(1, 3)
            self.args.save_conf = True
         

            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                # line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format

                line = (cls, conf) if self.args.save_conf else (cls, *xywh) 
                # with open(f'{self.txt_path}.txt', 'a') as f:
                #         f.write(('%g ' * len(line)).rstrip() % line + '\n')


                if cls in list1 and conf < rate : 
   
                    dict1['label'][line[0].item()] = line[1].item()

                    # with open(f'{self.txt_path}.txt', 'a') as f:
                    #     # f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    #     f.write(json.dumps(dict1) + '\n')

                    # if random_num == 1:
                    # with open(f'{self.txt_path}.txt', 'a') as f:
                    #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                # label = None if self.args.hide_labels else (
                #     self.model.names[c] if self.args.hide_conf else f'{sewm lf.model.names[c]} {conf:.2f}')
                # self.annotator.box_label(xyxy, label, color=colors(c, True))

                if cls in list1 and conf < rate : 
                    label = None if self.args.hide_labels else (
                        self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                    # self.annotator.box_label(xyxy, label, color=colors(c, True))

            if self.args.save_crop:
                imc = im0.copy()
                # save_one_box(xyxy,
                #              imc,
                #              file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.dajunta_path.stem}.jpg',
                #              BGR=True)
                if cls in list1 and conf < rate :
                    str1 = self.txt_path.split('/')[-1]
                    
                    # if random_num == 1:
                    save_img(imc,
                        file=self.save_dir / 'crops' / f'{str1}.jpg',
                        BGR=True)
        with open(f'{self.txt_path}.txt', 'a') as f:
        # f.write(('%g ' * len(line)).rstrip() % line + '\n')
            f.write(json.dumps(dict1) + '\n')
        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "best.pt"
    # cfg.model = cfg.model or "/home/su/su/yolo/yolov8_tracking-master/yolov8/runs/detect/v8_200/weights/best.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source or ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()
