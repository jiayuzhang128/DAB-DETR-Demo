import torch
from PIL import Image
import cv2
import glob
import os
import argparse

import datasets.transforms as T
from models import build_DABDETR
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

def infer_image(configs):
    print("============Initializing============")
    # 导入模型
    model_config_path = configs.config
    model_checkpoint_path = configs.ckpt

    # 加载配置参数
    args = SLConfig.fromfile(model_config_path) 

    # 创建DAB_DETR模型
    model, _ , postprocessors = build_DABDETR(args)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # 加载数据集
    dataset_val = build_dataset(image_set='val', args=args)
    cocojs = dataset_val.coco.dataset
    id2name = {item['id']: item['name'] for item in cocojs['categories']}

    # 读取图片
    image_path = configs.image_path
    image_names = glob.glob(image_path + "/*.png")
    image_names += glob.glob(image_path + "/*.jpg")
    image_names += glob.glob(image_path + "/*.jpeg")
    num = len(image_names)
    print("%d images"%num)
    print("============start infering============")
    for name in image_names:
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # 预处理
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image, _ = transform(image, None)

        # 模型推理
        output = model(image[None])
        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]))[0]

        thershold = 0.3 # set a thershold

        scores = output['scores']
        labels = output['labels']
        boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
        select_mask = scores > thershold

        box_label = [id2name[int(item)] for item in labels[select_mask]]
        pred_dict = {
            'boxes': boxes[select_mask],
            'size': torch.Tensor([image.shape[1], image.shape[2]]),
            'box_label': box_label,
            'image_id': name.split('/')[-1].split('.')[0]
        }
        os.makedirs(configs.save_path, exist_ok=True)
        vslzr = COCOVisualizer()
        vslzr.visualize(image, pred_dict, savedir=configs.save_path, show_in_console=configs.show_figure, show_save_name=configs.show_savename)
        print("Infer %s success!"%name)
    print("==============Done!==============")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer Image")
    parser.add_argument("--ckpt", type=str, default="./ckpts/DAB_DETR/R50/checkpoint.pth", help="path of checkpoint", required=False)
    parser.add_argument("--config", type=str, default="./ckpts/DAB_DETR/R50/config.json", help="path of configs", required=False)
    parser.add_argument("--image_path", type=str, default="./test/image", help="path of images", required=False)
    parser.add_argument("--save_path", type=str, default="./results/image", help="path of results", required=False)
    parser.add_argument("--show_figure", type=bool, default=False, help="show results or not", required=False)
    parser.add_argument("--show_savename", type=bool, default=False, help="show results name or not", required=False)
    configs = parser.parse_args()
    infer_image(configs)