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
    # 创建保存路径
    os.makedirs(configs.save_path, exist_ok=True)
    # 读取视频
    video_path = configs.video_path
    video_names = glob.glob(configs.video_path + "/*.mp4")
    num = len(video_names)
    print("%d videos"%num)
    print("============start infering============")
    count = 0
    for name in video_names:
        # 创建暂存路径
        os.makedirs(configs.save_path + "/tmp", exist_ok=True)
        count += 1
        cap = cv2.VideoCapture(name)
        # 获取帧率
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        # 获取分辨率
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 获取帧数
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("第%d个视频信息：帧数：%d，帧率：%.2f，分辨率：（%d,%d)"%(count, frame_count, fps, width, height))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_name = name.split('/')[-1].split('.')[0] + "_infer.avi"
        save_path = configs.save_path + "/" + video_name
        tmp_save_path = configs.save_path + "/tmp/" + video_name
        print("savedir: " + save_path)
        print("tmpsavedir: " + tmp_save_path)
        out = cv2.VideoWriter(filename=tmp_save_path, fourcc=fourcc, fps=fps, frameSize=(width, height), isColor=True)
        # 逐帧检测
        frame_idx = 0
        while True:
            frame_idx += 1
            ret, frame = cap.read()
            if not ret:
                break
            # if (frame_idx % int(fps) == 0) and frame_idx >= 20:
            #     break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            vslzr = COCOVisualizer()
            tmp_frame = vslzr.visualize(image, pred_dict, dpi=160, inches=(8.0, 4.5),savedir=configs.save_path, show_in_console=configs.show_figure, show_save_name=configs.show_savename, video=configs.video)
            # print(tmp_frame.shape)
            if tmp_frame.shape[0] != 720:
                tmp_frame = cv2.resize(tmp_frame, (width, height))
            out.write(tmp_frame)
        cap.release()
        out.release()
        os.rename(tmp_save_path, save_path)
        os.rmdir(configs.save_path+'/tmp')
        print("Infer %s success!"%name)
    print("==============Done!==============")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer Image")
    parser.add_argument("--ckpt", type=str, default="./ckpts/DAB_DETR/R50/checkpoint.pth", help="path of checkpoint", required=False)
    parser.add_argument("--config", type=str, default="./ckpts/DAB_DETR/R50/config.json", help="path of configs", required=False)
    parser.add_argument("--video_path", type=str, default="./test/video", help="path of video", required=False)
    parser.add_argument("--save_path", type=str, default="./results/video", help="path of results", required=False)
    parser.add_argument("--show_figure", type=bool, default=False, help="show results or not", required=False)
    parser.add_argument("--show_savename", type=bool, default=False, help="show results name or not", required=False)
    parser.add_argument("--video", type=bool, default=True, help="infer video or not", required=False)
    configs = parser.parse_args()
    infer_image(configs)