import argparse
from fastsam import FastSAM, FastSAMPrompt 
import ast
import torch
from utils.tools import convert_box_xywh_to_xyxy
import cv2
from mayfly.videocapture import VideoCapture
import matplotlib.pyplot as plt
import time
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM.pt", help="model"
    )
    parser.add_argument(
        "--img_path", type=str, default="./images/dogs.jpg", help="path to image file"
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--text_prompt", type=str, default=None, help='use text prompt eg: "a dog"'
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./output/", help="image save path"
    )
    parser.add_argument(
        "--randomcolor", type=bool, default=True, help="mask random color"
    )
    parser.add_argument(
        "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument("--box_prompt", type=str, default="[[0,0,0,0]]", help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes")
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )
    return parser.parse_args()

def fast_show_mask_gpu_blend(image, annotation):
    image = torch.from_numpy(image).float().to(annotation.device)
    msak_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    areas = torch.sum(annotation, dim=(1, 2))
    sorted_indices = torch.argsort(areas, descending=False)
    annotation = annotation[sorted_indices]
    # Find the index of the first non-zero value at each position.
    index = (annotation != 0).to(torch.long).argmax(dim=0)
    color = torch.rand((msak_sum, 1, 1, 3)).to(annotation.device)
    transparency = torch.ones((msak_sum, 1, 1, 1)).to(annotation.device) * 0.6
    visual = torch.cat([color, transparency], dim=-1)
    mask_image = torch.unsqueeze(annotation, -1) * visual
    # Select data according to the index. The index indicates which batch's data to choose at each position, converting the mask_image into a single batch form.
    show = torch.zeros((height, weight, 4)).to(annotation.device)
    h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight), indexing='ij')
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
    # Use vectorized indexing to update the values of 'show'.
    show[h_indices, w_indices, :] = mask_image[indices]

    # blend original image with segmentation masks on GPU
    foreground = show[:,:,0:3]
    background = image/255.
    alpha = show[:,:,3]
    blended = 0.5*foreground + 0.5*background
    mask = torch.zeros(image.shape, device=annotation.device)
    mask[:,:,0] = alpha
    mask[:,:,1] = alpha
    mask[:,:,2] = alpha
    show = torch.where(mask>0.01, blended, background)
    show_cpu = show.cpu().numpy()
    return show_cpu

def main(args):
    # load model
    model = FastSAM(args.model_path)
    args.point_prompt = ast.literal_eval(args.point_prompt)
    args.box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(args.box_prompt))
    args.point_label = ast.literal_eval(args.point_label)

    use_cuda=True
    config = ''
    cap = VideoCapture(list(range(64)), config)
    path = './'
    frame_idx = -1
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter('output.mp4', fourcc, 10, (1280,720))
    while True:
        frames = cap.read()
        frame_idx += 1
        frm = frames[0] # It may have more than 1 frame if sync cameras or ToF. We assume 1 frame
        if use_cuda:
            arr = torch.from_dlpack(frm['image']).cpu().numpy()
        else:
            arr = np.from_dlpack(frm['image']).copy()
        img = cv2.cvtColor(arr[:,:,:3], cv2.COLOR_RGB2BGR)
        everything_results = model(
            img,
            device=args.device,
            retina_masks=args.retina,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            )

        t0 = time.time()
        bboxes = None
        points = None
        point_label = None
        #prompt_process = FastSAMPrompt(args.img_path, everything_results, device=args.device)
        prompt_process = FastSAMPrompt.from_np_tensor(img, everything_results, device=args.device)
        if args.box_prompt[0][2] != 0 and args.box_prompt[0][3] != 0:
                ann = prompt_process.box_prompt(bboxes=args.box_prompt)
                bboxes = args.box_prompt
        elif args.text_prompt != None:
            ann = prompt_process.text_prompt(text=args.text_prompt)
        elif args.point_prompt[0] != [0, 0]:
            ann = prompt_process.point_prompt(
                points=args.point_prompt, pointlabel=args.point_label
            )
            points = args.point_prompt
            point_label = args.point_label
        else:
            ann = prompt_process.everything_prompt()
        print('Time (ms) prompt: ',1000.0*(time.time()-t0))

        t0 = time.time()
        res = fast_show_mask_gpu_blend(img, ann)
        res = (res*255.).astype(np.uint8)
        print('Time (ms) plot: ',1000.0*(time.time()-t0))
        cv2.imshow('FastSAM',res)
        cv2.waitKey(1)
        output.write(res)

if __name__ == "__main__":
    args = parse_args()
    main(args)
