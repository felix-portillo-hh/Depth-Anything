import argparse
import cv2
import os
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import datetime


THRESHOLD = 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth-path', type=str)
    parser.add_argument('--original-img-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    args = parser.parse_args()
    
    margin_width = 50
    caption_height = 60
    
    if os.path.isfile(args.depth_path):
        if args.depth_path.endswith('txt'):
            with open(args.depth_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.depth_path]
    else:
        filenames = os.listdir(args.depth_path)
        filenames = [os.path.join(args.depth_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    for filename in tqdm(filenames):
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        total_frames = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
        output_width = frame_width * 2 + margin_width

        filename = os.path.basename(filename)
        original_video = cv2.VideoCapture(args.original_img_path)
        
        output_path = os.path.join(args.outdir, filename[:filename.rfind('.')] + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S") + '_concat_video.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
        split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255

        print(total_frames)

        for fno in range(0, total_frames):
            _, raw_frame = raw_video.read()
            _, produced_frame = original_video.read()
            
            for y in range(0, raw_frame.shape[0]):
                for x in range(0, raw_frame.shape[1]):
                    gray_scale = np.mean(raw_frame[y, x])
                    produced_frame[y, x] = [255, 255, 255] if gray_scale <= THRESHOLD else produced_frame[y, x]
            
            combined_frame = cv2.hconcat([raw_frame, split_region, produced_frame])

            out.write(combined_frame)
            print(fno, " ---frame")

        
        raw_video.release()
        out.release()