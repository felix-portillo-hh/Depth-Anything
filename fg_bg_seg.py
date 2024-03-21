from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from PIL import Image

import numpy as np
import argparse
import os
from tqdm import tqdm
import cv2
import torch
from torchvision.transforms import Compose
import torchvision.models.segmentation as segmentation
import torch.nn.functional as F
from torchvision import transforms
from copy import deepcopy

THRESHOLD = 100

def build_model():
    encoder = 'vits' # can also be 'vitb' or 'vitl'
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).eval()

    return depth_anything

def build_transformer():
    return Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

def process_image_for_model(image, transformer):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = transformer({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0)

    return image

def segment_image(depth_map, original_image):

    model = segmentation.lraspp_mobilenet_v3_large(pretrained=True)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((513, 513)),  # DeepLabV3 expects input size (513, 513)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    _, thresholded_image = cv2.threshold(depth_map, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    produced_image = original_image

    '''
        REPLACES:
        for y in range(0, depth_map.shape[0]):
            for x in range(0, depth_map.shape[1]):
                produced_image[y, x] = [255, 255, 255] if refined_mask[y, x] <= THRESHOLD else original_image[y, x]
    '''
    refined_mask = cv2.bitwise_and(depth_map, thresholded_image)
    threshold_mask = refined_mask <= THRESHOLD
    produced_image[threshold_mask] = [255, 255, 255]
    produced_image[~threshold_mask] = original_image[~threshold_mask]

    seg_image = cv2.cvtColor(produced_image, cv2.COLOR_BGR2RGB)

    # Preprocess the input image
    input_image = transform(seg_image)
    input_image = input_image.unsqueeze(0)  # Add batch dimension

    # Perform semantic segmentation to extract person from image
    with torch.no_grad():
        output = model(input_image)['out'][0]
        output_predictions = output.argmax(0)

    person_mask = (output_predictions == 15).cpu().numpy().astype(np.uint8)
    person_mask_resized = cv2.resize(person_mask, (produced_image.shape[1], produced_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    produced_image[person_mask_resized] = [255, 255, 255] 
    produced_image[~person_mask_resized] = original_image[~person_mask_resized]

    contours, _ = cv2.findContours(person_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    return refined_mask, produced_image, [x,y,w,h]


def process_image(input_image, filename, blur=True):
    original_image = deepcopy(input_image)

    h, w = input_image.shape[:2]

    processed_img = process_image_for_model(input_image, transformer)
    
    # depth shape: 1xHxW
    with torch.no_grad():
        depth_map = depth_anything(processed_img)

    depth_map = F.interpolate(depth_map[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
    
    depth_map = depth_map.cpu().numpy().astype(np.uint8)

    image = cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB) / 255.0
    image = image.astype(np.float64)

    print(depth_map.shape[0])
    print(depth_map.shape[1])
    
    refined_mask, produced_image, bbox = segment_image(depth_map, input_image)

    cv2.rectangle(original_image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

    if blur:
        blurred = blur_image(original_image, produced_image)
        blurred.save(os.path.join(args.outdir, os.path.basename(filename[:filename.rfind('.')]) + '_blurred_img.png'))

    cv2.imwrite(os.path.join(args.outdir, os.path.basename(filename[:filename.rfind('.')]) + '_heatmap_img.png'), refined_mask)
    cv2.imwrite(os.path.join(args.outdir, os.path.basename(filename[:filename.rfind('.')]) + '_produced_img.png'), produced_image)
    cv2.imwrite(os.path.join(args.outdir, os.path.basename(filename[:filename.rfind('.')]) + '_bbox_img.png'), original_image)

def blur_image(original_image, produced_image):
    opencv_blurred = cv2.GaussianBlur(original_image, (99,99), 0)
    blurred_rgb = cv2.cvtColor(opencv_blurred, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.cvtColor(produced_image, cv2.COLOR_BGR2RGB)

    # Convert the OpenCV image to a PIL Image
    mask = Image.fromarray(mask_rgb)
    blurred = Image.fromarray(blurred_rgb)

    mask = mask.convert("RGBA")
    datas = mask.getdata()
    newData = []
    #make white background transparent
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    mask.putdata(newData)
    blurred.paste(mask, (0,0), mask)

    return blurred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--depth-path', type=str)
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    args = parser.parse_args()

    depth_anything = build_model()
    transformer = build_transformer()

    margin_width = 50
    caption_height = 60
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    if os.path.isfile(args.input_path):
        if args.input_path.endswith('txt'):
            with open(args.input_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.input_path]
    else:
        filenames = os.listdir(args.input_path)
        filenames = [os.path.join(args.input_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    
    os.makedirs(args.outdir, exist_ok=True)

    transformer = build_transformer()
    
    for filename in tqdm(filenames):
        input_image = cv2.imread(filename)

        process_image(input_image, filename) # Call this for single images or single frames in a video

        



        