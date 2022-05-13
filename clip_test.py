import json
import os
import clip
import torch
from utils import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path',type = str,\
        default='/home/xinyi/Language_Reasoning_Test/datasets/dataset_final/active_passive_captions_gruen_strict.json', \
        help = "path of caption input")

    args = parser.parse_args()

    items = read_json_file(args.file_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device is {device}")
    clip_model, clip_preprocess = clip.load('ViT-B/32', device)
    # ranks = []
    # scores = []
    clip_results = []

    for item in items:
        image_id = item['image_id']
        captions = item['caption_group'][0]
        img_path = get_img_path(image_id)
        image = show_image(img_path)

        rank, clip_score = rank_captions(image, [captions['True1'],captions['True2'], \
            captions['False1'], captions['False2']], clip_model, clip_preprocess, device)
        if "predicate" in captions:
            clip_results.append({"image_id": image_id, "rank": rank, \
                "predicate": captions["predicate"]})
        else:
            clip_results.append({"image_id": image_id, "rank": rank})
    
    print(args.file_path)
    dir = "/home/xinyi/Language_Reasoning_Test/output/clip_results"
    if not os.path.exists(dir):
        os.makedirs(dir)
        
    out_file = os.path.join(dir, \
        args.file_path.split('/')[-1][:-5] + "_clip_results"+".json")
    write_file(out_file, clip_results)
    print("finish\n")
    # write_file("active_scores.txt", scores)