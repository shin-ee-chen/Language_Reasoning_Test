import json
import os
import clip
import torch
from utils import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path',\
        default='~/datasets/output/active_passive_captions.txt', \
         action='store_true')

    args = parser.parse_args()

    JSON_FILE = args.file_path
    items = read_json_file(JSON_FILE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device is {device}")
    clip_model, clip_preprocess = clip.load('ViT-B/32', device)
    ranks = []
    scores = []
    for item in items:
        img_path = get_img_path(item['image_id'])
        image = show_image(img_path)
        rank, clip_score = rank_captions(image, [item['True1'], item['True2'], item['False1'], item['False2']], \
            clip_model, clip_preprocess, device)
        ranks.append(rank)
        scores.append(scores)
    
    active_acc = 0
    for rank in ranks:
        active_acc += get_accuracy(rank)
    print(f"Accuracy is {active_acc / len(ranks)}")
    write_file("active_ranks.txt",ranks)
    write_file("active_scores.txt",scores)