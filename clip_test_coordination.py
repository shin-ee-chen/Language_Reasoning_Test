import json
import os
import clip
import torch
from utils import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path',\
        default='/home/xinyi/datasets/output/co_image_captions.json', \
         action='store_true')

    args = parser.parse_args()

    samples = read_json_file(args.file_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device is {device}")
    clip_model, clip_preprocess = clip.load('ViT-B/32', device)
    ranks = []
    scores = []
    for img_id, item in samples.items():
        img_path = get_img_path(img_id)
        image = show_image(img_path)
        rank, clip_score = rank_captions(image, [item['True1'], item['True2'], item['False1'], item['False2']], \
            clip_model, clip_preprocess, device)
        ranks.append(rank)
        scores.append(scores)
    
    active_acc = 0
    rank_count = {}
    for rank in ranks:
        active_acc += get_accuracy(rank)
        str_rank = [str(r) for r in rank]
        rank_key = ''.join(str_rank)
        if rank_key in rank_count:
            rank_count[rank_key] += 1
        else:
            rank_count[rank_key] = 1

    print(f"Accuracy is {active_acc / len(ranks)}")
    print(rank_count)
    # write_file("active_ranks.txt",ranks)
    # write_file("active_scores.txt",scores)