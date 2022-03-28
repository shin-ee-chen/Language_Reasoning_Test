import json
import os
import clip
import torch
from utils import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path',type = str,\
        default='/home/xinyi/Language_Reasoning_Test/input/test.txt', help = "path of caption input")

    args = parser.parse_args()

    items = read_json_file(args.file_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device is {device}")
    clip_model, clip_preprocess = clip.load('ViT-B/32', device)
    ranks = []
    scores = []
    for item in items:
        image_id = item['image_id']
        captions = item['caption_group'][0]
        img_path = get_img_path(image_id)
        image = show_image(img_path)

        rank, clip_score = rank_captions(image, [captions['True1'], item['True2'], \
            captions['False1'], captions['False2']], clip_model, clip_preprocess, device)
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
    print(sorted(rank_count.items(), key= lambda x:x[1], reverse=True))

    rank_c = {}
    for r, c in rank_count.items():
        rank_c[r] = c
    
    print(args.file_path)
    dir = "/home/xinyi/Language_Reasoning_Test/output"
    out_file = os.path.join(dir, \
        args.file_path.split('/')[-1][:-5] + "_ranks"+".json")
    write_file(out_file, rank_c)
    print("finish\n")
    # write_file("active_scores.txt", scores)