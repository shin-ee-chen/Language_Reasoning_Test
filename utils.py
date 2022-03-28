import json
import os
import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize, pos_tag
import numpy as np
from PIL import Image
import clip
import torch


def isActionSentence(sent_tokens):
    sentence = pos_tag(sent_tokens)
    grammar = r'CHUNK: {<N.*>+<.*>?<V.*>+<.*>?<N.*>+}'
    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(sentence)
    for subtree in tree.subtrees():
        if subtree.label() == 'CHUNK': 
            return True
    return False

def show_image(img_path, show=False):
    image = Image.open(img_path).convert("RGB")
    if show:
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    return image

def show_originals(index, items, img_dir):
    item = items[index]
    print(index)
    img_path = os.path.join(img_dir, item['filepath'],item['filename'])
    for sen in item['sentences']:
        print(sen['raw'])
    show_image(img_path)


def rank_captions(image, texts, model, preprocess, device):
    text_inputs = clip.tokenize(texts).to(device)
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Calculate features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = ( image_features @ text_features.T)
    clip_score = 2.5 * torch.max(similarity, torch.zeros(similarity.shape).to(device))

    # Rank k captions for image
    _, indices = clip_score[0].topk(text_inputs.shape[0])
    rank = list(range(text_inputs.shape[0]))
    for i, value in enumerate(indices.to('cpu').numpy()):
        rank[value] = i+1 
    # print(clip_score.to('cpu').numpy())
    # print(rank)
    return rank, clip_score


def get_accuracy(rank):
    correct_count = 0
    total = len(rank)
    for i, r in enumerate(rank):
        if i < total/2 and r <= total/2:
            correct_count += 1
        elif i >= total/2 and r > total/2:
            correct_count += 1
    return correct_count/total

def get_img_path(img_id):
    VG_100K_path = "/home/xinyi/datasets/VG_100K/" + str(img_id) + ".jpg"
    VG_100K_2_path = "/home/xinyi/datasets/VG_100K_2/" + str(img_id) + ".jpg"
    if os.path.exists(VG_100K_path):
        return VG_100K_path
    elif os.path.exists(VG_100K_2_path):
        return VG_100K_2_path
    else:
        return None

def read_json_file(file_path):
    global json_file
    try:
        json_file = open(file_path, "r")
        return json.load(json_file)
    finally:
        if json_file:
            print("close file...")
            json_file.close()

def write_file(file_path, content):
    if not os.path.exists('./output/'):
        os.makedirs('./output/')
    with open(file_path, 'w') as f:
        # f.write(str(content))
        json.dump(content, f)



if __name__ == "__main__":
    print(get_accuracy([3,4,1,2]))
    print(get_accuracy([2,3,1,4]))