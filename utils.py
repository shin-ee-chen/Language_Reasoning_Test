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

def show_image(img_path):
    image = Image.open(img_path).convert("RGB")
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
    print(clip_score.to('cpu').numpy())
    print(rank)