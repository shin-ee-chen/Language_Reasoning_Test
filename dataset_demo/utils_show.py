import matplotlib.pyplot as plt
from PIL import Image

import json
import os

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


def get_img_path(img_id):
    base_dir = "/Users/xinyichen/Desktop/Thesis/Dataset_Construction"
    VG_100K_path = os.path.join(base_dir, "VG_100K", str(img_id) + ".jpg")
    VG_100K_2_path = os.path.join(base_dir, "VG_100K_2", str(img_id) + ".jpg")
    if os.path.exists(VG_100K_path):
        return VG_100K_path
    elif os.path.exists(VG_100K_2_path):
        return VG_100K_2_path
    else:
        return None

def show_image(img_id):
    img_path = get_img_path(img_id)
    if img_path == None:
        return
    image = Image.open(img_path).convert("RGB")
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    return image

def get_caption_dict(file_path):
    captions = read_json_file(file_path)
    caption_dict = {}
    for caption in captions:
        caption_dict[caption['index']] = {"True1": caption["True1"], "True2": caption["True2"],\
             "False1": caption["False1"], "False2": caption["False2"], "image_id": caption["image_id"], \
                 'predicate': caption['predicate']}
    return caption_dict


def show_example_by_index(index, caption_dict):
    caption = caption_dict[index]
    show_image(caption["image_id"])
    print("index:", index)
    print(caption["True1"])
    print(caption["True2"])
    print(caption["False1"])
    print(caption["False2"])

def show_example_by_img_id(img_id, caption_dict):
    found_img = False
    for index, caption in caption_dict.items():
        if caption['image_id'] == img_id:
            found_img = True
            show_image(caption["image_id"])
            print("index:", index)
            print(caption["True1"])
            print(caption["True2"])
            print(caption["False1"])
            print(caption["False2"])
        elif found_img:
            break
    return index


def get_index_by_img_id(img_id, caption_dict):
    found_img = False
    index_list = []
    for index, caption in caption_dict.items():
        if caption['image_id'] == img_id:
            found_img = True
            index_list.append(index)
        elif found_img:
            break
    return index_list


if __name__ == "__main__":
    img_id = 2403500
    base_dir = "/Users/xinyichen/Desktop/Thesis/Dataset_Construction/"
    print(os.path.join(base_dir, "VG_100K"))
    # , str(img_id) + ".jpg"))
    print(get_img_path(img_id))