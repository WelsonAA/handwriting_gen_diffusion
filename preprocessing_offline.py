import os
import numpy as np
import tensorflow as tf
import random
import pickle
import argparse
from utils import Tokenizer

'''
Creates the offline dataset for training

Before running this script, download the following files from 
https://fki.tic.heia-fr.ch/databases/download-the-iam-on-line-handwriting-database

data/lineImages-all.tar.gz    -   the images for the offline dataset
ascii-all.tar.gz              -   the text labels for the dataset

Extract these contents and put them in the ./data directory (unless otherwise specified).
They should have the same names, e.g., "lineImages-all" for images.
'''


def remove_whitespace(img, thresh, remove_middle=False):
    # removes any column or row without a pixel less than specified threshold
    row_mins = np.amin(img, axis=1)
    col_mins = np.amin(img, axis=0)

    rows = np.where(row_mins < thresh)
    cols = np.where(col_mins < thresh)

    if remove_middle:
        return img[rows[0]][:, cols[0]]
    else:
        rows, cols = rows[0], cols[0]
        return img[rows[0]:rows[-1], cols[0]:cols[-1]]


def parse_page_text(dir_path, id):
    text_dict = {}
    with open(dir_path + '/' + id) as f:
        has_started = False
        line_num = -1
        for l in f.readlines():
            if 'CSR' in l:
                has_started = True
            # the text under 'CSR' is correct, the one labeled under 'OCR' is not
            if has_started:
                if line_num > 0:  # there is one space after 'CSR'
                    text_dict[id[:-4] + '-%02d' % line_num] = l.strip()
                line_num += 1
    return text_dict


def create_dict(path):
    # creates a dictionary of all the line IDs and their respective texts
    text_dict = {}
    for dir in os.listdir(path):
        dirpath = os.path.join(path, dir)
        for subdir in os.listdir(dirpath):
            subdirpath = os.path.join(dirpath, subdir)
            forms = os.listdir(subdirpath)
            [text_dict.update(parse_page_text(subdirpath, f)) for f in forms]
    return text_dict


def read_img(path, height):
    img = tf.keras.preprocessing.image.load_img(path, color_mode='grayscale')
    img_arr = tf.keras.preprocessing.image.img_to_array(img).astype('uint8')
    img_arr = remove_whitespace(img_arr, thresh=127)
    h, w, _ = img_arr.shape
    img_arr = tf.image.resize(img_arr, (height, height * w // h))
    return img_arr.numpy().astype('uint8')


def create_offline_dataset(formlist, images_path, tokenizer, text_dict, height):
    dataset = []
    forms = open(formlist).readlines()

    for f in forms:
        offline_path = os.path.join(images_path, f[1:4], f[1:8])
        offline_samples = [s for s in os.listdir(offline_path) if f[1:-1] in s]
        shuffled_offline_samples = offline_samples.copy()
        random.shuffle(shuffled_offline_samples)

        for img_name in offline_samples:
            img_path = os.path.join(offline_path, img_name)
            if img_name[:-4] in text_dict:
                encoded_text = tokenizer.encode(text_dict[img_name[:-4]])
                img_data = read_img(img_path, height)
                dataset.append((encoded_text, img_data))

    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text_path', help='path to text labels, \
                        default ./data/ascii-all', default='./data/ascii-all')

    parser.add_argument('-i', '--images_path', help='path to line images, \
                        default ./data/lineImages-all', default='./data/lineImages-all')

    parser.add_argument('-H', '--height', help='the height of offline images, \
                        default 96', type=int, default=96)

    args = parser.parse_args()
    t_path = args.text_path
    i_path = args.images_path
    H = args.height

    train_info = 'data/trainset.txt'
    val1_info = 'data/testset_f.txt'  # labeled as test, we use validation set 1 as test instead
    val2_info = 'data/testset_t.txt'
    test_info = 'data/testset_v.txt'  # labeled as validation, but we use as test

    tok = Tokenizer()
    labels = create_dict(t_path)

    # Create offline datasets
    train_dataset = create_offline_dataset(train_info, i_path, tok, labels, H)
    val1_dataset = create_offline_dataset(val1_info, i_path, tok, labels, H)
    val2_dataset = create_offline_dataset(val2_info, i_path, tok, labels, H)
    test_dataset = create_offline_dataset(test_info, i_path, tok, labels, H)

    # Combine training sets and shuffle
    train_dataset += val1_dataset
    train_dataset += val2_dataset
    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    # Save datasets
    with open('./data_offline/train_offline.p', 'wb') as f:
        pickle.dump(train_dataset, f)
    with open('./data_offline/test_offline.p', 'wb') as f:
        pickle.dump(test_dataset, f)


if __name__ == '__main__':
    main()
