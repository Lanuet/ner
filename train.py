import json
import numpy as np
import glob
import os
import utils
from network import build_model

def load_data():
    files = glob.glob("data/*.npy")
    data = {}
    for f in files:
        name = utils.get_file_name(f)
        data[name] = np.load(f)
    data = utils.dict_to_object(data)

    data.max_sen_len, data.max_word_len = data.words_train.shape[1:]

    pos2idx = utils.json_load("embedding/pos2idx")
    data.pos_alphabet_size = len(pos2idx)
    char2idx = utils.json_load("embedding/char2idx")
    data.char_alphabet_size = len(char2idx)
    ner2idx = utils.json_load("embedding/ner2idx")
    data.num_labels = len(ner2idx)

def main():
    print("loading data")
    data = load_data()
    model = build_model(data)
    print("")

if __name__ == '__main__':
    main()