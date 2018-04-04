import json
import pickle
import os
import traceback
from collections import namedtuple

def parse(content, separator):
    if len(separator) == 0:
        return content.strip()
    else:
        return [parse(c, separator[1:]) for c in content.strip().split(separator[0])]


def write(path, string):
    with open(path, "w", encoding="utf-8-sig") as f:
        f.write(string)


def read(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return f.read()


def json_dump(data, path):
    json.dump(data, open(path, "w", encoding="utf8"), ensure_ascii=False)


def json_load(path):
    return json.load(open(path, "r", encoding="utf8"))

def get_file_name_w_extension(path):
    filename_w_ext = os.path.basename(path)
    filename, file_extension = os.path.splitext(filename_w_ext)
    return filename, file_extension

def get_file_extension(path):
    filename, file_extension = get_file_name_w_extension(path)
    return file_extension

def get_file_name(path):
    filename, file_extension = get_file_name_w_extension(path)
    return filename


def pkl_load(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def make_dict(*expr):
    (filename, line_number, function_name, text) = traceback.extract_stack()[-2]
    begin = text.find('make_dict(') + len('make_dict(')
    end = text.find(')', begin)
    text = [name.strip() for name in text[begin:end].split(',')]
    return dict(zip(text, expr))


def dict_to_object(d):
    return namedtuple('Struct', d.keys())(*d.values())
