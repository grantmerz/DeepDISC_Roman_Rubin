# Splitting into train, val, and test
import json, random
from deepdisc.data_format.conversions import convert_to_json

def split_data(data, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    random.shuffle(data)
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data

# with open('roman_data/annotations/all_metadata.json', 'r') as f:
#     data = json.load(f)

# with open('../lsst_data/annotations/3828.json', 'r') as f:
#     data = json.load(f)

with open('./lsst_data/annotations/all_metadata.json', 'r') as f:
    data = json.load(f)

train_data, val_data, test_data = split_data(data)

# train_file = './roman_data/annotations/train_roman.json'
# val_file = './roman_data/annotations/val_roman.json'
# test_file = './roman_data/annotations/test_roman.json'
train_file = './lsst_data/annotations/train.json'
val_file = './lsst_data/annotations/val.json'
test_file = './lsst_data/annotations/test.json'

convert_to_json(train_data, train_file)
convert_to_json(val_data, val_file)
convert_to_json(test_data, test_file)