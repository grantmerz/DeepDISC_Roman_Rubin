"""make_subsets.py — the train_30k / val_4k / test_8k annotation subsets used in the paper.

Two modes:

  first-n     What the paper did (2025-10-31): take the first N images of train.json and the same fraction of
              val.json and test.json, and write train_{N//1000}k.json, val_{...}k.json and test_{...}k.json
              next to them.

  from-lists  Rebuild the paper's exact subsets from the image lists in data_processing/splits/. The paper's
              train/val/test split was made with an unseeded shuffle, so re-running preprocess_anns.py gives a
              different split; this mode picks the paper's images, in the paper's order, out of any annotation
              file that contains them (for example all_metadata_clean.json).

Usage:
    python data_processing/make_subsets.py first-n --anns-dir /path/to/lsst_data/annotations_lvl5 --train-size 30000
    python data_processing/make_subsets.py from-lists --source /path/to/annotations_lvl5/all_metadata_clean.json \
        --out-dir /path/to/annotations_lvl5
"""
import argparse
import csv
import json
import os

SPLITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'splits')
SUBSETS = ['train_30k', 'val_4k', 'test_8k']


def resample_splits_by_ratio(base_dir, new_train_size=30000):
    """First new_train_size images of train.json; val.json and test.json cut by the same ratio."""
    with open(os.path.join(base_dir, 'train.json')) as f:
        train_data = json.load(f)
    with open(os.path.join(base_dir, 'val.json')) as f:
        val_data = json.load(f)
    with open(os.path.join(base_dir, 'test.json')) as f:
        test_data = json.load(f)

    ratio = new_train_size / len(train_data)
    new_val_size = int(len(val_data) * ratio)
    new_test_size = int(len(test_data) * ratio)
    print(f'Original sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}; ratio={ratio:.4f}')

    for name, data, n in [('train', train_data, new_train_size), ('val', val_data, new_val_size),
                          ('test', test_data, new_test_size)]:
        out_path = os.path.join(base_dir, f'{name}_{n // 1000}k.json')
        with open(out_path, 'w') as f:
            json.dump(data[:n], f, indent=4)
        print(f'  {name}: {n} images -> {out_path}')


def _key(file_name):
    """tile/cutout part of a file name, so lists match whatever data root the annotations use."""
    return '/'.join(file_name.replace('\\', '/').split('/')[-2:])


def subsets_from_lists(source, out_dir, splits_dir=SPLITS_DIR):
    """Write train_30k.json, val_4k.json and test_8k.json with the paper's images, in list order."""
    with open(source) as f:
        records = {_key(d['file_name']): d for d in json.load(f)}
    for name in SUBSETS:
        with open(os.path.join(splits_dir, f'{name}_images.csv')) as f:
            wanted = [_key(row['file_name']) for row in csv.DictReader(f)]
        missing = [k for k in wanted if k not in records]
        if missing:
            raise KeyError(f'{name}: {len(missing)} of {len(wanted)} images are not in {source} '
                           f'(first: {missing[0]})')
        out_path = os.path.join(out_dir, f'{name}.json')
        with open(out_path, 'w') as f:
            json.dump([records[k] for k in wanted], f, indent=4)
        print(f'  {name}: {len(wanted)} images -> {out_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='mode', required=True)
    p1 = sub.add_parser('first-n', help='first N images of train.json, same fraction of val/test (paper method)')
    p1.add_argument('--anns-dir', required=True, help='directory with train.json, val.json and test.json')
    p1.add_argument('--train-size', type=int, default=30000)
    p2 = sub.add_parser('from-lists', help="rebuild the paper's subsets from data_processing/splits/")
    p2.add_argument('--source', required=True, help='annotation JSON containing every listed image')
    p2.add_argument('--out-dir', required=True)
    p2.add_argument('--splits-dir', default=SPLITS_DIR)
    args = parser.parse_args()
    if args.mode == 'first-n':
        resample_splits_by_ratio(args.anns_dir, args.train_size)
    else:
        subsets_from_lists(args.source, args.out_dir, args.splits_dir)


if __name__ == '__main__':
    main()
