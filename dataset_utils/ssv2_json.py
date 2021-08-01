from os.path import join
import argparse
import json
from pathlib import Path

from utils import get_n_frames


def convert_csv_to_dict(json_path, subset, label_dict):
    # Loading labels.
    with json_path.open('r') as f:
        data = json.load(f)

    database = {}
    for video in data:
        video_name = video["id"]
        database[video_name] = {}
        if subset == 'test':
            database[video_name]['subset'] = subset
            database[video_name]['annotations'] = {}
        else:
            template = video["template"]
            template = template.replace("[", "")
            template = template.replace("]", "")
            label = int(label_dict[template])
            database[video_name]['subset'] = subset
            database[video_name]['annotations'] = {'label': label}

    return database


def load_labels(label_json_path):
    # Loading label names.
    with label_json_path.open("r") as f:
        label_dict = json.load(f)
    return label_dict


def convert_smt_smtv2_csv_to_json(label_json_path, train_json_path, val_json_path, test_json_path,
                               video_dir_path, dst_json_path):
    labels = load_labels(label_json_path)
    train_database = convert_csv_to_dict(train_json_path, 'train', labels)
    val_database = convert_csv_to_dict(val_json_path, 'val', labels)
    test_database = convert_csv_to_dict(test_json_path, 'test', labels)

    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)
    dst_data['database'].update(test_database)

    for k, v in dst_data['database'].items():
        video_path = video_dir_path / str(k)
        n_frames = get_n_frames(video_path, keyword="")
        v['annotations']['segment'] = (1, n_frames + 1)

    with dst_json_path.open('w') as dst_file:
        json.dump(dst_data, dst_file, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_path',
                        default=None,
                        type=Path,
                        help=('Directory path including, '
                              'something-something-v2-{train, validation, test, labels}.csv'))
    parser.add_argument('video_path',
                        default=None,
                        type=Path,
                        help=('Path of video directory (jpg).'
                              'Using to get n_frames of each video.'))
    parser.add_argument('dst_path',
                        default=None,
                        type=Path,
                        help='Directory path of dst json file.')

    args = parser.parse_args()

    label_csv_path = args.dir_path / 'something-something-v2-labels.json'
    train_csv_path = args.dir_path / 'something-something-v2-train.json'
    val_csv_path = args.dir_path / 'something-something-v2-validation.json'
    test_csv_path = args.dir_path / 'something-something-v2-test.json'
    dst_json_path = args.dst_path / 'something-something-v2_info.json'

    convert_smt_smtv2_csv_to_json(label_csv_path, train_csv_path, val_csv_path, test_csv_path,
                                  args.video_path, dst_json_path)