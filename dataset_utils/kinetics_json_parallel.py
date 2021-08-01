import argparse
import json
from functools import partial
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

from utils import get_n_frames, get_n_frames_hdf5


def convert_csv_to_dict(csv_path, subset, video_dir_path, class_name_sep):
    data = pd.read_csv(csv_path)
    keys = []
    classes = set()
    key_labels = []
    for i in tqdm(range(data.shape[0]), total=data.shape[0]):
        # fixme: Debug mode only
        # if i == 100:
        #   break
        # ----------------------
        row = data.iloc[i, :]
        basename = "%s_%s_%s" % (
            row["youtube_id"],
            "%06d" % row["time_start"],
            "%06d" % row["time_end"],
        )# "%06d" % row["time_start"],
            #"%06d" % row["time_end"],

        if subset != "test":
            if class_name_sep == 'underline':
                class_name = "_".join(row["label"].split())
            else:
                class_name = row["label"]
            video_path = video_dir_path / class_name / basename
            if video_path.exists():
                key_labels.append(class_name)
                keys.append(basename)
                classes.add(class_name)
        else:
            keys.append(basename)

    database = {}
    for i in tqdm(range(len(keys)), total=len(keys)):
        # fixme: Debug mode only
        # if i == 100:
        #   break
        # ----------------------
        key = keys[i]
        database[key] = {}
        database[key]["subset"] = subset
        if subset != "test":
            label = key_labels[i]
            database[key]["annotations"] = {"label": label}
        else:
            database[key]["annotations"] = {}

    return database, classes


def load_labels(train_csv_path):
    data = pd.read_csv(train_csv_path)
    return data["label"].unique().tolist()


def make_segments(k, video_info):
    dictionary, video_type, video_dir_path = video_info
    v = dictionary[k]
    if "label" in v["annotations"]:
        label = v["annotations"]["label"]
    else:
        label = "test"

    if video_type == "jpg":
        if v["subset"] == "train":
            video_path = video_dir_path / "train" / label / k
        elif v["subset"] == "val":
            video_path = video_dir_path / "val" / label / k

        if video_path.exists():
            n_frames = get_n_frames(video_path)
            v["annotations"]["segment"] = (1, n_frames + 1)
            # print(n_frames)
    else:
        video_path = video_dir_path / label / f"{k}.hdf5"
        if video_path.exists():
            n_frames = get_n_frames_hdf5(video_path)
            v["annotations"]["segment"] = (0, n_frames)

    dictionary[k] = v
    print(dictionary[k]["annotations"]["segment"])


def convert_kinetics_csv_to_json(
    train_csv_path,
    val_csv_path,
    test_csv_path,
    video_dir_path,
    video_type,
    dst_json_path,
    class_name_sep,
    n_jobs
):
    labels = load_labels(train_csv_path)
    train_database, train_classes = convert_csv_to_dict(
        train_csv_path, "train", video_dir_path / "train", class_name_sep
    )
    val_database, val_classes = convert_csv_to_dict(
        val_csv_path, "val", video_dir_path / "val", class_name_sep
    )
    test_video_path = video_dir_path / "test"
    test_exists = test_csv_path.exists() and test_video_path.exists()
    if test_exists:
        test_database, _ = convert_csv_to_dict(test_csv_path, "test", test_video_path)

    labels = train_classes.intersection(val_classes)
    dst_data = {}
    dst_data["labels"] = sorted(list(labels))
    dst_data["database"] = {}
    dst_data["database"].update(train_database)
    dst_data["database"].update(val_database)
    if test_exists:
        dst_data["database"].update(test_database)

    p = mp.Pool(mp.cpu_count())
    managed_dict = mp.Manager().dict(dst_data["database"])
    print("Parallel Segments Processing Started with {} jobs".format(n_jobs))
    p.map(
        partial(make_segments, video_info=(managed_dict, video_type, video_dir_path)),
        managed_dict.keys(),
    )

    dst_data["database"] = managed_dict.copy()
    with dst_json_path.open("w") as dst_file:
        json.dump(dst_data, dst_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir_path",
        default=None,
        type=Path,
        help=(
            "Directory path including "
            "kinetics_train.csv, kinetics_val.csv, "
            "(kinetics_test.csv (optional))"
        ),
    )
    parser.add_argument(
        "n_classes",
        default=400,
        type=int,
        help="400, 600, or 700 (Kinetics-400, Kinetics-600, or Kinetics-700)",
    )
    parser.add_argument(
        "video_path",
        default=None,
        type=Path,
        help=(
            "Path of video directory (jpg or hdf5)."
            "Using to get n_frames of each video."
        ),
    )
    parser.add_argument("video_type", default="jpg", type=str, help=("jpg or hdf5"))
    parser.add_argument(
        "dst_path", default=None, type=Path, help="Path of dst json file."
    )
    parser.add_argument("--class_name_sep",
                        default="space",
                        type=str,
                        choices=['space', "underline"],
                        help="Separator of class names which can be underline ('_') or single space")
    parser.add_argument("--n_jobs", default=-1, type=int, help="Number of parallel jobs")

    args = parser.parse_args()

    assert args.video_type in ["jpg", "hdf5"]

    train_csv_path = args.dir_path / "kinetics-{}_train.csv".format(args.n_classes)
    val_csv_path = args.dir_path / "kinetics-{}_val.csv".format(args.n_classes)
    test_csv_path = args.dir_path / "kinetics-{}_test.csv".format(args.n_classes)

    convert_kinetics_csv_to_json(
        train_csv_path,
        val_csv_path,
        test_csv_path,
        args.video_path,
        args.video_type,
        args.dst_path,
        args.class_name_sep,
        args.n_jobs,
    )
