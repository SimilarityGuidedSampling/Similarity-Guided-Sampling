import json
import random
from pathlib import Path

import torch
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
from tqdm import tqdm

from .frame_loader import VideoLoader

from .build import DATASET_REGISTRY

from . import transform as transform
from .temporal_transforms import TemporalSubsampling, TemporalRandomCrop, LoopPadding
from .temporal_transforms import Compose as TemporalCompose, temporal_slicing
from .utils import pack_pathway_output, spatial_sampling
import sgs.utils.logging as logging

logger = logging.get_logger(__name__)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    label_list = data["labels"]
    if isinstance(data["labels"], dict):
        label_list = map(int, data["labels"].values())
    for class_label in label_list:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_database(data, subset, root_path, video_path_formatter, num_clips=1):
    video_ids = []
    video_paths = []
    annotations = []
    spatial_temporal_idx = []
    class_n_videos = {}
    for key, value in tqdm(data["database"].items()):
        this_subset = value["subset"]
        if this_subset == subset:
            for idx in range(num_clips):
                video_ids.append(key)
                annotations.append(value["annotations"])
                spatial_temporal_idx.append(idx)
                if "video_path" in value:
                    video_paths.append(Path(value["video_path"]))
                else:
                    label = value["annotations"]["label"]
                    video_paths.append(video_path_formatter(root_path, label, key))
                    if label in class_n_videos:
                        class_n_videos[label] += 1
                    else:
                        class_n_videos[label] = 1

    return video_ids, video_paths, annotations, spatial_temporal_idx, class_n_videos


class VideoDataset(data.Dataset):
    def __init__(
            self,
            cfg,
            split,
            root_path,
            annotation_path,
            num_retries=10,
            video_loader=None,
            video_path_formatter=(lambda root_path, label, video_id: root_path / label / video_id),
            image_name_formatter=lambda x: f'image_{x:05d}.jpg',
            target_type="label"
    ):
        self.cfg = cfg
        self._num_retries = num_retries
        self._split = split

        if split in ["train", "val"]:
            self._num_clips = 1
        elif split in ["test"]:
            self._num_clips = (
                    cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        # just to make sure we use all videos in validation set
        if split in ['val', 'test']:
            cfg.DATA.DATA_FRACTION = 1.0

        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, split, video_path_formatter, self._num_clips, cfg.DATA.DATA_FRACTION
        )
        assert len(self.class_names) == cfg.MODEL.NUM_CLASSES, f"{cfg.MODEL.NUM_CLASSES} Number of classes for model is wrong."
        logger.info(
            "Constructing {} dataloader (size: {}) from {} with {} classes".format(
                cfg.TRAIN.DATASET.upper(), len(self.data), root_path, len(self.class_names)
            )
        )
        logger.info(f"Using {int(cfg.DATA.DATA_FRACTION*100)}% of videos per class for {split}")

        temporal_transform = []
        if split in ["train", "val"]:
            # -1 indicates random sampling.
            self.temporal_sample_index = -1
            self.spatial_sample_index = -1
            self.min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            self.max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            self.crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if self.cfg.DATA.SAMPLING_RATE > 1:
                temporal_transform.append(TemporalSubsampling(self.cfg.DATA.SAMPLING_RATE))

            # maybe for validation this needs to be changed
            temporal_transform.append(TemporalRandomCrop(self.cfg.DATA.NUM_FRAMES))
            self.temporal_transform = TemporalCompose(temporal_transform)

        self.spatial_transform = Compose(
            [ToTensor(), Normalize(mean=self.cfg.DATA.MEAN, std=self.cfg.DATA.STD)]
        )

        self.target_transform = None

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type

    def _spatial_sampling(
        self, frames, spatial_idx=-1, min_scale=256, max_scale=320, crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.random_crop(frames, crop_size)
            frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames

    def __make_dataset(self, root_path,
                       annotation_path,
                       subset,
                       video_path_formatter,
                       num_clips,
                       fraction=1.0):
        if subset == 'test':
            subset = 'val'
            logger.info("Using val data instead of test.")

        with annotation_path.open("r") as f:
            data = json.load(f)

        video_ids, video_paths, annotations, spatial_temporal_idx, n_videos_per_class = get_database(
            data, subset, root_path, video_path_formatter, num_clips
        )

        class_to_idx = get_class_labels(data)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        class_to_n_videos = {}
        for name, n_video in n_videos_per_class.items():
            class_to_n_videos[name] = int(fraction * n_video)

        n_videos = len(video_ids)
        dataset = []
        num_frames = []
        for i in tqdm(range(n_videos)):
            # if i % (n_videos // 5) == 0:
            #     print('dataset loading [{}/{}]'.format(i, len(video_ids)))
            if "label" in annotations[i]:
                label = annotations[i]["label"]
                label_id = class_to_idx[label]
                class_to_n_videos[label] -= 1
                if class_to_n_videos[label] < 0 and subset == 'train':
                    continue
            else:
                label = "test"
                label_id = -1

            video_path = video_paths[i]
            # It's better to not to check the video_path here. It makes the dataset initialization too slow!
            # if not video_path.exists():
            #    continue

            segment = annotations[i]["segment"]
            if segment[1] == 1:
                continue
            num_frames.append(segment[1]-segment[0])
            frame_indices = list(range(segment[0], segment[1]))
            sample = {
                "video": video_path,
                "segment": segment,
                "frame_indices": frame_indices,
                "video_id": video_ids[i],
                "clip_id": spatial_temporal_idx[i],
                "label": label_id,
            }
            dataset.append(sample)

        return dataset, idx_to_class

    def __frame_sampling(self, frame_indices):
        if self._split in ["test"]:
            temporal_transform = []
            if self.cfg.DATA.SAMPLING_RATE > 1:
                temporal_transform.append(TemporalSubsampling(self.cfg.DATA.SAMPLING_RATE))
            temporal_transform = TemporalCompose(temporal_transform)
            frame_indices = temporal_transform(frame_indices)
            frame_indices = temporal_slicing(frame_indices,
                                             self.cfg.DATA.NUM_FRAMES,
                                             self.temporal_sample_index,
                                             self._num_clips)
        else:
            frame_indices = self.temporal_transform(frame_indices)

        return frame_indices

    def __getitem__(self, index):
        if self._split in ["test"]:
            self.temporal_sample_index = (
                    self.data[index]['clip_id']
                    // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            self.spatial_sample_index = (
                    self.data[index]['clip_id']
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            self.min_scale, self.max_scale, self.crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({self.min_scale, self.max_scale, self.crop_size}) == 1

        for _ in range(self._num_retries):
            path = self.data[index]["video"]
            frame_indices = self.data[index]["frame_indices"]
            frame_indices = self.__frame_sampling(frame_indices)
            clip = None
            try:
                clip = self.loader(path, frame_indices)
            except Exception as e:
                logger.info(f"Failed to load video from {path} with error {e}")

            if clip is None:
                index = random.randint(0, len(self.data) - 1)
                continue

            if self.spatial_transform is not None:
                clip = [self.spatial_transform(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

            clip = self._spatial_sampling(
                clip,
                spatial_idx=self.spatial_sample_index,
                min_scale=self.min_scale,
                max_scale=self.max_scale,
                crop_size=self.crop_size,
            )

            target = self.data[index]
            if self.target_transform is not None:
                target = self.target_transform(target)
            clip = pack_pathway_output(self.cfg, clip)

            return clip, target["label"], index, {}
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(self._num_retries)
            )

    def __len__(self):
        return len(self.data)


@DATASET_REGISTRY.register()
class Ucf101(VideoDataset):
    def __init__(self, cfg, split):
        root_path = Path(cfg.DATA.PATH_TO_DATA_DIR).joinpath("data")
        annotation_path = Path(cfg.DATA.PATH_TO_DATA_DIR).joinpath(
            cfg.DATA.JSON_META_FILE
        )
        super(Ucf101, self).__init__(cfg=cfg,
                                     split=split,
                                     root_path=root_path,
                                     annotation_path=annotation_path)


@DATASET_REGISTRY.register()
class Hmdb51(VideoDataset):
    def __init__(self, cfg, split):
        root_path = Path(cfg.DATA.PATH_TO_DATA_DIR).joinpath("data")
        annotation_path = Path(cfg.DATA.PATH_TO_DATA_DIR).joinpath(
            cfg.DATA.JSON_META_FILE
        )
        super(Hmdb51, self).__init__(cfg=cfg,
                                     split=split,
                                     root_path=root_path,
                                     annotation_path=annotation_path)


@DATASET_REGISTRY.register()
class SmthV1(VideoDataset):
    def __init__(self, cfg, split):
        root_path = Path(cfg.DATA.PATH_TO_DATA_DIR).joinpath("20bn-something-something-v1")
        annotation_path = Path(cfg.DATA.PATH_TO_DATA_DIR).joinpath(
            cfg.DATA.JSON_META_FILE
        )
        video_path_formatter = lambda root_path, label, video_id: root_path / video_id
        image_name_formatter = lambda x: f'{x:05d}.jpg'
        super(SmthV1, self).__init__(cfg=cfg,
                                   split=split,
                                   root_path=root_path,
                                   annotation_path=annotation_path,
                                   video_path_formatter=video_path_formatter,
                                   image_name_formatter=image_name_formatter)


@DATASET_REGISTRY.register()
class Ssv2(VideoDataset):
    def __init__(self, cfg, split):
        root_path = Path(cfg.DATA.PATH_TO_DATA_DIR).joinpath("jpgs")
        annotation_path = Path(cfg.DATA.PATH_TO_DATA_DIR).joinpath(
            cfg.DATA.JSON_META_FILE
        )
        video_path_formatter = lambda root_path, label, video_id: root_path / video_id

        super(Ssv2, self).__init__(cfg=cfg,
                                   split=split,
                                   root_path=root_path,
                                   annotation_path=annotation_path,
                                   video_path_formatter=video_path_formatter)

    def __frame_sampling(self, frame_indices):
        video_length = len(frame_indices)
        num_frames = self.cfg.DATA.NUM_FRAMES
        seg_size = float(video_length - 1) / num_frames
        seq = []
        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if self._split == "train":
                rand_index = random.randint(start, end)
                seq.append(frame_indices[rand_index])
            else:
                index = (start + end) // 2
                seq.append(frame_indices[index])

        return seq

    def __getitem__(self, index):
        if self._split in ["test"]:
            self.temporal_sample_index = (
                    self.data[index]['clip_id']
                    // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            self.spatial_sample_index = (
                    self.data[index]['clip_id']
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            self.min_scale, self.max_scale, self.crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({self.min_scale, self.max_scale, self.crop_size}) == 1

        for _ in range(self._num_retries):
            path = self.data[index]["video"]
            frame_indices = self.data[index]["frame_indices"]
            frame_indices = self.__frame_sampling(frame_indices)
            clip = None
            try:
                clip = self.loader(path, frame_indices)
            except Exception as e:
                logger.info(f"Failed to load video from {path} with error {e}")

            if clip is None:
                index = random.randint(0, len(self.data) - 1)
                continue

            if self.spatial_transform is not None:
                clip = [self.spatial_transform(img) for img in clip]
            # T H W C -> C T H W.
            frames = torch.stack(clip, 0).permute(1, 0, 2, 3)

            # Perform data augmentation.
            frames = spatial_sampling(
                frames,
                spatial_idx=self.spatial_sample_index,
                min_scale=self.min_scale,
                max_scale=self.max_scale,
                crop_size=self.crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )
            target = self.data[index]
            if self.target_transform is not None:
                target = self.target_transform(target)
            frames = pack_pathway_output(self.cfg, frames)

            return frames, target["label"], index, {}
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(self._num_retries)
            )


@DATASET_REGISTRY.register()
class Kineticsjpg(VideoDataset):
    def __init__(self, cfg, split):

        if split in ["test", "val"]:
            root_path = Path(cfg.DATA.PATH_TO_DATA_DIR).joinpath("val")
        else:
            root_path = Path(cfg.DATA.PATH_TO_DATA_DIR).joinpath(f"{split}")
        annotation_path = Path(cfg.DATA.PATH_TO_DATA_DIR).joinpath(
            cfg.DATA.JSON_META_FILE
        )
        super(Kineticsjpg, self).__init__(cfg=cfg,
                                          split=split,
                                          root_path=root_path,
                                          annotation_path=annotation_path)