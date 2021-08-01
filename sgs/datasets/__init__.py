#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import DATASET_REGISTRY, build_dataset  # noqa
from .kinetics import Kinetics  # noqa
# from .kinetics_jpg import Kineticsjpg
from .videodataset import Ucf101, Hmdb51, Ssv2, Kineticsjpg, SmthV1
