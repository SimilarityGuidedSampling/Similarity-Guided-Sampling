# Getting Started

This document provides a brief intro of launching jobs for training and testing. Before launching any job, make sure you have properly installed the code in the repository following the instruction in [README.md](README.md) and you have prepared the dataset following [DATASET.md](sgs/datasets/DATASET.md) with the correct format.

## Train on Mini-Kinetics from Scratch

Here is how we can train models on Mini-Kinetics dataset:
```
export PYTHONPATH=/path/to/Similarity-Guided-Sampling/sgs:$PYTHONPATH
python tools/run_net.py \
  --cfg configs/Mini_Kinetics200/YOUR_CONFIG_FILE.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS K \
  TRAIN.BATCH_SIZE K x 8 \
  DATA_LOADER.NUM_WORKERS K x 8 x 3 \
```
K is the number of GPUs, Batch size should be 8 per each GPU. Number of workers is better to be 8 x 3 factor of number of GPUS. You should set the NUM_WORKERS carefully, it depends on the number of CPU thread available in the system and also free RAM.


## Train on Kinetics from Scratch

It is the same as Mini-Kinetics as described above with the only difference that the config file and DATA.PATH_TO_DATA_DIR need to be set. For example: 
```
export PYTHONPATH=/path/to/Similarity-Guided-Sampling/sgs:$PYTHONPATH
python tools/run_net.py \
  --cfg configs/Kinetics/YOUR_CONFIG_FILE.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS K \
  TRAIN.BATCH_SIZE K x 8 \
  DATA_LOADER.NUM_WORKERS K x 8 x 3 \
```


## Resume Training
You can resume training by setting TRAIN.AUTO_RESUME to True and passing **expr_num** to the command line. 
The *expr_num* can be found in the logs of your training. For example, in your log file you'll find a line like *./experiments/kinetics/3DResNet50/checkpoints/1* in which expr_num is 1.

```
python tools/run_net.py \
  --cfg configs/Kinetics/3DResNet50.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 2 \
  TRAIN.BATCH_SIZE 16 \
  TRAIN.AUTO_RESUME True \
  TRAIN.RESUME_EXPR_NUM 1 \
```


## Resume from an Existing Checkpoint
If your checkpoint is trained by PyTorch, then you can add the following line in the command line, or you can also add it in the YAML config:

```
TRAIN.CHECKPOINT_FILE_PATH path_to_your_PyTorch_checkpoint
```

If the checkpoint in trained by Caffe2, then you can do the following:

```
TRAIN.CHECKPOINT_FILE_PATH path_to_your_Caffe2_checkpoint \
TRAIN.CHECKPOINT_TYPE caffe2
```

If you need to performance inflation on the checkpoint, remember to set `TRAIN.CHECKPOINT_INFLATE` to True.


## Perform Test
We have `TRAIN.ENABLE` and `TEST.ENABLE` to control whether training or testing is required for the current job.
If only testing is preferred, you can set the `TRAIN.ENABLE` to False, 
and do not forget to pass the path to the model you want to test to TEST.CHECKPOINT_FILE_PATH.
```
python tools/run_net.py \
  --cfg configs/Kinetics/3DResNet50.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  TEST.CHECKPOINT_FILE_PATH path_to_your_checkpoint \
  TRAIN.ENABLE False \
```
