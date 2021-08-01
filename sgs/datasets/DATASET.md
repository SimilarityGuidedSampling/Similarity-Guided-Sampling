# Dataset Preparation

## Mini-Kinetics-JPG

* Download videos
  * Locate test set in ```video_directory/test```.
* Convert from avi to jpg files using ```dataset_utils/generate_video_jpgs.py```

```bash
python dataset_utils/generate_video_jpgs.py mp4_video_dir_path jpg_video_dir_path kinetics --n_jobs NUM_CPU_CORES
```

* Generate annotation file in json format similar to ActivityNet using ```dataset_utils/kinetics_json_parallel.py```
  * The CSV files (kinetics-600_{train, val, test}.csv) are included in the crawler.

```bash
python dataset_utils/kinetics_json.py csv_dir_path 600 jpg_video_dir_path jpg dst_json_path --n_jobs NUM_CPU_CORES
```

## Kinetics-JPG

* Download videos
  * Locate test set in ```video_directory/test```.
* Convert from avi to jpg files using ```dataset_utils/generate_video_jpgs.py```

```bash
python dataset_utils/generate_video_jpgs.py mp4_video_dir_path jpg_video_dir_path --vid_ext avi --size 452 --n_jobs NUM_CPU_CORES
```

* Generate annotation file in json format similar to ActivityNet using ```dataset_utils/kinetics_json.py```
  * The CSV files (kinetics-400_{train, val, test}.csv) are included in the crawler.

```bash
python dataset_utils/kinetics_json.py csv_dir_path 400 jpg_video_dir_path jpg dst_json_file
```
