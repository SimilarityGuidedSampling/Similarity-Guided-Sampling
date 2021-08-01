from os.path import basename
import subprocess
import argparse
from pathlib import Path

from joblib import Parallel, delayed


def video_process(video_file_path, dst_root_path, ext, fps=-1, size=240):
    if ext != video_file_path.suffix:
        return

    ffprobe_cmd = ('ffprobe -v error -select_streams v:0 '
                   '-of default=noprint_wrappers=1:nokey=1 -show_entries '
                   'stream=width,height,avg_frame_rate,duration').split()
    ffprobe_cmd.append(str(video_file_path))

    p = subprocess.run(ffprobe_cmd, capture_output=True)
    res = p.stdout.decode('utf-8').splitlines()
    if len(res) < 4:
        return

    frame_rate = [float(r) for r in res[2].split('/')]
    frame_rate = frame_rate[0] / frame_rate[1]
    # duration = float(res[3])
    # n_frames = int(frame_rate * duration)

    name = video_file_path.stem
    dst_dir_path = dst_root_path / name
    dst_dir_path.mkdir(exist_ok=True)
    n_exist_frames = len([
        x for x in dst_dir_path.iterdir()
        if x.suffix == '.jpg' and x.name[0] != '.'
    ])

    if n_exist_frames >= 180:
        return

    # width = int(res[0])
    # height = int(res[1])
    # if width > height:
    #     vf_param = 'scale=-1:{}'.format(size)
    # else:
    #     vf_param = 'scale={}:-1'.format(size)

    ffmpeg_cmd = ['ffmpeg', '-i', str(video_file_path), "-r", str(fps), "-q:v", '1']
    ffmpeg_cmd += ['-threads', '1', '{}/image_%05d.jpg'.format(dst_dir_path)]
    print(ffmpeg_cmd)
    subprocess.run(ffmpeg_cmd)
    print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir_path', default=None, type=Path, help='Directory path of ssv2 videos')
    parser.add_argument(
        'dst_path',
        default=None,
        type=Path,
        help='Directory path of jpg videos')
    parser.add_argument(
        '--n_jobs', default=-1, type=int, help='Number of parallel jobs')
    parser.add_argument(
        '--fps',
        default=30,
        type=int,
        help=('Frame rates of output videos. '
              '-1 means original frame rates.'))
    parser.add_argument(
        '--size', default=256, type=int, help='Frame size of output videos.')
    args = parser.parse_args()

    ext = '.webm'

    video_dir_paths = [x for x in sorted(args.dir_path.iterdir())]

    test_set_video_path = args.dir_path / 'test'
    if test_set_video_path.exists():
        video_dir_paths.append(test_set_video_path)

    status_list = Parallel(
        n_jobs=args.n_jobs,
        backend='threading')(delayed(video_process)(
            vid_path, args.dst_path, ext, args.fps, args.size)
                             for vid_path in video_dir_paths)
