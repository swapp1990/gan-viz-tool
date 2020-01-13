import wget
import sys
import os
import subprocess
import uuid
import glob
from subprocess import Popen, PIPE, STDOUT
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import pandas as pd
from collections import OrderedDict

def downloadTrainVal():
    url = "https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt"
    wget.download(url, "C:\\Users\Swapinl\Documents\Datasets\\ava")

def downloadVids():
# file = open(r"C:\\Users\Swapinl\Documents\Datasets\\ava\\ava_file_names_trainval_v2.1.txt","w+") 
    with open("C:\\Users\Swapinl\Documents\Datasets\\ava\\ava_file_names_trainval_v2.1.txt") as fp:
        line = fp.readline()
        print(line)
        cnt = 1
        while line:
            print("Saving {}: {}".format(cnt, line.strip()))
            url = "https://s3.amazonaws.com/ava-dataset/trainval/"+line.strip()
            wget.download(url, "C:\\Users\Swapinl\Documents\Datasets\\ava")
            line = fp.readline()
            cnt += 1

def cutVids15min():
    IN_DATA_DIR="C:\\Users\Swapinl\Documents\Datasets\\ava\\"
    OUT_DATA_DIR="C:\\Users\Swapinl\Documents\Datasets\\ava\\videos_15min"

    ffmpeg_extract_subclip("C:\\Users\Swapinl\Documents\Datasets\\ava\\videos\_7oWZq_s_Sk.mkv", 900, 915, targetname="test.mkv")

    # ffmpeg -ss 900 -t 901 -i "${video}" "${out_name}"

def create_video_folders(dataset, output_dir, tmp_dir):
    if 'label-name' not in dataset.columns:
        this_dir = os.path.join(output_dir, 'test')
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
        # I should return a dict but ...
        return this_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    label_to_dir = {}
    for label_name in dataset['label-name'].unique():
        this_dir = os.path.join(output_dir, label_name)
        if not os.path.exists(this_dir):
            os.makedirs(this_dir)
        label_to_dir[label_name] = this_dir
    return label_to_dir

def parse_kinetics_annotations(ignore_is_cc=False):
    df = pd.read_csv("kinetics-600_test.csv")
    if 'youtube_id' in df.columns:
        columns = OrderedDict([
            ('youtube_id', 'video-id'),
            ('time_start', 'start-time'),
            ('time_end', 'end-time'),
            ('label', 'label-name')])
        df.rename(columns=columns, inplace=True)
        if ignore_is_cc:
            df = df.loc[:, df.columns.tolist()[:-1]]
    return df

def construct_video_filename(row, label_to_dir, trim_format='%06d'):
    """Given a dataset row, this function constructs the
       output filename for a given video.
    """
    basename = '%s_%s_%s.mp4' % (row['video-id'],
                                 trim_format % row['start-time'],
                                 trim_format % row['end-time'])
    if not isinstance(label_to_dir, dict):
        dirname = label_to_dir
    else:
        dirname = label_to_dir[row['label-name']]
    output_filename = os.path.join(dirname, basename)
    return output_filename

def download_clip(video_identifier, output_filename,
                  start_time, end_time,
                  tmp_dir='/tmp/kinetics',
                  num_attempts=5,
                  url_base='https://www.youtube.com/watch?v='):
    # Defensive argument checking.
    assert isinstance(video_identifier, str), 'video_identifier must be string'
    assert isinstance(output_filename, str), 'output_filename must be string'
    assert len(video_identifier) == 11, 'video_identifier must have length 11'

    status = False
    # Construct command line for getting the direct video link.
    tmp_filename = os.path.join(tmp_dir,
                                '%ss' % "temp1234")
    print(tmp_filename)
    command = ['youtube-dl',
               '--quiet', '--no-warnings',
               '-f', 'mp4',
               '-o', '"%s"' % tmp_filename,
               '"%s"' % (url_base + video_identifier)]
    command = ' '.join(command)
    attempts = 0
    while True:
        try:
            output = subprocess.check_output(command, shell=True,stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            attempts += 1
            if attempts == num_attempts:
                return status, err.output
        else:
            break
        
    # tmp_filename = glob.glob('%s*' % tmp_filename.split('.')[0])[0]
    # Construct command to trim the videos (ffmpeg required).
    command = ['ffmpeg',
               '-i', '"%s"' % tmp_filename,
               '-ss', str(start_time),
               '-t', str(end_time - start_time),
               '-c:v', 'libx264', '-c:a', 'copy',
               '-threads', '1',
               '-loglevel', 'panic',
               '"%s"' % output_filename]
    command = ' '.join(command)
    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        return status, err.output

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    os.remove(tmp_filename)
    return status, 'Downloaded'

def download_clip_wrapper(row, label_to_dir, trim_format, tmp_dir):
    output_filename = construct_video_filename(row, label_to_dir,
                                               trim_format)
    clip_id = os.path.basename(output_filename).split('.mp4')[0]
    if os.path.exists(output_filename):
        status = tuple([clip_id, True, 'Exists'])
        return status
    downloaded, log = download_clip(row['video-id'], output_filename,
                                    row['start-time'], row['end-time'],
                                    tmp_dir=tmp_dir)
    print(log)

def main():
    dataset = parse_kinetics_annotations()
    print(dataset)
    # Creates folders where videos will be saved later.
    label_to_dir = create_video_folders(dataset, "vids", "tmp")
    # Download all clips.
    status_lst = []
    trim_format='%06d'
    tmp_dir='/tmp'
    for i, row in dataset.iterrows():
        download_clip_wrapper(row, label_to_dir, trim_format, tmp_dir)

if __name__ == '__main__':
    main()