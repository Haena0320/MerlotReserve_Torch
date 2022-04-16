import subprocess
from threading import Thread
from glob import glob
import os,sys
import csv
from tqdm import tqdm
import tensorflow as tf

channels_video_ids = []
#with tf.io.gfile.GFile("/mnt2/user15/merlot_r/merlot_reserve_backup/data/video_csv_split/yttemporal1b_ids_val.csv", 'r') as f:
#for idx in range(200):
#    with tf.io.gfile.GFile(f"/mnt2/user15/merlot_r/merlot_reserve_backup/data/video_log/video_log_{idx}.csv", "r") as f:
#        reader = csv.DictReader(f)
#        for i, row in enumerate(reader):
#            channels_video_ids.append(row['video_id'])
#        print(f'{idx} video file has {i} videos..oh my god')

#print(len(channels_video_ids))

 #for i, x in tqdm(enumerate(channels_video_ids)):
 #    file_n = i % 10 # split file num
 #    file_dest = f'/mnt2/user15/merlot_r/merlot_reserve_backup/data/video_csv_split/video_split_{file_n}.csv'
 #    fieldnames = ['video_id']
 #    if not os.path.exists(file_dest):
 #        with open(file_dest, 'w') as f:
 #            writer = csv.DictWriter(f, fieldnames=fieldnames)
 #            writer.writeheader()
 #            writer.writerow({'video_id': x})
 #    else:
 #        with open(file_dest, 'a') as f:
 #            writer = csv.DictWriter(f, fieldnames=fieldnames)
 #            writer.writerow({'video_id': x})

import os, sys


def make_video(file_num, input_csv, output_csv):
    ls = subprocess.run(f"python /mnt2/user15/merlot_r/merlot_reserve_backup/data/process.py -fold {file_num} -ids_fn {input_csv} -ids_fn_o {output_csv}", shell=True, text=True).stdout

tasks = []
input_csv = "/mnt2/user15/merlot_r/merlot_reserve_backup/data/video_csv_split_1"
output_csv = "/mnt2/user15/merlot_r/merlot_reserve_backup/data/video_log_1"

import time
for time_ in tqdm(range(4)): # CYCLE
    tt = range(time_*3, (time_+1)*3) # FILE NUM IN SAME TIME
    if 9 in tt:
        tt = tt[:-2]
    for file_num in tt:
        file_ = 200 + file_num
        input_csv_ = os.path.join(input_csv, f"video_split_{file_num}.csv")
        output_csv_ = os.path.join(output_csv, f"video_log_{file_num}.csv")
        thread = Thread(target=make_video, args=(file_num, input_csv_,output_csv_))
        tasks.append(thread)
        thread.start()
        time.sleep(5)

    for task in tasks:
        task.join()
    print("one cycle is finished")


