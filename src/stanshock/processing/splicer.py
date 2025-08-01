from PIL import Image
import os
import glob
import cv2
import shutil
from datetime import datetime
import numpy as np
def splicer(t_sim, image_folder, output_dir, copy_path):
    # image_folder = 'figures/anim'
    # output_dir = 'figures'
    name = 'property_anim'
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{time_stamp}.mp4"
    output_video = os.path.join(output_dir, filename)

    image_pattern = os.path.join(image_folder, 'test_*.png')
    image_files = sorted(glob.glob(image_pattern))

    N = len(image_files)
    k = 400 #scaling factor from sim time to video time: 0.01 s sim time gets 4 sec video
    fps = np.ceil(N / (k*t_sim)).astype(np.int64)

    first_frame = cv2.imread(image_files[0])
    height, width, layers = first_frame.shape
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'avc1' if 'mp4v' gives issues
    out = cv2.VideoWriter(output_video, fourcc, fps, size)
    for file in image_files:
        img = cv2.imread(file)
        out.write(img)

    out.release()
    dest_path = os.path.join(copy_path, filename)
    shutil.copyfile(output_video, dest_path)
    print(f"Video saved to {output_video} and copied to {copy_path}.")