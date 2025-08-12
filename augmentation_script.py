import os
import sys
import subprocess

src = sys.argv[1]

folder_lst = os.listdir(src)

max_dir = ""
max_size = 0
for dir in folder_lst:
    size_dir = len(os.listdir(f"{src}/{dir}"))

    if size_dir > max_size:
        max_dir = dir
        max_size = size_dir

folder_lst.remove(max_dir)

for dir in folder_lst:
    img_lst = os.listdir(f"{src}/{dir}")
    size_dir = len(img_lst)

    diff_size = max_size - size_dir
    transform_size = diff_size // 6

    for i in range(transform_size):
        subprocess.run(
            [
                "python3",
                "/home/vallun/42/Outer/Leaffliction/Augmentation.py",
                f"{src}/{dir}/image ({i + 1}).JPG",
            ]
        )
