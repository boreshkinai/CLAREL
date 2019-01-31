#!/usr/bin/python

# Copyright (c) 2018 ELEMENT AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# docker pull tensorflow/tensorflow:1.8.0-devel-gpu-py3
# docker tag tensorflow/tensorflow:1.8.0-devel-gpu-py3 images.borgy.elementai.lan/tensorflow/tensorflow:1.8.0-devel-gpu-py3
# docker push images.borgy.elementai.lan/tensorflow/tensorflow:1.8.0-devel-gpu-py3
# chmod +x zeroshot-pairwise/train.py

from common.gen_experiments import gen_experiments_dir, find_variables

import os
import time
import argparse
import pathlib

# os.system('pip install gitpython --user')
import git

os.environ['LANG'] = 'en_CA.UTF-8'

if __name__ == "__main__":
    exp_description = "test"

    params = dict(
        repeat=list(range(0, 1)),  # used to repeate the same experiment
        dataset='cvpr2016_cub',
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root_dir', type=str, default="experiments_zeroshot_pairwise", 
                        help='folder to save the batch of experiments.')
    parser.add_argument('--project_dir', type=str, default="dev/zeroshot-pairwise", 
                        help='folder in /home/user where project is cloned.')
    
    exp_root_dir = parser.parse_known_args()[0].exp_root_dir
    project_dir = parser.parse_known_args()[0].project_dir
    
    home_folder = os.path.expanduser("~")
    
    if not os.path.isdir(os.path.join(home_folder, exp_root_dir)):
        pathlib.Path(os.path.join(home_folder, exp_root_dir)).mkdir(parents=True)
    exp_tag = '_'.join(find_variables(params))  # extract variable names
    exp_dir = os.path.join(home_folder, exp_root_dir,
                           "%s_%s_%s" % (time.strftime("%y%m%d_%H%M%S"), exp_tag, exp_description))

    project_path = os.path.join(home_folder, project_dir)

    # This is for the reproducibility purposes
    repo_path = '/mnt' + project_path
    repo = git.Repo(path=repo_path)
    params['commit'] = repo.head.object.hexsha

    borgy_args = [
        "--image=images.borgy.elementai.lan/tensorflow/zeroshot-pairwise-tensorflow-1.8.0",
        "-w", "/",
        "-e", "PYTHONPATH=%s" % repo_path,
        "-v", "/mnt/datasets/public/:/mnt/datasets/public/",
        "-v", "/mnt/home/boris/:/mnt/home/boris/",
        "--cpu=2",
        "--gpu=1",
        "--mem=16",
        "--restartable"
    ]

    cmd = os.path.join(repo_path, "train.py")

    gen_experiments_dir(params, exp_dir, exp_description, cmd, blocking=True, borgy_args=borgy_args)
