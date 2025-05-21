import numpy as np
import pickle
import sys
import os
import subprocess
import anthropic
import copy
import json
from docker_client import DockerClient
import subprocess

def check_container_exists(container_name):
    try:
        result = subprocess.run(["docker", "inspect", container_name], capture_output=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False

def build_container(dockerfile_path, container_name):
    try:
        # Change to the script's directory
        original_dir = os.getcwd()
        os.chdir(dockerfile_path.replace("Dockerfile",""))
        subprocess.run(["docker", "build", "-t", container_name, "."], check=True)
        os.chdir(original_dir)
        print(f"Container {container_name} built successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error building container: {e}")
        return False

def try_load_container(config):
    CONTAINER_NAME = config['docker_env_name']

    if check_container_exists(CONTAINER_NAME):
        print(f"Container {CONTAINER_NAME} already exists.")
    else:
        print(f"Container {CONTAINER_NAME} does not exist. Building...")
        dockerfile_path = os.path.join(root_base, 'Dockerfile')
        if build_container(dockerfile_path, CONTAINER_NAME):
            if check_container_exists(CONTAINER_NAME):
                print(f"Container {CONTAINER_NAME} now exists after building.")
            else:
                print(f"Container {CONTAINER_NAME} still does not exist after building. Please check for errors.")
                exit(1)
        else:
            print("Failed to build the container.")
            exit(1)


root_base = sys.argv[1]

idx = int(sys.argv[2]) if len(sys.argv) > 2 else None

config = json.load(open(os.path.join(root_base, 'config.json')))


try_load_container(config)

CONTAINER_NAME = config['docker_env_name']

GPUS = "0"

import uuid

def init():
    container = DockerClient(CONTAINER_NAME, gpus=GPUS)
    files = {}
    for root,_,fs in os.walk(root_base):
        for f in fs:
            files[os.path.join(root, f).replace(root_base+"/","")] = open(os.path.join(root, f),"rb").read()
    container.write_file(files)
    container.run_bash("echo export TF_CPP_MIN_LOG_LEVEL=3 >> ~/.bashrc")
    return container


def verify(defconfig):
    if defconfig['dataset'] in ['cifar10', 'cifar-10']:
        dataset = "cifar10_test"
    elif defconfig['dataset'] in ['mnist']:
        dataset = "mnist_test"
    elif defconfig['dataset'] in ['imagenet']:
        dataset = "imagenet_test"

    PYTHON = config.get('python', 'python')
        
    imgs = {}
    for f in os.listdir(f"../{dataset}/"):
        imgs[f"/usr/src/app/{dataset}/{f}"] = open(f"../{dataset}/{f}","rb").read()
    container.write_file(imgs)

    out = container.run_bash(f"{PYTHON} {defconfig['forward']} {dataset}", timeout=3000).replace("\r\n","\n")

    if '\nLabels\n' in out:
        out = out.split("\nLabels\n")[-1].split()[:1000]
    elif out.startswith("Labels\n"):
        out = out.split("Labels\n")[1].split()[:1000]
    else:
        print("CRASH!")
        print((out))
        exit(1)
    out = np.array(list(map(int,out)))
    ground_truth = open(f"../{dataset}/labels.txt").read().split()
    mean = np.mean(np.array(list(map(int,ground_truth)) == out))
    print('mean acc', mean)



container = init()
if idx is not None:
    defense = list(config['defenses'].values())[idx]
    print(defense['forward'])
    verify(defense)
    exit(0)
for defense in config['defenses'].values():
    print("Verifying", defense['forward'])
    verify(defense)
