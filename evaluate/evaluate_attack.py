import numpy as np
import pickle
import sys
import io
import tarfile
import os
import subprocess
import anthropic
import copy
import json
from docker_client import DockerClient
import subprocess

import os
import numpy as np
from PIL import Image


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

def get_utility(PYTHON, forward, dataset):
    try:
        out = container.run_bash(f"{PYTHON} {forward} {dataset}", timeout=3000).replace("\r\n","\n")

        if '\nLabels\n' in out:
            out = out.split("\nLabels\n")[-1].split()[:1000]
        elif out.startswith("Labels\n"):
            out = out.split("Labels\n")[1].split()[:1000]
        else:
            print("CRASH!")
            print((out))
            raise
        out = np.array(list(map(int,out)))
        return out
    except:
        print("CRASH", PYTHON, forward, dataset)
        return 1.0
    
def numpy_to_tar(images):
    # Create a BytesIO object to hold the tar file in memory
    tar_bytes = io.BytesIO()
    
    # Create a TarFile object
    with tarfile.open(fileobj=tar_bytes, mode='w') as tar:
        for i, img_array in enumerate(images):
            # Transpose the image from [3, 32, 32] to [32, 32, 3]
            if len(img_array) == 2:
                pass
            elif img_array.shape[2] < img_array.shape[1]:
                pass
            else:
                img_array = np.transpose(img_array, (1, 2, 0))

            # Convert to uint8 if not already
            if img_array.dtype != np.uint8:
                img_array = (img_array * 255).astype(np.uint8)
            
            # Create a PIL Image
            img = Image.fromarray(img_array)
            
            # Save the image to a BytesIO object
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Create a TarInfo object
            tar_info = tarfile.TarInfo(name=f'{i:04d}.png')
            tar_info.size = img_bytes.getbuffer().nbytes
            
            # Add the image to the tar file
            tar.addfile(tar_info, img_bytes)
    
    # Reset the BytesIO object to the beginning
    tar_bytes.seek(0)
    return tar_bytes.getvalue()

import tarfile
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path

def compute_linf_norm(img1_array, img2_array):
    return np.max(np.abs(img1_array - img2_array))

def check_images(tar_path, reference_dataset_path, threshold=0.03):
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            img_bytes = tar.extractfile(member).read()
            img1 = np.array(Image.open(BytesIO(img_bytes))).astype(float) / 255.0
            
            ref_path = Path(reference_dataset_path) / member.name
            img2 = np.array(Image.open(ref_path)).astype(float) / 255.0
            
            if compute_linf_norm(img1, img2) > threshold:
                return False
    
    return True


def verify(defconfig, advx):
    if defconfig['dataset'] in ['cifar10', 'cifar-10']:
        dataset = "cifar10_test"
        distortion = 0.032
    elif defconfig['dataset'] in ['mnist']:
        dataset = "mnist_test"
        distortion = 0.31
    elif defconfig['dataset'] in ['imagenet']:
        dataset = "imagenet_test"
        distortion = 0.032

    PYTHON = config.get('python', 'python')
        
    imgs = {}
    for f in os.listdir(f"../{dataset}/"):
        imgs[f"/usr/src/app/{dataset}/{f}"] = open(f"../{dataset}/{f}","rb").read()
        
    container.write_file(imgs)

    if not check_images(advx, f"../{dataset}/", distortion):
        return clean_acc, clean_acc # violated distortion bound
    
    ground_truth = open(f"../{dataset}/labels.txt").read().split()
    ground_truth = np.array(list(map(int,ground_truth)))

    clean_labels = get_utility(PYTHON, defconfig['forward'], dataset)
    clean_acc = np.mean(ground_truth == clean_labels)

    #adv_tar = numpy_to_tar(advx)
    #open("/tmp/a.tar","wb").write(adv_tar)
    
    container.write_file({"/usr/src/app/adv_imgs/adv.tar": open(advx,"rb").read()})
    container.run_bash("cd adv_imgs; tar -xf adv.tar; rm adv.tar")
    container.write_file({"/usr/src/app/adv_imgs/labels.txt": "0\n"*1000})
    
    adv_labels = get_utility(PYTHON, defconfig['forward'], "adv_imgs")

    robust_acc = np.mean((ground_truth == adv_labels) | (adv_labels == -1))

    return clean_acc, robust_acc


LOGPATH = sys.argv[1]
GPUS = sys.argv[2] if len(sys.argv) > 2 else "0"
DEFPATH = "../defenses"

for root_base_ in os.listdir(DEFPATH):
    root_base = os.path.join(DEFPATH, root_base_)
    config = json.load(open(os.path.join(root_base, 'config.json')))

    for i,defense in enumerate(config['defenses'].values()):
        advx_path = os.path.join(LOGPATH, root_base_ + "-" + str(i)+".tar")
        if os.path.exists(advx_path):
            print("Checking attack success rate for defense", root_base, defense['forward'])
            try_load_container(config)
            CONTAINER_NAME = config['docker_env_name']
            container = init()
            clean_acc, robust_acc = verify(defense, advx_path)
            print('path', root_base, 'idx', i, 'clean:', clean_acc, 'robust:', robust_acc)
            
            container.remove_container()

