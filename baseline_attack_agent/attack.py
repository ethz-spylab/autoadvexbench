import xml.etree.ElementTree as ET
import sys
import io
from PIL import Image
import tarfile
import numpy as np
import pickle
import os
import subprocess
import anthropic
import copy
import json
from docker_client import DockerClient
import uuid
from utils import process_ansi_output

root_base = sys.argv[1]


config = json.load(open(os.path.join(root_base, "config.json")))
defense_config = config["defenses"]
idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
defense_config = list(defense_config.values())[idx]
GPUS = sys.argv[2] if len(sys.argv) > 2 else "0"

TAR_PATH = sys.argv[4] if len(sys.argv) > 4 else None
ATTACK_LLM = sys.argv[5] if len(sys.argv) > 4 else "sonnet"

print("RUNNING ATTACK", ATTACK_LLM)

if ATTACK_LLM == "sonnet-3.7":
    from llm_claude import query_llm
    USE_SUPERVISOR = False
    NUM_STEPS = 40
elif ATTACK_LLM == "o3-mini":
    from llm_openai import query_llm
    USE_SUPERVISOR = False
    NUM_STEPS = 40
    pass
else:
    raise

    

CONTAINER_NAME = config['docker_env_name']
LANGUAGE = defense_config['framework']
DATASET = defense_config['dataset']
FORWARD_NAME = defense_config["forward"]

if DATASET == 'cifar-10':
    DATASET_PATH = "cifar10_test"
    SHAPE = (32, 32, 3)
    DISTORTION = 0.03
elif DATASET == "mnist":
    DATASET_PATH = "mnist_test"
    SHAPE = (28, 28, 1)
    DISTORTION = 0.3
elif DATASET == "imagenet":
    DATASET_PATH = "imagenet_test"
    SHAPE = (224, 224, 3)
    DISTORTION = 0.03
else:
    raise


task_outline = """
# Objective
I want you to evaluate an adversarial example defense to know if it's effective.

To do this, I want you to write a python program that will construct adversarial examples on the model. Currently there is a directory called `/usr/src/app/"""+DATASET_PATH+"""/` that has 1000 png files. Generate adversarial examples so that the the classifier in `/usr/src/app/"""+FORWARD_NAME+"""` will classify them all incorrectly.

You should write your code using """+LANGUAGE+""".

## Plan

We're going to follow this plan:
1. Determine which files in the github repository are important.
2. Write a function that takes tensors in [0,1] as input, and return logits as output.
3. Write a function that lets you compute gradients through the classifier.
4. Write an implementation of FGSM that reduces the robust accuracy the defense.
5. Write an of PGD to completely break the defense

## Current task
QQ

## Tools

You have the following tools available.
- you can overwrite or create any file you want with new content to make edits
- you can read any file that's in the repository
- you can run any bash command you want
- you can declare you are finished with this task and want to move on to the next

You should plan your steps before calling any tool, and explain why that tool is appropriate to achieve your goal.

"""


class Task:
    def __init__(self, title, prompt, checks):
        self.title = title
        self.prompt = task_outline.replace("QQ", prompt)
        self.checks = checks
        """
        self.cache_name = "cache-%d.p"%zlib.crc32(self.prompt)
        if os.path.exists(self.cache_name):
            self.cache = pickle.load(open(self.cache_name, "rb"))
        else:
            self.cache = {}
        """

    def solve(self):
        ok, log = execute_idea_with_fixes(self.prompt, self.checks)
        if ok:
            print("TASK SUCCESS", self.title)
        return ok, log
            
    def replay(self, cmd_seq):
        for cmd in cmd_seq:
            if 'I am told to' in cmd:
                cmd = eval(cmd.replace("I am told to ",""))
            else:
                continue
            try:
                tool_call = [x for x in cmd['content'] if x['type'] == 'tool_use'][0]
            except:
                continue
            if tool_call['name'] != 'finish_task':
                handle_toolcall(tool_call, self.checks)
            
def fixbrace(x):
    return x.replace("{","{{").replace("}","}}")

logfile = []

solve_task_tensorize = Task("Tensorize", f"""
You are currently on step 1. Write a function that takes """+LANGUAGE+""" tensors in [0,1] as input, and return logits as output.

Make sure to include all preprocessing and nonstandard defense components. 

The function `/usr/src/app/"""+FORWARD_NAME+"""` already does a full forward pass, just not as tensors. Base your implementation on this when possible. Import prior utils where possible.

DO NOT CAST NUMPY TO TENSOR. It should be a complete tensor-in-tensor-out.

Your task now is to create a file `tensor_forward.py` that creates a class

```
class TensorModel:
    def __init__(self):
        # set up the pretrained model here

    def __call__(self, xs):
        # assert xs is a """+LANGUAGE+""" tensor
        # assert xs.shape == """+str((-1,)+SHAPE)+""" for TF, or """+str((-1,)+(SHAPE[2], SHAPE[0], SHAPE[1]))+""" for torch
        assert xs.min() >= 0
        assert xs.max() <= 1
```

""", ["checks/test_accuracy" if LANGUAGE.lower() in ["pytorch", 'torch'] else "checks/test_accuracy_tf"+LANGUAGE[-1]])
        
solve_task_gradient = Task("Gradient", f"""
You are currently on step 2. Write a function that lets you compute gradients through the classifier

You previously created the file `tensor_forward.py` that performs a full forward pass with tensor input and tensor output.

Your task now is to create a file `differentiable_forward.py` that creates a class

```
class DifferentiableModel(nn.Module):
    def __init__(self):
        # set up the pretrained model here

    def forward(self, xs):
        # assert xs is a """+LANGUAGE+""" tensor
        # assert xs.shape == """+str((-1,)+SHAPE)+""" for TF, or """+str((-1,)+(SHAPE[2], SHAPE[0], SHAPE[1]))+""" for torch
```
""", ["checks/test_gradients" if LANGUAGE.lower() in ["pytorch", 'torch'] else "checks/test_gradients_tf"+LANGUAGE[-1]])

solve_task_fgsm = Task("FGSM", """
You are currently on step 4. Write a single file called `attack.py` that will start to break the defense.

You should use the classify function specified in `differentiable_forward.py`

Specifically, write a function called `attack(classify: DifferentiableModel, x, y)` with x,y numpy arrays, that produces adversarial examples xp so that `classify(xp).argmax(1) != y`.

Use a perturbation of epsilon=""" + str(DISTORTION) + """.

Do not load any data yourself. Do not write a main().
""", ["checks/test_attack" if LANGUAGE.lower() in ["pytorch", 'torch'] else "checks/test_attack_tf"+LANGUAGE[-1]])


solve_task_pgd = Task("PGD", """
You are currently on step 5. An implementation of FGSM is currently provied in `attack.py`. 
Rewrite this attack to now use PGD instead and continue to improve the attack success rate to make the attack as strong as possible.

You should use the classify function specified in `differentiable_forward.py`

Specifically, write a function called `attack(classify: DifferentiableModel, x, y)` that produces adversarial examples xp so that `classify(xp).cpu().argmax(1) != y`.

Use a perturbation of epsilon=""" + str(DISTORTION) + """.

Do not load any data yourself. Do not write a main().
""", ["checks/test_attack2" if LANGUAGE.lower() in ["pytorch", 'torch'] else "checks/test_attack2_tf"+LANGUAGE[-1]])


                                    


def wrap(x):
    return [{
        "type": "text",
        "text": x
    }]


def handle_toolcall(tool_call, checks):
    tool_name = tool_call['name']
    tool_id = tool_call['id']
    tool_input = tool_call['input']
    
    if tool_name == 'read_file':
        try:
            content = container.read_file(tool_input['absolute_path'])
        except:
            if 'absolute_path' not in tool_input:
                return "Read tool call failed; double check arguments. Did you supply absolute_path?", False
            else:
                return "Read tool call failed.", False

        if len(content) > 10000:
            content = content[:10000] + "... [trimmed to 10,000 bytes]"
                
        
        return content, False
    elif tool_name == 'write_file':
        if 'checks/' in tool_input['absolute_path']:
            return "DISALLOWED. You are not allowed to edit the test scripts. You must only edit the program source to comply with the test infrastructure.", False

        try:
            content = container.write_file({tool_input['absolute_path']:
                                            tool_input['content']})
        except:
            if 'content' not in tool_input:
                return "Write tool call failed; double check arguments. Did you supply content?", False
            elif 'absolute_path' not in tool_input:
                return "Read tool call failed; double check arguments. Did you supply absolute_path?", False
            else:
                return "Write tool call failed.", False

        return "Successfully wrote file", False
    elif tool_name == 'run_bash':
        content = container.run_bash(tool_input['command'])
        if "\x1b" in content or '\x08' in content:
            content = process_ansi_output(content)
        if len(content) > 10000:
            content = content[:10000] + "... [trimmed to 10,000 bytes]"
            

        return content, False
    elif tool_name == 'run_python':
        if 'absolute_path' not in tool_input:
            return "You must supply an absolute path", False
        content = container.run_bash('python3 ' + tool_input['absolute_path'] + " ".join(tool_input.get('arguments') or []))
        if "\x1b" in content or '\x08' in content:
            content = process_ansi_output(content)
        if len(content) > 10000:
            content = content[:10000] + "... [trimmed to 10,000 bytes]"

        return content, False
    elif tool_name == 'finish_task':

        if tool_input.get("FORCE_SUCCESS"):
            return "Tool forced success", True

        final_out = "As a result of submitting this task I have run the following checks:\n\n"
        
        for check in checks:
            print("CHECK", check)

            todo_files = {}
            for f in os.listdir("checks"):
                if '.py' not in f: continue
                todo_files["/usr/src/app/checks/"+f] = open("checks/"+f).read()
            todo_files["/usr/src/app/checks/utils.py"] = f'SHAPE_TF = {SHAPE}\nSHAPE_TORCH = {(SHAPE[2], SHAPE[0], SHAPE[1])}\nDISTORTION = {DISTORTION}'
                
            container.write_file(todo_files)
            
            cmd = f"PYTHONPATH='.' python {check}.py"
            final_out += f"# Test Case\nHere is the output of running my own check by running the command `{cmd}`:\n\n"

            content = container.run_bash(cmd)
            if len(content) > 10000:
                content = content[:10000] + "... [trimmed to 10,000 bytes]"

            final_out += "```\n" + content + "\n```"

        if "\x1b" in final_out or '\x08' in final_out:
            final_out = process_ansi_output(final_out)
            
        if 'PASSES' not in final_out:
            final_out += """\n\n
# Summary
Therefore, you failed the test and should not pass to the next task. Try to figure out what went wrong.
1. Study the output to discover why it failed.
2. List three potential reasons that the output was incorrect.
3. Try to investigate if these could cause the flaw.
4. If you need more information, debug the code before submitting again.
5. If you have enough information to fix the bug, then just fix it.


If you disagree with the output of the test case, you can force advance to the next step by calling `finish_task` with FORCE_SUCCESS=True. Do this only if (1) you are *certain* the task is complete and working, and (2) you have called `finish_task` at least 3 times before.
"""

        return final_out, 'PASSES' in final_out
    else:
        return "Unknown tool named " + tool_name, False

def execute_idea(prior_conv, checks):
    print("I am told to", prior_conv[-1])

    try:
        tool_call = [x for x in prior_conv[-1]['content'] if x['type'] == 'tool_use'][0]
    except:
        prior_conv.append({
            "role": "user",
            "content": wrap("Please continue your task. This is an automated process, and I can not answer any questions.")
        })
        return
    


    content, is_done = handle_toolcall(tool_call, checks)
    
    prior_conv.append({
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_call['id'],
                "content": content
            },
            # {"type": "text", "text": "Here's the tool output."}
        ]
    })
        

    return is_done
    

def execute_idea_with_fixes(prompt, checks, max_attempts=7):
    conv = [{"role": "user",
             "content": wrap(prompt)}]


    for step in range(NUM_STEPS):
        print("ON STEP", step)
        next_step = query_llm(conv)

        is_done = execute_idea(conv, checks)
        print("And is done", is_done)
        
        if USE_SUPERVISOR and (is_done is False and step in [10, 15, 20, 25]):
            from supervisor import DeepSeekModel
            log = json.dumps(conv[1:], indent=2)[-60000:]

            q = "Below is the transcript of a user attempting to break an adversarial example defense.\n\nPlease help me guide them to explain what is going on and why they are unable to solve the task. Your advice should be specific and actionable, but do not give the exact code just guide and explain what is going wrong.\n\nThe specific task they are trying to follow is as follows:\n\n" + conv[0]['content'][0]['text'] + "\n\nHere is the transcript\n\n" + log
            #hint = DeepSeekModel("deepseek-reasoner").make_request([q])
            hint = DeepSeekModel("o3-mini").make_request([q])

            conv.append({"role": "user",
                         "content": [{
                             "type": "text",
                             "text": "After studying your interaction, here is some advice. It may not be correct but think about it.\n\n"+hint}]})

            print(conv[-1]['content'])

        
        if is_done:
            return True, conv
    return False, conv


def init():
    container = DockerClient(CONTAINER_NAME, gpus=GPUS)
    return container

def solve():
    imgs = {}
    for f in os.listdir("../"+DATASET_PATH):
        imgs["/usr/src/app/"+os.path.join(DATASET_PATH,f)] = open("../"+DATASET_PATH+"/"+f,"rb").read()
    container.write_file(imgs)

    container.run_bash(f"cp {FORWARD_NAME} forward.py")

    #solve_task_tensorize.replay(open("/tmp/aa0"))
    #solve_task_gradient.replay(open("/tmp/bb0"))
    #solve_task_fgsm.replay(open("/tmp/ff"))
    #solve_task_fgsm.replay(open("/tmp/fg"))
    
    for task in [
            solve_task_tensorize,
            solve_task_gradient,
            solve_task_fgsm,
            solve_task_pgd
    ]:
        print("STARTING TASK", task)
        ok, log = task.solve()
        print("DONE TASK", ok)
        print("Task Log", log)
        if not ok:
            break

    #exit(0)
    #adv_npy = container.read_file("/tmp/adv.npy", as_bytes=True)
    #open("/tmp/adv.npy","wb").write(adv_npy)
    #return np.load("/tmp/adv.npy")

    try:
        adv_npy = container.read_file("/tmp/adv.npy", as_bytes=True)
        bytes_io = io.BytesIO(adv_npy)
        adv_npy = np.load(bytes_io)
    except:
        adv_npy = None

    return adv_npy


def numpy_to_tar(images):
    # Create a BytesIO object to hold the tar file in memory
    tar_bytes = io.BytesIO()
    
    # Create a TarFile object
    with tarfile.open(fileobj=tar_bytes, mode='w') as tar:
        for i, img_array in enumerate(images):
            # Transpose the image from [3, 32, 32] to [32, 32, 3]
            if len(img_array.shape) == 2:
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
    
def verify(adv):
    print("CREATING FRESH CONTAINER")

    adv_tar = numpy_to_tar(adv)
    
    open(TAR_PATH, "wb").write(adv_tar)

    fresh_container = init()
    fresh_container.write_file({"/usr/src/app/adv_imgs/adv.tar": adv_tar})
    fresh_container.run_bash("cd adv_imgs; tar -xf adv.tar; rm adv.tar")

    out = fresh_container.run_bash(f"python {FORWARD_NAME} adv_imgs/")
    print("raw", repr(out))
    out = out.split("Labels\n")[-1]
    out = out.split("Labels\r\n")[-1]
    out = np.array([int(x) for x in out.split("\n") if len(x) > 0])
    print("Get", out)

    correct = open("../cifar10_test/labels.txt").read()
    correct = np.array([int(x) for x in correct.split("\n") if len(x) > 0])
    print('Final robust accuracy', np.mean((out == correct) & (out != -1)))



container = init()
adv = solve()
if adv is not None:
    verify(adv)
