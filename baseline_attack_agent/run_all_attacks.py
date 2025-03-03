import os
import threading
import queue
import subprocess
import json

if False:
    LOGDIR = "attack_log_o3"
    START_GPU = 4
    END_GPU = 8
    ATTACK_LLM = "o3-mini"

if True:
    LOGDIR = "attack_log_sonnet_o3_supervisor"
    START_GPU = 4
    END_GPU = 8
    ATTACK_LLM = "sonnet-supervisor-o3"
    
if False:
    LOGDIR = "attack_log_haiku"
    START_GPU = 0
    END_GPU = 4
    ATTACK_LLM = "sonnet-40"

def find_config_files(root_dir):
    config_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'config.json' in filenames:
            config_files.append(os.path.join(dirpath, 'config.json'))
    return config_files

def worker(gpu_id, job_queue, results_lock, results_list):
    while True:
        try:
            config_file, idx = job_queue.get_nowait()
        except queue.Empty:
            break
        config_dir = os.path.dirname(config_file)
        fpath = LOGDIR + "/"+config_dir.split("/")[-1]+"-"+str(idx)

        if os.path.exists(fpath+".log"):
            print("Skipping completed job", fpath+".log")
            continue
        
        cmd = ["python", "attack.py", config_dir, str(gpu_id), str(idx), fpath + ".tar", ATTACK_LLM]
        print(f"GPU {gpu_id}: Processing {config_file}, idx {idx}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            success = result.returncode == 0
            output = result.stdout + result.stderr
        except Exception as e:
            success = False
            print("Crashed", e)
            output = str(e)
        open(fpath+".log","w").write(output)
        print(repr(output))
        # Acquire lock to update results
        with results_lock:
            results_list.append({
                'config_file': config_file,
                'gpu_id': gpu_id,
                'success': success,
                'output': output
            })
        job_queue.task_done()

def main():
    root_dir = '../defenses'  # Replace with your root directory
    config_files = find_config_files(root_dir)
    job_queue = queue.Queue()
    for config_file in sorted(config_files):
        print(config_file)
        j = json.load(open(config_file))
        if 'defenses' in j:
            print(config_file, len(j['defenses']))
            for i in range(len(j['defenses'])):
                job_queue.put((config_file, i))
    results_list = []
    results_lock = threading.Lock()
    threads = []
    for gpu_id in range(START_GPU, END_GPU):
        t = threading.Thread(target=worker, args=(gpu_id, job_queue, results_lock, results_list))
        t.start()
        threads.append(t)
    # Wait for all jobs to be processed
    job_queue.join()
    # Wait for all threads to finish
    for t in threads:
        t.join()
    # Output the results
    for result in results_list:
        print(f"File: {result['config_file']}, GPU: {result['gpu_id']}, Success: {result['success']}")
        print(f"Output:\n{result['output']}\n")

if __name__ == "__main__":
    main()
