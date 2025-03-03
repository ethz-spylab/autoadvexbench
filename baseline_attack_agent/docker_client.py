import requests
import os
import base64

class DockerClient:
    def __init__(self, container_name, container_id=None, base_url='http://127.0.0.1:5000', gpus=None):
        self.base_url = base_url
        if container_id is None:
            self.create_container(container_name, gpus)
        else:
            self.container_id = container_id

    def create_container(self, container_name, gpus=None):
        url = f'{self.base_url}/new'
        data = {'container_name': container_name, 'gpus': gpus}
        response = requests.post(url, json=data)
        self.container_id = response.json()['container_id']
        return response.json()

    def write_file(self, files):
        url = f'{self.base_url}/write'
        for k,v in files.items():
            #print("Writing file", k, repr(v[:50]))
            is_b64 = False
            try:
                out = base64.b64decode(v)
                if base64.b64encode(out) == v and len(v) >= 16:
                    print("BASE64 DECODED??")
                    print(v)
                    is_b64 = True
            except:
                pass
            if is_b64:
                print("HOW DID THIS HAPPEN")
                print("TOLD TO WRITE", {k:v[:100] for k,v in files.items()})
                exit(1)
            if type(v) == bytes:
                files[k] = base64.b64encode(v).decode("ascii")
            else:
                files[k] = base64.b64encode(bytes(v,'utf8')).decode("ascii")
        data = {'container_id': self.container_id,
                'files': files}
        response = requests.post(url, json=data)
        return response.json()

    def write_dir(self, directory):
        todo_files = {}
        for root,_,fs in os.walk(directory):
            for f in fs:
                todo_files[os.path.join(root,f)] = open(os.path.join(root,f),"rb").read()
        self.write_file(todo_files)

    def run_command(self, command, timeout=600):
        url = f'{self.base_url}/run'
        data = {'container_id': self.container_id, 'command': command, 'timeout': timeout}
        response = requests.post(url, json=data)
        return response.json()['output']

    def stop_container(self):
        url = f'{self.base_url}/stop'
        data = {'container_id': self.container_id}
        response = requests.post(url, json=data)
        return response.json()

    def read_file(self, file_path, as_bytes=False):
        url = f'{self.base_url}/read'
        data = {'container_id': self.container_id, 'file_path': file_path}
        response = requests.post(url, json=data)
        if as_bytes:
            return response.content
        else:
            return response.text

    def run_bash(self, cmds):
        self.write_file({"/usr/src/app/tmp/run.sh": "export TF_CPP_MIN_LOG_LEVEL=3\n"+cmds})
        return self.run_command("bash /usr/src/app/tmp/run.sh")

# Example usage
if __name__ == '__main__':
    print("Creating client")
    client = DockerClient('ab')
    print("Created client")
    
    # Write a file in the container
    files_to_write = {'test.txt': 'Hello, World!'}
    write_response = client.write_file(files_to_write)
    print('write',write_response)
    
    # Run a command in the container
    run_response = client.run_command('cat test.txt')
    print('cat',run_response)
    
    # Read a file from the container
    read_response = client.read_file('test.txt')
    print('read',read_response)
    
    # Stop and remove the container
    stop_response = client.stop_container()
    print(stop_response)
