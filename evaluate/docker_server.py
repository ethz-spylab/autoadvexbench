from flask import Flask, request, jsonify
import base64
import docker
import io
import tarfile
import time

app = Flask(__name__)
client = docker.from_env()

containers = {}


def make_tar(files):
    file_like_object = io.BytesIO()
    tar = tarfile.TarFile(fileobj=file_like_object, mode='w')
    
    for file_name, file_content in files.items():
        file_content = base64.b64decode(file_content)
        tarinfo = tarfile.TarInfo(name=file_name)
        tarinfo.size = len(file_content)
        tarinfo.mtime = time.time()
        tar.addfile(tarinfo, io.BytesIO(file_content))

    tar.close()

    file_like_object.seek(0)

    return file_like_object


@app.route('/new', methods=['POST'])
def create_container():
    data = request.json
    container_name = data.get('container_name')
    gpus = data.get('gpus')
    
    if not container_name:
        return jsonify({"error": "container_name is required"}), 400
    
    if gpus is not None:
        device_requests = [
            docker.types.DeviceRequest(device_ids=[gpus], capabilities=[['gpu']])
        ]
    else:
        device_requests = None

    try:
        container = client.containers.run(
            container_name,
            detach=True,
            tty=True,
            device_requests=device_requests,
        )
        containers[container.id] = container
        return jsonify({"container_id": container.id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/remove/<container_id>', methods=['DELETE'])
def remove_container(container_id):
    try:
        if container_id in containers:
            container = containers[container_id]
            container.remove(force=True)  # force=True removes even if running
            del containers[container_id]
            return jsonify({"message": "Container removed successfully"}), 200
        else:
            return jsonify({"error": "Container not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500    
    
@app.route('/write', methods=['POST'])
def write_file():
    data = request.json
    container_id = data.get('container_id')
    files = data.get('files')
    
    if not container_id or not files:
        return jsonify({"error": "container_id and files are required"}), 400
    
    try:
        container = containers.get(container_id)
        if not container:
            return jsonify({"error": "container not found"}), 404

        tarfile = make_tar(files)
        container.put_archive("/", tarfile)
        
        return jsonify({"message": "files written successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/run', methods=['POST'])
def run_command():
    data = request.json
    command = data.get('command')
    container_id = data.get('container_id')
    timeout = data.get('timeout') or "600"

    if not command or not container_id:
        return jsonify({"error": "command and container_id are required"}), 400

    try:
        container = containers.get(container_id)
        if not container:
            return jsonify({"error": "container not found"}), 404

        # Use full path to timeout
        timeout_command = f"/usr/bin/timeout {timeout}s {command}"
        result = container.exec_run(timeout_command, tty=True)
        
        if result.exit_code == 124:  # timeout's exit code for timeout
            return jsonify({
                "output": "Error: Request timed out after {timeout} seconds.\nPartial STDOUT:\n" + result.output.decode('utf-8')
            }), 408
            
        return jsonify({"output": result.output.decode('utf-8')}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stop', methods=['POST'])
def stop_container():
    data = request.json
    container_id = data.get('container_id')
    
    if not container_id:
        return jsonify({"error": "container_id is required"}), 400
    
    try:
        container = containers.pop(container_id, None)
        if not container:
            return jsonify({"error": "container not found"}), 404
        container.stop()
        container.remove()
        return jsonify({"message": "container stopped and removed successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/read', methods=['POST'])
def read_file():
    data = request.json
    container_id = data.get('container_id')
    file_path = data.get('file_path')
    
    if not container_id or not file_path:
        return "", 500
    
    try:
        container = containers.get(container_id)
        if not container:
            return "", 500

        result = container.exec_run(f"cat {file_path}")
        
        if result.exit_code != 0:
            return "", 500

        return result.output, 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
