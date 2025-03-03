import time
import anthropic
import numpy as np
import copy

tools=[
    {
        "name": "write_file",
        "description": "Write data to a file at a particular location. Overwrites the file if it already exists, or creates a new file if it does not.",
        "input_schema": {
            "type": "object",
            "properties": {
                "absolute_path": {
                    "type": "string",
                    "description": "Absolute path to the file you would like to write",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file.",
                }
            },
            "required": ["absolute_path", "content"],
        },
    },
    {
        "name": "read_file",
        "description": "Read data from a file at a particular location. Returns an error if the file does not exist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "absolute_path": {
                    "type": "string",
                    "description": "Absolute path to the file you would like to read"
                }
            },
            "required": ["absolute_path"]
        }
    },
    {
        "name": "run_bash",
        "description": "Execute a bash command in the shell and return its output. Command runs with the permissions of the calling process.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
            },
            "required": ["command"]
        }
    },
    {
        "name": "run_python",
        "description": "Run a given python script with specific arguments",
        "input_schema": {
            "type": "object",
            "properties": {
                "absolute_path": {
                    "type": "string",
                    "description": "The absolute path to the python program to execute"
                },
                "arguments": {
                    "type": "array",
                    "description": "The arguments to the python script",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["command"]
        }
    },
    {
        "name": "finish_task",
        "description": "Indicate that the task is complete, and we should proceed to the next task. This may trigger tests to run that verify the task was completed correctly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Completion text that you would like to pass on to the next step with any important information, including the results of running this current step. Be concise but complete."
                },
                "FORCE_SUCCESS": {
                    "type": "boolean",
                    "description": "Force move on to the next task. Use ONLY if finish_task previously failed but you are sure you know better and want to move on to the next step even though the current tests fail."
                },
            },
            "required": ["description"]
        }
    }
]

cost = []
def query_llm(history):
    #print("QUERY MODEL", history)
    """
    hkey = hh(history)
    if hkey in qhist:
        print("Cache hit")
        history.append({'role': 'assistant', 'content': qhist[hkey]})
        logfile.append(copy.deepcopy(history))
        pickle.dump(logfile, open("/tmp/logfile.p", "wb"))
        return qhist[hkey]
    """

    send = copy.deepcopy(history)
    for n in range(len(send)):
        if n == len(send)-4 or n == len(send)-3 or n == len(send)-2 or n == len(send)-1:
            send[n]['content'][0]['cache_control'] = {"type": "ephemeral"}
        else:
            if 'cache_control' in send[n]['content'][0]:
                del send[n]['content'][0]['cache_control']


    print("SENDING", send)
    for _ in range(8):
        try:
            attack_llm="claude-3-7-sonnet-latest"
            response = anthropic.Anthropic(api_key=open(".CLAUDE").read().strip()).messages.create(
                model=attack_llm,
                max_tokens=4096,
                messages=send,
                tool_choice={"type": "auto"},
                tools=tools
            )
            time.sleep(30)
            break
        except:
            raise
    print(response)
    #exit(0)

    cost.append(response.usage.input_tokens*3 + response.usage.output_tokens*15 + response.usage.cache_read_input_tokens*.3 + response.usage.cache_creation_input_tokens*3.75)
    print('cost sum', cost[-1]/1e6, 'sum', np.sum(cost)/1e6)
    
    out = response.content

    oout = []
    for x in out:
        if x.type == 'tool_use':
            oout.append({'id': x.id,
                         'name': x.name,
                         'input': x.input,
                         'type': x.type})
        elif x.type == 'text':
            oout.append({'text': x.text,
                         'type': x.type})
        else:
            print(x)
            raise
                         
    print("OUT", oout)

    #qhist[hkey] = out
    #pickle.dump(qhist, open("/tmp/hist.p","wb"))
    history.append({'role': 'assistant', 'content': oout})
    #print("Logfile len", len(logfile))
    #logfile.append(copy.deepcopy(history))
    #pickle.dump(logfile, open("/tmp/logfile.p", "wb"))

    return out
