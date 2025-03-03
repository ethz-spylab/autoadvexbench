import json
tools = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write data to a file at a particular location. Overwrites the file if it already exists, or creates a new file if it does not.",
            "parameters": {
                "type": "object",
                "properties": {
                    "absolute_path": {
                        "type": "string",
                        "description": "Absolute path to the file you would like to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file."
                    }
                },
                "required": ["absolute_path", "content"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read data from a file at a particular location. Returns an error if the file does not exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "absolute_path": {
                        "type": "string",
                        "description": "Absolute path to the file you would like to read"
                    }
                },
                "required": ["absolute_path"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Execute a bash command in the shell and return its output. Command runs with the permissions of the calling process.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    }
                },
                "required": ["command"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run a given python script with specific arguments",
            "parameters": {
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
                "required": ["absolute_path"],
                "additionalProperties": False
            },
            "strict": False
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish_task",
            "description": "Indicate that the task is complete, and we should proceed to the next task. This may trigger tests to run that verify the task was completed correctly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Completion text that you would like to pass on to the next step with any important information, including the results of running this current step. Be concise but complete."
                    },
                    "FORCE_SUCCESS": {
                        "type": "boolean",
                        "description": "Force move on to the next task. Use ONLY if finish_task previously failed but you are sure you know better and want to move on to the next step even though the current tests fail."
                    }
                },
                "required": ["description"],
                "additionalProperties": False
            },
            "strict": False
        }
    }
]


def query_llm(history, retry=True):
    """
    hkey = hh(history)
    if hkey in qhist:
        print("Cache hit")
        history.append({'role': 'assistant', 'content': qhist[hkey]})
        return qhist[hkey]
    """
    import openai
    api_key = open(".OPENAI").read().strip()
    oaiclient = openai.OpenAI(api_key=api_key)

    print("SENDING", history)


    mhist = []
    for x in history:
        if x['role'] == 'user':
            if x['content'][0]['type'] == 'tool_result':
                mhist.append({'role': 'tool',
                              'tool_call_id': x['content'][0]['tool_use_id'],
                              'content': x['content'][0]['content']})
            else:
                mhist.append(x)
        elif x['role'] == 'assistant':
            if len(x['content']) == 2:
                mhist.append({'role': 'assistant',
                              'content': x['content'][0]['text'],
                              'tool_calls': [{'id': x['content'][1]['id'],
                                              'function': {
                                                  'arguments': json.dumps(x['content'][1]['input']),
                                                  'name': x['content'][1]['name']
                                              },
                                              'type': 'function'}]})
            else:
                mhist.append({'role': 'assistant',
                              'content': x['content'][0]['text']
                              })

    print("MOD", mhist)
    attack_llm = "o3-mini"
    print("Asking", attack_llm)
    out = oaiclient.chat.completions.create(
        messages=mhist,
        tools=tools,
#        max_tokens=4096,
        model=attack_llm)
    out = out.choices[0].message
    """
    qhist[hkey] = out
    pickle.dump(qhist, open("/tmp/ohist.p","wb"))
    """


    print(out)

    if out.tool_calls is not None and len(out.tool_calls) > 0:
        tool_call = out.tool_calls[0]

        try:
            json_out = json.loads(tool_call.function.arguments)
        except:
            print("JSON Failure")
            if retry is False:
                raise
            else:
                return query_llm(history, retry=False)

        d = [{"type": "text",
              "text": out.content},
             {'id': tool_call.id,
              'input': json_out,
              'type': 'tool_use',
              'name': tool_call.function.name}]
    else:
        d = [{"type": "text",
              "text": out.content}]
        

    history.append({'role': 'assistant', 'content': d})
        
    return out


