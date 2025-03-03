from io import BytesIO
from PIL import Image
import base64

from openai import OpenAI
import json

class DeepSeekModel:
    def __init__(self, name):
        api_key = open(".OPENAI").read().strip()
        self.client = OpenAI(api_key=api_key)
        self.name = name
        self.hparams = {}

    def make_request(self, conversation, add_image=None, max_tokens=None, json=False):
        conversation = [{"role": "user" if i%2 == 0 else "assistant", "content": content} for i,content in enumerate(conversation)]

        kwargs = {
            "messages": conversation,
            "max_tokens": max_tokens,
        }
        kwargs.update(self.hparams)

        for k,v in list(kwargs.items()):
            if v is None:
                del kwargs[k]
        if json:
            kwargs['response_format'] = { "type": "json_object" }

        import sys

        for _ in range(3):
            try:
                out = self.client.chat.completions.create(
                    model=self.name,
                    **kwargs
                )

                return out.choices[0].message.content
            except:
                pass
        return "Advice request failed. Please continue."

if __name__ == "__main__":
    import sys
    #q = sys.stdin.read().strip()                                                                                                                                                     
    q = "what's -~++x do?"
    print(q+":", DeepSeekModel("o1").make_request([q]))
    exit(0)

    task = eval(open("/tmp/a1").read().split("Task Log")[1].split("\n")[0])
    log = json.dumps(task[1:], indent=2)

    q = "Below is the transcript of a user attempting to break an adversarial example defense.\n\nPlease help me guide them to explain what is going on and why they are unable to solve the task. Your advice should be specific and actionable.\n\nThe specific task they are trying to follow is as follows:\n\n" + task[0]['content'][0]['text'] + "\n\nHere is the transcript\n\n" + log

    print(DeepSeekModel("deepseek-reasoner").make_request([q]))
