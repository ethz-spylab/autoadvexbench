import json
import sys

fp = sys.argv[1]


r = []
if 'Task Log' in open(fp).read():
    for line in open(fp):
        if "Task Log" in line:
            r.append(json.dumps(eval(line[9:])))
else:
    d = open(fp).read()
    r.append(json.dumps(eval(d.split("SENDING")[-1].split("\n")[0])))
open("view/data.js","w").write("DATA = ["+",".join(r)+"]")
