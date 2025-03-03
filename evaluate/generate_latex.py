import os
import numpy as np

# LaTeX table header
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\begin{tabular}{l|" + "c"*6 + "c}")
print(r"\toprule")

# Column headers
models = ["No Attack", "GPT-4o", "o1", "Sonnet 3.5 + o1", "Sonnet 3.5", "Sonnet 3.5 40"]
paths = ["tmp_claude_full_supervisor", "tmp-gpt-4o-full", "tmp_o1_full", 
         "tmp_claude_full_supervisor/", "tmp_claude_full/", "tmp_claude_full_40/"]

print("Defense & " + " & ".join(models) + " & Broken" + r" \\")
print(r"\midrule")

# Store all results in a dictionary of dictionaries
all_results = {}
for model, path in zip(models, paths):
    results = {}
    for out in os.listdir(path):
        data = open(os.path.join(path,out)).read()
        if 'Final robust accuracy' in data and model != "No Attack":
            try:
                acc = float(data.split('Final robust accuracy')[1].split()[0])
            except:
                acc = 1.0
            try:
                rej = data.split("\nGet [")[1].split("]")[0].split()
                rej = sum(x == '-1' for x in rej)
            except:
                rej = 0
            results[out] = acc + rej/1000
        else:
            name, _, idx = out.rpartition("-")
            try:
                clean = open("log/"+name+".txt").read()
                clean = float(clean.split("mean acc")[1+int(idx)].split()[0])
                results[out] = clean
            except:
                pass
    all_results[model] = results

# Get all unique defense names and their worst attack accuracy
defense_stats = {}
for defense in set().union(*[set(results.keys()) for results in all_results.values()]):
    attack_accs = []
    clean_acc = all_results["No Attack"].get(defense, 0)
    for model in models[1:]:  # Skip "No Attack"
        if defense in all_results[model]:
            attack_accs.append(all_results[model][defense])
    if attack_accs:
        worst_acc = min(attack_accs)
        defense_stats[defense] = {
            'worst_acc': worst_acc,
            'broken': any(acc < clean_acc/2 for acc in attack_accs)
        }

# Sort defenses by worst accuracy
sorted_defenses = sorted(defense_stats.keys(), 
                        key=lambda x: defense_stats[x]['worst_acc'])

# Print each row
for defense in sorted_defenses:
    row = [defense]
    values = []
    best_attack_acc = float('inf')
    
    # First get the non-attack accuracy
    clean_acc = all_results["No Attack"].get(defense, "")
    if clean_acc:
        values.append(f"{clean_acc:.3f}")
    else:
        values.append("-")
    
    # Then get attack accuracies
    attack_accs = []
    for model in models[1:]:  # Skip "No Attack"
        val = all_results[model].get(defense, "")
        if val:
            attack_accs.append(val)
            best_attack_acc = min(best_attack_acc, val)
        else:
            attack_accs.append(None)
    
    # Add values with bold for best attacks
    for acc in attack_accs:
        if acc is None:
            values.append("-")
        elif acc == best_attack_acc:
            values.append(f"\\textbf{{{acc:.3f}}}")
        else:
            values.append(f"{acc:.3f}")
    
    # Add checkmark if defense is broken
    if defense_stats[defense]['broken']:
        values.append(r"\checkmark")
    else:
        values.append("")
    
    print(" & ".join([defense] + values) + r" \\")

# LaTeX table footer
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\caption{Accuracy of different models against various defenses, sorted by worst-case performance. Bold indicates best attack(s) for each defense. Checkmark indicates at least one attack achieves accuracy below half of clean accuracy.}")
print(r"\label{tab:defense-accuracy}")
print(r"\end{table}")
