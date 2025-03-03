import matplotlib.pyplot as plt
import numpy as np

import re

def parse_defense_log(log_content):
    # Regular expression to match the path, idx, clean, and robust values
    pattern = r"path ([^\s]+) idx (\d+) clean: ([\d.]+) robust: ([\d.]+)"
    
    # Dictionary to store results
    results = {}
    
    # Find all matches in the log content
    matches = re.finditer(pattern, log_content)
    
    # Process each match
    for match in matches:
        path = match.group(1)
        idx = int(match.group(2))
        clean = float(match.group(3))
        robust = float(match.group(4))

        if robust > clean: robust = clean
        # Store in dictionary with tuple key
        results[(path, idx)] = (clean, robust)
    
    return results

import os
import re

def parse_clean_accuracy_log(filename):
    """Parse a single clean accuracy log file."""
    results = {}
    defense_name = os.path.basename(filename).replace('.log', '')
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Pattern to match "Verifying forward_X.py" followed by "mean acc Y"
    pattern = r"Verifying .*\s+mean acc ([\d.]+)"
    matches = re.finditer(pattern, content)

    idx = 0
    for match in matches:
        acc = float(match.group(1))
        key = (f"../defenses/{defense_name}", idx)
        idx += 1
        results[key] = (acc, acc)
    
    return results

def process_clean_logs(log_dir):
    """Process all clean accuracy log files in a directory."""
    all_results = {}
    
    for filename in os.listdir(log_dir):
        if not filename.endswith('.log'):
            continue
        
        filepath = os.path.join(log_dir, filename)
        try:
            results = parse_clean_accuracy_log(filepath)
            all_results.update(results)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    return all_results

        
log_dir = "log"  # Directory containing log files
clean = process_clean_logs("log")
clean_keys = clean.keys()

clean[('../defenses/robust-ecoc', 0)] = (0.89, 0.89)
clean[('../defenses/ISEAT', 0)] = (0.904, 0.904)
clean[('../defenses/trapdoor', 0)] = (0.377, 0.377)
clean[('../defenses/Mixup-Inference', 0)] = (0.934, 0.934)
clean[('../defenses/Combating-Adversaries-with-Anti-Adversaries', 0)] = (0.849, 0.849)
clean[('../defenses/MART', 0)] = (0.876, 0.876)
clean[('../defenses/MART', 0)] = (0.876, 0.876)
clean[('../defenses/MagNet.pytorch', 0)] = (0.711, 0.711)
clean[('../defenses/disco', 0)] = (0.089, 0.089)
clean[('../defenses/ISEAT', 0)] = (0.904, 0.904)
clean[('../defenses/Mixup-Inference', 1)] = (0.886, 0.886)
clean[('../defenses/MagNet.pytorch', 0)] = (0.711, 0.711)
clean[('../defenses/ISEAT', 0)] = (0.904, 0.904)
clean[('../defenses/Mixup-Inference', 0)] = (0.934, 0.934)
clean[('../defenses/Mixup-Inference', 2)] = (0.794, 0.794)
clean[('../defenses/Mixup-Inference', 1)] = (0.897, 0.897)
clean[('../defenses/MART', 0)] = (0.876, 0.876)
clean[('../defenses/MagNet.pytorch', 0)] = (0.711, 0.711)
clean[('../defenses/ISEAT', 0)] = (0.904, 0.904)
clean[('../defenses/TurningWeaknessIntoStrength', 0)] = (0.491, 0.491)
clean[('../defenses/Mixup-Inference', 0)] = (0.934, 0.934)
clean[('../defenses/Combating-Adversaries-with-Anti-Adversaries', 0)] = (0.849, 0.849)
clean[('../defenses/MART', 0)] = (0.876, 0.876)
clean[('../defenses/MagNet.pytorch', 0)] = (0.711, 0.711)
clean[('../defenses/ISEAT', 0)] = (0.904, 0.904)
clean[('../defenses/trapdoor', 0)] = (0.377, 0.377)
clean[('../defenses/Mixup-Inference', 0)] = (0.934, 0.934)
clean[('../defenses/Combating-Adversaries-with-Anti-Adversaries', 0)] = (0.849, 0.849)
clean[('../defenses/Mixup-Inference', 2)] = (0.8, 0.8)
clean[('../defenses/Mixup-Inference', 1)] = (0.9, 0.9)
clean[('../defenses/MART', 0)] = (0.876, 0.876)
clean[('../defenses/MagNet.pytorch', 0)] = (0.711, 0.711)

                                                                                    
# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

# Function to process and plot data
def plot_data(ax, filter_selfstudy, title):
    for name, fp in [("GPT-4o", "log_evaluate/attack_log_4o"),
                     ("o3-mini", "log_evaluate/attack_log_o3"),
                     ("o1", "log_evaluate/attack_log_o1"),
                     ("Haiku 3.5", "log_evaluate/attack_log_haiku"),
                     ("Sonnet 3.5 (+o3)", "log_evaluate/attack_log_sonnet_o3_supervisor"),
                     ("Sonnet 3.5", "log_evaluate/attack_log_sonnet_30"),
                     ("Sonnet 3.5 (40)", "log_evaluate/attack_log_sonnet_40")][::-1]:
        results = parse_defense_log(open(fp).read())
        copy = dict(clean)
        copy.update(results)
        for x in set(copy.keys()) - set(clean_keys):
            print(f"clean[{x}] = {(copy[x][0], copy[x][0])}")
        
        if filter_selfstudy:
            copy = {k:v for (k,v) in copy.items() if 'selfstudy' in k[0]}
        else:
            copy = {k:v for (k,v) in copy.items() if 'selfstudy' not in k[0]}
        
        ax.plot(sorted([x[1] for x in sorted(copy.values())])[::-1], label=name)
        #print(name, sum((b<a/2) for a,b in copy.values()))
        attack = sum((b<a/2) for a,b in copy.values())
        acc = np.mean([x[1] for x in copy.values()])
        print(f"{name} attacks {100*attack/51:.0f}% of defenses with reducing the average robust accuracy to {100*acc:.1f}%.")

    ax.set_ylim(1.02, -0.02)
    ax.set_xlabel("Defenses (sorted order, CDF-like)")
    ax.set_ylabel("Robust Accuracy")
    ax.set_title(title)

# Plot both versions
plot_data(ax1, filter_selfstudy=True, title="CTF-Like Defenses")
plot_data(ax2, filter_selfstudy=False, title="Real-World Defenses")

# Create a single legend at the top of the figure
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), 
          ncol=len(labels), frameon=False)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # Make room for the legend at the top

# Save the figure
plt.savefig("acc.pdf", bbox_inches='tight')
