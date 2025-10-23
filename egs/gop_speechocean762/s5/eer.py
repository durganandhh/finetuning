import json
import numpy as np
from sklearn.metrics import roc_curve


GOP_FILE = "exp/gop_test/gop.1.txt" 
SCORE_JSON = "scores.json"            
THRESHOLD = -5                       


with open(SCORE_JSON, "r") as f:
    human_scores = json.load(f)

# Function to get per-phone ground truth from score.json
def get_phone_labels(score_entry):
    labels = []
    for word in score_entry['words']:
        for acc in word['phones-accuracy']:
            label = 0 if acc == 2 else 1  # 0=correct, 1=mispronounced
            labels.append(label)
    return labels

# Build dictionary: utt_id -> list of phone labels
gt_dict = {utt_id: get_phone_labels(data) for utt_id, data in human_scores.items()}

# -------- 2. Load GOP scores --------
utt_gop = {}
with open(GOP_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        utt_id = parts[0]
        gop_vals = []

        # get all [phone_idx value] pairs
        pairs = parts[1].split("] [")
        for p in pairs:
            p = p.replace("[","").replace("]","").strip()
            if not p:
                continue
            phone_idx, val = p.split()
            gop_vals.append(float(val))
        utt_gop[utt_id] = gop_vals

# -------- 3. Prepare lists for EER --------
y_true = []
y_score = []

for utt_id, gop_vals in utt_gop.items():
    if utt_id not in gt_dict:
        continue
    gt_labels = gt_dict[utt_id]
    min_len = min(len(gt_labels), len(gop_vals))
    y_true.extend(gt_labels[:min_len])
    y_score.extend(gop_vals[:min_len])

y_true = np.array(y_true)
y_score = np.array(y_score)

# -------- 4. Compute EER --------
# Use -GOP: lower GOP = more likely mispronounced
fpr, tpr, thresholds = roc_curve(y_true, -y_score)
fnr = 1 - tpr
eer_idx = np.nanargmin(np.abs(fnr - fpr))
eer = fpr[eer_idx]

# Compute the corresponding GOP threshold
eer_threshold_gop = -thresholds[eer_idx]  # negate because we used -y_score

print(f"Number of phones evaluated: {len(y_true)}")
print(f"EER: {eer*100:.2f}%")
print(f"EER threshold (GOP value): {eer_threshold_gop:.3f}")  # <--- print it here

# -------- 5. Sanity check: compare GOP vs human per utterance --------
# for utt_id, gop_vals in utt_gop.items():
#     if utt_id not in gt_dict:
#         continue
#     gt_labels = gt_dict[utt_id]
#     min_len = min(len(gt_labels), len(gop_vals))
    
#     print(f"\nUtterance: {utt_id}")
#     print("Phone\tHuman\tGOP")
#     for i in range(min_len):
#         human_label = gt_labels[i]
#         gop_val = gop_vals[i]
#         print(f"{i+1}\t{human_label}\t{gop_val:.3f}")




