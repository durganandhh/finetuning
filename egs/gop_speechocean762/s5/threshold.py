import json
import numpy as np

# -------- Config --------
GOP_FILE = "exp/gop_test/gop.1.txt"  # your GOP output
SCORE_JSON = "scores.json"            # your human scores
THRESHOLD = -5                        # fixed GOP threshold for mispronunciation

# -------- 1. Load human scores --------
with open(SCORE_JSON, "r") as f:
    human_scores = json.load(f)

def get_phone_labels(score_entry):
    labels = []
    for word in score_entry['words']:
        for acc in word['phones-accuracy']:
            label = 0 if acc == 2 else 1  # 0=correct, 1=mispronounced
            labels.append(label)
    return labels

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

        pairs = parts[1].split("] [")
        for p in pairs:
            p = p.replace("[","").replace("]","").strip()
            if not p:
                continue
            phone_idx, val = p.split()
            gop_vals.append(float(val))
        utt_gop[utt_id] = gop_vals

# -------- 3. Prepare lists --------
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

# -------- 4. Make predictions using fixed threshold --------
y_pred_fixed = np.array([1 if val < THRESHOLD else 0 for val in y_score])

# -------- 5. Compute error rate --------
error_fixed = 1 - np.mean(y_pred_fixed == y_true)

print(f"Number of phones evaluated: {len(y_true)}")
print(f"Error rate using fixed threshold ({THRESHOLD}): {error_fixed*100:.2f}%")

# -------- 6. Optional: per-utterance sanity check --------
# for utt_id, gop_vals in utt_gop.items():
#     if utt_id not in gt_dict:
#         continue
#     gt_labels = gt_dict[utt_id]
#     min_len = min(len(gt_labels), len(gop_vals))
    
#     print(f"\nUtterance: {utt_id}")
#     print("Phone\tHuman\tGOP\tPred(Fixed)")
#     for i in range(min_len):
#         human_label = gt_labels[i]
#         gop_val = gop_vals[i]
#         pred_fixed = 1 if gop_val < THRESHOLD else 0
#         print(f"{i+1}\t{human_label}\t{gop_val:.3f}\t{pred_fixed}")
