import json
import numpy as np
from sklearn.metrics import roc_curve
import csv

# -------- Config --------
GOP_FILE = "exp/gop_test/gop.1.txt"
SCORE_JSON = "scores.json"
OUTPUT_CSV = "eer_results.csv"
MIN_PHONES = 3  # minimum phones required for EER computation

# -------- 1. Load human scores --------
with open(SCORE_JSON, "r") as f:
    human_scores = json.load(f)

def get_phone_labels(score_entry):
    """Convert human phone accuracy to binary labels: 0=correct, 1=mispronounced"""
    labels = []
    for word in score_entry['words']:
        for acc in word['phones-accuracy']:
            labels.append(0 if acc == 2 else 1)
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

# -------- 3. Open CSV to write results --------
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Utterance_ID", "Num_Phones", "EER(%)", "EER_Threshold_GOP"])

    all_eers = []

    # -------- 4. Compute EER per utterance safely --------
    for utt_id, gop_vals in utt_gop.items():
        if utt_id not in gt_dict:
            continue

        gt_labels = np.array(gt_dict[utt_id])
        gop_vals = np.array(gop_vals)
        min_len = min(len(gt_labels), len(gop_vals))

        if min_len < MIN_PHONES:
            print(f"Skipping {utt_id}: too few phones ({min_len})")
            continue

        y_true = gt_labels[:min_len]
        y_score = gop_vals[:min_len]

        # -------- Try-except for utterances with only one class --------
        try:
            fpr, tpr, thresholds = roc_curve(y_true, -y_score)
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.abs(fnr - fpr))
            eer = fpr[eer_idx]
            eer_threshold_gop = -thresholds[eer_idx]
        except ValueError:
            eer = None
            eer_threshold_gop = None
            print(f"Cannot compute EER for {utt_id}: all one class")

        writer.writerow([utt_id, min_len, round(eer*100,2) if eer is not None else "NA",
                         round(eer_threshold_gop,3) if eer_threshold_gop is not None else "NA"])
        if eer is not None:
            all_eers.append(eer)

# -------- 5. Summary --------
if all_eers:
    print(f"\nSaved EER results for {len(all_eers)} utterances → {OUTPUT_CSV}")
    print(f"Average EER across utterances: {np.mean(all_eers)*100:.2f}%")
else:
    print("No valid utterances found for EER computation.")
