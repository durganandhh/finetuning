import json

# Load human phoneme scores
with open("/home/mcw/durga/kaldi/egs/gop_speechocean762/s5/scores.json", "r") as f:
    human_scores = json.load(f)

# Load predicted phoneme scores
predicted_scores = {}
with open("/home/mcw/durga/kaldi/egs/gop_speechocean762/s5/exp/gop_test/predicted_scores.txt", "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            full_id, score, phn_id = parts
            utt_id, phn_index = full_id.rsplit(".", 1)
            predicted_scores[(utt_id, int(phn_index))] = {
                "predicted_score": float(score),
                "phn_id": int(phn_id)
            }

# Load phoneme ID-to-symbol mapping
phnid_to_symbol = {}
with open("/home/mcw/durga/kaldi/egs/gop_speechocean762/s5/exp/gop_test/phones-pure.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            symbol, phnid = parts
            phnid_to_symbol[int(phnid)] = symbol

# Compare phoneme-level scores
phoneme_comparisons = []

for utt_id, human_data in human_scores.items():
    phn_counter = 1
    for word in human_data.get("words", []):
        for phone, phone_score in zip(word["phones"], word["phones-accuracy"]):
            key = (utt_id, phn_counter)
            if key in predicted_scores:
                predicted_info = predicted_scores[key]
                predicted_symbol = phnid_to_symbol.get(predicted_info["phn_id"], "UNK")
                phoneme_comparisons.append({
                    "utterance_id": utt_id,
                    "phoneme_index": phn_counter,
                    "word": word["text"],
                    "human_phoneme": phone,
                    "predicted_phoneme": predicted_symbol,
                    "human_score": phone_score,
                    "predicted_score": predicted_info["predicted_score"]
                })
            phn_counter += 1

# Output results
for entry in phoneme_comparisons:
    print(f"{entry['utterance_id']} | {entry['phoneme_index']} | {entry['word']} | Human={entry['human_phoneme']} | Predicted={entry['predicted_phoneme']} | HumanScore={entry['human_score']} | PredictedScore={entry['predicted_score']}")
