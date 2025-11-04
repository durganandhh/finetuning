# Copyright 2021  Xiaomi Corporation (Author: Junbo Zhang)
# Apache 2.0

# This script trains SVR models to convert GOP-based features into
# human expert scores, one model per phone.

# MSE: 0.16
# Corr: 0.45
#
#               precision    recall  f1-score   support
#
#            0       0.42      0.30      0.35      1339
#            1       0.16      0.36      0.22      1828
#            2       0.97      0.92      0.94     44079
#
#     accuracy                           0.88     47246
#    macro avg       0.52      0.53      0.50     47246
# weighted avg       0.92      0.88      0.90     47246

import sys
import argparse
import pickle
import kaldi_io
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.svm import SVR

# Import helper functions from local utils.py
sys.path.append('/home/mcw/durga/kaldi/egs/gop_speechocean762/s5/local')
from utils import (
    load_phone_symbol_table,
    load_human_scores,
    add_more_negative_data
)


def get_args():
    parser = argparse.ArgumentParser(
        description='Train SVR models to convert GOP-based features to human expert scores',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--phone-symbol-table', type=str, default='',
                        help='Phone symbol table (used for symbol ↔ int mapping)')
    parser.add_argument('--nj', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('feature_scp',
                        help='Input GOP-based feature file (Kaldi .scp format)')
    parser.add_argument('human_scoring_json',
                        help='Input human scores file (JSON format)')
    parser.add_argument('model',
                        help='Output model file (pickle)')
    sys.stderr.write(' '.join(sys.argv) + "\n")
    return parser.parse_args()


def train_model_for_phone(label_feat_pairs):
    """Train an SVR model for one phone."""
    model = SVR()
    labels, feats = list(zip(*label_feat_pairs))
    labels = np.array(labels).ravel()
    feats = np.array(feats)
    model.fit(feats, labels)
    return model


def main():
    args = get_args()

    # Load phone symbol table
    _, phone_int2sym = load_phone_symbol_table(args.phone_symbol_table)
    sym_to_int = {v: k for k, v in phone_int2sym.items()}
    print(f"Loaded {len(sym_to_int)} phone symbols from {args.phone_symbol_table}")

    # Load human expert scores
    score_of, phone_of = load_human_scores(args.human_scoring_json, floor=1)
    print(f"Loaded {len(score_of)} human score entries.")

    # Prepare training data
    train_data_of = {}

    for ph_key, feat in kaldi_io.read_vec_flt_scp(args.feature_scp):
        if ph_key not in score_of:
            print(f'Warning: no human score for {ph_key}')
            continue

        # Get expected phone symbol for this key
        expected_sym = phone_of.get(ph_key, None)
        if expected_sym is None:
            print(f'Warning: no phone symbol for {ph_key}')
            continue

        # Convert phone symbol to integer ID
        ph = sym_to_int.get(expected_sym, None)
        if ph is None:
            print(f'Warning: phone symbol "{expected_sym}" not found in symbol table')
            continue

        # Debug print (optional)
        print(f"DEBUG: ph_key={ph_key}, phone symbol={expected_sym}, phone ID={ph}")

        # Human score and feature vector
        # score = score_of[ph_key]
        # train_data_of.setdefault(ph, []).append((score, feat))
        score = score_of[ph_key]
        train_data_of.setdefault(ph, []).append((score, feat[1:]))  # remove first element


    # Balance the dataset (e.g., add more negative samples)
    train_data_of = add_more_negative_data(train_data_of)

    # Train per-phone SVR models in parallel
    model_of = {}
    with ProcessPoolExecutor(args.nj) as ex:
        futures = {ph: ex.submit(train_model_for_phone, pairs)
                   for ph, pairs in train_data_of.items()}
        for ph, future in futures.items():
            model_of[ph] = future.result()
            print(f"Trained SVR model for phone ID {ph} ({len(train_data_of[ph])} samples)")

    # Save models
    with open(args.model, 'wb') as f:
        pickle.dump(model_of, f)

    print(f"All models saved to {args.model}")


if __name__ == "__main__":
    main()
