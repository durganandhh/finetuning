# Copyright 2021  Xiaomi Corporation (Author: Junbo Zhang, Yongqing Wang)
# Apache 2.0

# This script trains a simple polynomial regression model to convert GOP into
# human expert scores.


import sys
import argparse
import pickle
import kaldi_io
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import difflib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from utils import load_phone_symbol_table, load_human_scores, balanced_sampling

def align_sequences(gop_seq, human_seq):
    matcher = difflib.SequenceMatcher(None, gop_seq, human_seq)
    alignment = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal' or tag == 'replace':
            for i, j in zip(range(i1, i2), range(j1, j2)):
                alignment.append((i, j))
        elif tag == 'delete':
            for i in range(i1, i2):
                alignment.append((i, None))
        elif tag == 'insert':
            for j in range(j1, j2):
                alignment.append((None, j))
    return alignment

def get_args():
    parser = argparse.ArgumentParser(
        description='Train a simple polynomial regression model to convert '
                    'gop into human expert score',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--phone-symbol-table', type=str, default='',
                        help='Phone symbol table, used for detect unmatch '
                             'feature and labels.')
    parser.add_argument('--nj', type=int, default=1, help='Job number')
    parser.add_argument('gop_scp', help='Input gop file, in Kaldi scp')
    parser.add_argument('human_scoring_json',
                        help='Input human scores file, in JSON format')
    parser.add_argument('model', help='Output the model file')
    sys.stderr.write(' '.join(sys.argv) + "\n")
    args = parser.parse_args()
    return args


def train_model_for_phone(label_feat_pairs):
    model = LinearRegression()
    labels, gops = list(zip(*label_feat_pairs))
    labels = np.array(labels).reshape(-1, 1)
    gops = np.array(gops).reshape(-1, 1)
    gops = PolynomialFeatures(2).fit_transform(gops)
    gops, labels = balanced_sampling(gops, labels)
    model.fit(gops, labels)
    return model.coef_, model.intercept_


def main():
    args = get_args()

    # Phone symbol table
    _, phone_int2sym = load_phone_symbol_table(args.phone_symbol_table)

    # Human expert scores
    score_of, phone_of = load_human_scores(args.human_scoring_json, floor=1)

    # Prepare training data
    train_data_of = {}
    # for key, gops in kaldi_io.read_post_scp(args.gop_scp):
    #     for i, [(ph, gop)] in enumerate(gops):
    #         ph_key = f'{key}.{i}'
    #         if ph_key not in score_of:
    #             print(f'Warning: no human score for {ph_key}')
    #             continue
    #         if phone_int2sym is not None and phone_int2sym[ph] != phone_of[ph_key]:
    #             print(f'Unmatch: {phone_int2sym[ph]} <--> {phone_of[ph_key]} ')
    #             continue
    #         score = score_of[ph_key]
    #         train_data_of.setdefault(ph, []).append((score, gop))
    from custom_gop_parser import parse_gop_txt
    # gop_data = parse_gop_txt(args.gop_scp)
    gop_data = parse_gop_txt(args.gop_scp, phone_int2sym)
    for key, gops in gop_data.items():
        gop_labels = [phone_int2sym[ph].split('_')[0] for ph, _ in gops]
        human_labels = [phone_of[f'{key}.{i}'] for i in range(len(gops)) if f'{key}.{i}' in phone_of]

        alignment = align_sequences(gop_labels, human_labels)
        print(f"\nUtterance: {key}")
        print("GOP phones:   ", gop_labels)
        print("Human phones: ", human_labels)
        print("Alignment:")
        for gop_idx, human_idx in alignment:
            g = gop_labels[gop_idx] if gop_idx is not None else "-"
            h = human_labels[human_idx] if human_idx is not None else "-"
            print(f"  {g:<5} <--> {h}")
        
        for gop_idx, human_idx in alignment:
            if gop_idx is None or human_idx is None:
                continue

            ph, gop_score = gops[gop_idx]
            ph_key = f'{key}.{human_idx}'

            if ph_key not in score_of:
                print(f'Warning: no human score for {ph_key}')
                continue

            gop_phone = phone_int2sym[ph].split('_')[0]
            human_phone = phone_of[ph_key]

            if gop_phone != human_phone:
                print(f'Unmatch: {gop_phone} <--> {human_phone}')
                continue

            score = score_of[ph_key]
            train_data_of.setdefault(ph, []).append((score, gop_score))


    # Train polynomial regression
    with ProcessPoolExecutor(args.nj) as ex:
        future_to_model = [(ph, ex.submit(train_model_for_phone, pairs))
                           for ph, pairs in train_data_of.items()]
        model_of = {ph: future.result() for ph, future in future_to_model}

    # Write to file
    with open(args.model, 'wb') as f:
        pickle.dump(model_of, f)


if __name__ == "__main__":
    main()
