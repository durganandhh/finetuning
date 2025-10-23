#!/usr/bin/env bash
# --------------------------------------------------------
# Fine-tune Librispeech TDNN Chain model on Speechocean762
# Author: You :)
# --------------------------------------------------------

set -e
. ./cmd.sh
. ./path.sh
. utils/parse_options.sh || exit 1

# -------------------------------
# Configuration
# -------------------------------
stage=0
nj=10
train_set=speechocean762_train
dev_set=speechocean762_dev

# Paths (update if your structure differs)
lang=/home/mcw/durga/kaldi/libreespeech2/0013_librispeech_v1_lm/data/lang_test_tgsmall
pretrained_model=/home/mcw/durga/kaldi/libreespeech2/0013_librispeech_v1_chain/exp/chain_cleaned/tdnn_1d_sp
ivector_extractor=/home/mcw/durga/kaldi/libreespeech2/0013_librispeech_v1_extractor/exp/nnet3_cleaned/extractor
adapt_dir=exp/chain_finetune/tdnn_1d_sp_adapt
mfcc_conf=conf/mfcc_hires.conf

# -------------------------------
# Stage 0: Feature Extraction
# -------------------------------
if [ $stage -le 0 ]; then
  echo "=== Stage 0: Extracting MFCCs and i-vectors ==="
  for x in $train_set $dev_set; do
    steps/make_mfcc.sh --nj $nj --mfcc-config $mfcc_conf \
      data/$x exp/make_hires/$x mfcc
    steps/compute_cmvn_stats.sh data/$x exp/make_hires/$x mfcc
    utils/fix_data_dir.sh data/$x
  done

  echo "Extracting i-vectors..."
  for x in $train_set $dev_set; do
    steps/online/nnet2/extract_ivectors_online.sh --nj $nj \
      data/$x $ivector_extractor exp/nnet3_cleaned/ivectors_$x
  done
fi

# -------------------------------
# Stage 1: Align with pretrained model
# -------------------------------
if [ $stage -le 1 ]; then
  echo "=== Stage 1: Aligning training data with pretrained TDNN model ==="
  steps/nnet3/align.sh --cmd "run.pl" --nj $nj --use-gpu true \
    --online-ivector-dir exp/nnet3_cleaned/ivectors_$train_set \
    data/$train_set $lang $pretrained_model exp/align_speechocean762
fi

# -------------------------------
# Stage 1.5: Generate lattices (CHAIN)
# -------------------------------
if [ $stage -le 1.5 ]; then
  echo "=== Stage 1.5: Generating lattices for chain fine-tuning ==="
  steps/nnet3/align_lats.sh --cmd "run.pl" --nj $nj --use-gpu true \
    --online-ivector-dir exp/nnet3_cleaned/ivectors_$train_set \
    data/$train_set $lang $pretrained_model exp/align_lats_speechocean762
  echo "Lattices generated successfully in exp/align_lats_speechocean762"
fi

# -------------------------------
# Stage 2: Fine-tuning TDNN
# -------------------------------
if [ $stage -le 2 ]; then
  echo "=== Stage 2: Fine-tuning chain TDNN model ==="
  steps/nnet3/chain/train.py \
    --stage 0 \
    --feat-dir data/$train_set \
    --feat.online-ivector-dir exp/nnet3_cleaned/ivectors_$train_set \
    --lat-dir exp/align_lats_speechocean762 \
    --tree-dir $pretrained_model/tree \
    --trainer.input-model $pretrained_model/final.mdl \
    --dir $adapt_dir \
    --trainer.num-epochs 3 \
    --trainer.optimization.initial-effective-lrate 0.0001 \
    --trainer.optimization.final-effective-lrate 0.00005 \
    --trainer.dropout-schedule '0,0@0.20,0.1@0.50,0' \
    --egs.stage 0 \
    --cleanup.remove-egs true \
    --use-gpu true
fi

# -------------------------------
# Stage 3: Completion Message
# -------------------------------
if [ $stage -le 3 ]; then
  echo "=== Stage 3: Fine-tuning completed successfully ==="
  echo "✅ Adapted TDNN model available at:"
  echo "👉 $adapt_dir/final.mdl"
fi
