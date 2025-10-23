#!/usr/bin/env bash
# Fine-tune TDNN Chain model on Speechocean762
# Assumes Kaldi env is set up: path.sh, cmd.sh, utils/, etc.

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
# lang=data/lang
# pretrained_model=exp/chain_cleaned/tdnn_1d_sp
# ivector_extractor=exp/nnet3_cleaned/extractor
# adapt_dir=exp/chain_finetune/tdnn_1d_sp_adapt

# Language directory
lang=/home/mcw/durga/kaldi/libreespeech2/0013_librispeech_v1_lm/data/lang_test_tgsmall

# Pretrained acoustic model (LibriSpeech TDNN)
pretrained_model=/home/mcw/durga/kaldi/libreespeech2/0013_librispeech_v1_chain/exp/chain_cleaned/tdnn_1d_sp

# I-vector extractor (LibriSpeech)
ivector_extractor=/home/mcw/durga/kaldi/libreespeech2/0013_librispeech_v1_extractor/exp/nnet3_cleaned/extractor

# Output adapted/fine-tuned model
adapt_dir=/home/mcw/durga/kaldi/egs/gop_speechocean762/s5/exp/chain_finetune/tdnn_1d_sp_adapt
mfcc_conf=conf/mfcc_hires.conf

# -------------------------------
# Stage 0: Feature extraction
# -------------------------------
if [ $stage -le 0 ]; then
  for x in $train_set $dev_set; do
    steps/make_mfcc.sh --nj $nj --mfcc-config $mfcc_conf \
      data/$x exp/make_hires/$x mfcc
    steps/compute_cmvn_stats.sh data/$x exp/make_hires/$x mfcc
    utils/fix_data_dir.sh data/$x
  done

  # Extract i-vectors for adaptation
  for x in $train_set $dev_set; do
    steps/online/nnet2/extract_ivectors_online.sh --nj $nj \
      data/$x $ivector_extractor exp/nnet3_cleaned/ivectors_$x
  done
fi

# -------------------------------
# Stage 1: Align train set using pretrained model
# -------------------------------
# if [ $stage -le 1 ]; then
#   steps/nnet3/align.sh --cmd "$train_cmd" --nj $nj --use_gpu true
#  \
#     --online-ivector-dir exp/nnet3_cleaned/ivectors_$train_set \
#     data/$train_set $lang $pretrained_model exp/align_speechocean762
# fi
if [ $stage -le 1 ]; then
    steps/nnet3/align.sh --cmd "run.pl" --nj $nj --use_gpu true \
        --online-ivector-dir exp/nnet3_cleaned/ivectors_$train_set \
        data/$train_set $lang $pretrained_model exp/align_speechocean762
# fi
# # ====== Stage 2: Fine-tuning ======
# if [ $stage -le 2 ]; then
#   echo "Starting fine-tuning from pretrained Librispeech TDNN model..."
  
#   steps/nnet3/chain/train_tdnn.sh \
#     --stage 0 \
#     --trainer.num-epochs 3 \
#     --trainer.optimization.initial-effective-lrate 0.0001 \
#     --trainer.optimization.final-effective-lrate 0.00005 \
#     --feat.cmvn-opts "--norm-means=true --norm-vars=true" \
#     --chain.adapt-model $pretrained_model/final.mdl \
#     --feat-dir data/$train_set \
#     --ali-dir exp/align_speechocean762 \
#     --dir $adapt_dir \
#     --online-ivector-dir exp/nnet3_cleaned/ivectors_$train_set

#   echo "✅ Fine-tuning complete. Model saved in: $adapt_dir"
# fi


# -------------------------------
# Stage 3: Done
# -------------------------------
if [ $stage -le 3 ]; then
  echo "Fine-tuning complete. Adapted model is in: $adapt_dir/final.mdl"
fi
