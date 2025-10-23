#!/usr/bin/env bash
# --------------------------------------------------------
# Fine-tune Librispeech TDNN Chain model on Speechocean762
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


lang=/home/mcw/durga/kaldi/libreespeech2/0013_librispeech_v1_lm/data/lang_test_tgsmall
pretrained_model=/home/mcw/durga/kaldi/libreespeech2/0013_librispeech_v1_chain/exp/chain_cleaned/tdnn_1d_sp
ivector_extractor=/home/mcw/durga/kaldi/libreespeech2/0013_librispeech_v1_extractor/exp/nnet3_cleaned/extractor
adapt_dir=tune/chain_finetune/tdnn_1d_sp_adapt
mfcc_conf=conf/mfcc_hires.conf
align_dir=tune/align_speechocean762
lat_dir=tune/align_lats_speechocean762
tree_dir=/home/mcw/durga/kaldi/libreespeech2/0013_librispeech_v1_chain/exp/chain_cleaned/tdnn_1d_sp


# -------------------------------
# Stage 0: Feature Extraction
# -------------------------------
if [ $stage -le 0 ]; then
  echo "=== Stage 0: Extracting MFCCs and i-vectors ==="
  for x in $train_set $dev_set; do
  utils/fix_data_dir.sh data/$x 
  steps/make_mfcc.sh --nj $nj --mfcc-config $mfcc_conf \
    data/$x tune/make_hires/$x mfcc
  steps/compute_cmvn_stats.sh data/$x tune/make_hires/$x mfcc
done


  echo "Extracting i-vectors..."
  for x in $train_set $dev_set; do
    steps/online/nnet2/extract_ivectors_online.sh --nj $nj \
      data/$x $ivector_extractor tune/nnet3_cleaned/ivectors_$x
  done
fi

# -------------------------------
# Stage 1: Align with pretrained model
# -------------------------------
# if [ $stage -le 1 ]; then
#   echo "=== Stage 1: Aligning training data with pretrained TDNN model ==="
#   steps/nnet3/align_lats.sh --cmd "run.pl" --nj $nj \
#     --online-ivector-dir tune/nnet3_cleaned/ivectors_$train_set \
#     data/$train_set $lang $pretrained_model $align_dir

# fi

# -------------------------------
# Stage 2: Generate lattices (CHAIN)
# -------------------------------
if [ $stage -le 2 ]; then
  echo "=== Stage 2: Generating lattices for chain fine-tuning ==="
  echo "Removing old lattices (if any)..."
  rm -rf $lat_dir
  mkdir -p $lat_dir
  echo "Generating new lattices..."
  steps/nnet3/align_lats.sh \
  --nj 10 \
  --cmd run.pl \
  --acoustic-scale 1.0 \
  --scale-opts '--transition-scale=1.0 --self-loop-scale=1.0' \
  --generate-ali-from-lats true \
  --online-ivector-dir tune/nnet3_cleaned/ivectors_${train_set} \
  data/${train_set} $lang $pretrained_model $lat_dir

fi
# if [ $stage -le 2 ]; then
#   echo "=== Stage 2: Generating lattices using GMM aligner ==="
#   rm -rf $lat_dir
#   mkdir -p $lat_dir
#   echo "Generating new lattices with fMLLR adaptation..."

#   steps/align_fmllr_lats.sh \
#     --nj 10 \
#     --cmd run.pl \
#     --acoustic-scale 0.1 \
#     --scale-opts "--transition-scale=1.0 --self-loop-scale=0.1" \
#     --fmllr-update-type full \
#     --generate-ali-from-lats true \
#     data/${train_set} $lang $pretrained_model $lat_dir || exit 1;
# fi

# -------------------------------
# Stage 3: Create feat_dim and egs/info
# -------------------------------
if [ $stage -le 3 ]; then
  echo "=== Stage 3: Generating chain egs ==="
  steps/nnet3/chain/get_egs.sh \
  data/$train_set $pretrained_model \
  $lat_dir $adapt_dir/egs

fi

# -------------------------------
# Stage 4: Fine-tuning TDNN
# -------------------------------
if [ $stage -le 3 ]; then
  echo "=== Stage 3: Fine-tuning chain TDNN model ==="
  steps/nnet3/chain/train.py \
    --stage 0 \
    --feat-dir data/$train_set \
    --feat.online-ivector-dir tune/nnet3_cleaned/ivectors_$train_set \
    --lat-dir $lat_dir \
    --tree-dir $tree_dir \
    --trainer.input-model $pretrained_model/final.mdl \
    --dir $adapt_dir \
    # --trainer.num-epochs 3 \
    # --trainer.optimization.initial-effective-lrate 0.0001 \
    # --trainer.optimization.final-effective-lrate 0.00005 \
    # --trainer.dropout-schedule '0,0@0.20,0.1@0.50,0' \
    # --cleanup.remove-egs true \
    --egs.opts "--frame-subsampling-factor 3 --ali-dir $lat_dir"
    --use-gpu true
fi

# -------------------------------
# Stage 5: Completion Message
# -------------------------------
if [ $stage -le 5 ]; then
  echo "=== Stage 3: Fine-tuning completed successfully ==="
  echo "Adapted TDNN model available at:"
  echo "$adapt_dir/final.mdl"
fi
