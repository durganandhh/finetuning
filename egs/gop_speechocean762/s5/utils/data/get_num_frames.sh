#!/usr/bin/env bash
# Get total number of frames in a data directory
# Usage: get_num_frames.sh <data-dir>

if [ -f path.sh ]; then . ./path.sh; fi
. utils/parse_options.sh || exit 1

if [ $# -ne 1 ]; then
  echo "Usage: $0 <data-dir>" 1>&2
  exit 1
fi

data=$1

# Make sure utt2dur exists
if [ ! -f $data/utt2dur ]; then
  utils/data/get_utt2dur.sh $data 1>&2 || exit 1
fi

# Get frame shift in seconds
frame_shift=$(utils/data/get_frame_shift.sh $data) || exit 1

# Compute total frames
total_frames=$(awk -v s=$frame_shift '{n += $2} END{printf("%.0f\n", n / s)}' <$data/utt2dur)

echo $total_frames
exit 0
