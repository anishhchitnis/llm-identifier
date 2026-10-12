mkdir -p logs results data
pip install -q datasets pyarrow > logs/pip.txt 2>&1

if ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "No GPU found. In Colab choose Runtime > Change runtime type > T4 GPU, then run this again."
  exit 1
fi

step() {
  name="$1"
  shift
  echo "starting $name"
  if "$@" > "logs/$name.txt" 2>&1; then
    echo "finished $name"
  else
    echo "FAILED $name (see logs/$name.txt)"
  fi
}

step build_dataset python src/build_dataset.py
step preprocess python src/preprocess.py
step baselines python src/baselines.py
step cnn_output python src/train.py --model cnn --mode output --seeds 3
step lstm_output python src/train.py --model lstm --mode output --seeds 3

for L in 16 32 64 128 256; do
  step cnn_len$L python src/train.py --model cnn --mode output --seeds 3 --max_len $L --tag len$L
  step lstm_len$L python src/train.py --model lstm --mode output --seeds 3 --max_len $L --tag len$L
done

for M in cnn lstm; do
  for MODE in input both; do
    step ${M}_$MODE python src/train.py --model $M --mode $MODE --seeds 3
  done
done

step rq2_diagnostics python src/rq2_diagnostics.py

python src/summarize.py | tee logs/summary.txt
zip -rq results_backup.zip results logs data/test.parquet data/dataset_stats.csv data/leak_rows.parquet
echo "saved results_backup.zip"
