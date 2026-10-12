# llm-identifier

Course project by Anish Chitnis. The goal is to guess which LLM family wrote a text, using a CNN and an LSTM.

The full write up is in `report.pdf`.

## Task

Each row is (LLM name, input, output). A model reads the text and predicts one of five families:
GPT, Llama 2, Bard, Falcon and MPT.

- RQ1. Can we find the family from the output alone?
- RQ2. What changes if the model sees only the input, only the output, or both?

## Data

The data comes from the Hugging Face dataset `openbmb/UltraFeedback` (MIT license).
`src/build_dataset.py` maps its 17 models to the five families, keeps answers between 20 and 8000 characters,
removes duplicates, and keeps at most 1000 rows for each family and domain pair.
The domain is the source dataset of the prompt (Evol-Instruct, Flan, ShareGPT, UltraChat).
Bard only has 701 Flan answers in UltraFeedback, so that pair is smaller.

`src/preprocess.py` removes answers that say the model name, such as "As ChatGPT", and then splits 80, 10, 10
by prompt, so the same prompt never appears in two splits.

The data files are not stored in this repo. The scripts download and rebuild them.

## Main results

Test macro-F1, output only, 384 tokens. Chance is 0.20.

| Model | Macro-F1 |
| --- | --- |
| Majority class | 0.062 |
| TF-IDF words + logistic regression | 0.773 |
| TF-IDF characters + logistic regression | 0.769 |
| CNN (3 seeds) | 0.743 |
| BiLSTM (3 seeds) | 0.708 |

RQ2 (macro-F1):

| Model | Output only | Input only | Input + output |
| --- | --- | --- | --- |
| CNN | 0.743 | 0.219 | 0.732 |
| BiLSTM | 0.708 | 0.217 | 0.704 |

See the report for the truncation study and the analysis of why input only is a little above chance.

## How to run

A GPU is needed for the neural networks. It takes about an hour and a half on a Colab T4.

```
pip install -r requirements.txt
bash run_all.sh
python src/make_figures.py
python src/make_report.py
```

## Files

- `src/build_dataset.py` builds the dataset from UltraFeedback
- `src/preprocess.py` removes name leaks and makes the splits
- `src/baselines.py` majority class and TF-IDF baselines
- `src/common.py` tokenizer, vocabulary and encoding
- `src/models.py` the CNN and the BiLSTM
- `src/train.py` training and evaluation
- `src/summarize.py` collects the results into one table
- `src/rq2_diagnostics.py` checks on the input only result
- `src/make_figures.py` draws the figures
- `src/make_report.py` builds `report.pdf`
- `run_all.sh` runs the whole pipeline
- `results/` result tables
- `figures/` figures used in the report
