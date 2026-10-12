# LLM identifier

Course project by Anish Chitnis. The goal is to guess which LLM family wrote a text, using a CNN and an LSTM.

The full write up is in report.pdf.

## Task

Each row is (LLM name, input, output). A model reads the text and predicts one of five families, which are GPT, Llama 2, Bard, Falcon and MPT.

* RQ1. Can we find the family from the output alone?
* RQ2. What changes if the model sees only the input, only the output, or both?
* RQ3. Do the fingerprints carry over to a domain the model never saw in training?
* RQ4. What features of the text tell the LLMs apart?

## Data

The data comes from the Hugging Face dataset openbmb/UltraFeedback (MIT license).

src/build_dataset.py maps its 17 models to the five families, keeps answers between 20 and 8000 characters, removes duplicates, and keeps at most 1000 rows for each family and domain pair. The domain is the source dataset of the prompt, which is Evol Instruct, Flan, ShareGPT or UltraChat. Bard only has 701 Flan answers in UltraFeedback, so that pair is smaller.

src/preprocess.py removes answers that say the model name, such as "As ChatGPT", and then splits the data 80, 10, 10 by prompt, so the same prompt never appears in two splits.

The data files are not stored in this repo. The scripts download and rebuild them.

## Main results

Test macro F1 with the output only and 384 tokens. Chance is 0.20.

* Majority class: 0.062
* TF IDF words and logistic regression: 0.773
* TF IDF characters and logistic regression: 0.769
* CNN (3 seeds): 0.743
* BiLSTM (3 seeds): 0.708

RQ2, test macro F1 for output only, input only, and input plus output.

* CNN: 0.743, 0.219, 0.732
* BiLSTM: 0.708, 0.217, 0.704

RQ3, test macro F1 when the domain was seen in training and when it was held out (CNN with 3 seeds, BiLSTM with 2 seeds).

* Evol Instruct: CNN 0.777 and 0.722, BiLSTM 0.702 and 0.625
* Flan: CNN 0.707 and 0.529, BiLSTM 0.714 and 0.513
* ShareGPT: CNN 0.761 and 0.687, BiLSTM 0.694 and 0.641
* UltraChat: CNN 0.745 and 0.704, BiLSTM 0.722 and 0.648

RQ4, test macro F1 with TF IDF and logistic regression after removing signals. Full text 0.772, formatting removed 0.712, first and last sentence removed 0.676, formatting only 0.594, function words only 0.585.

The report also has the truncation study and the analysis of why input only is a little above chance.

## How to run

A GPU is needed for the neural networks. I used a Colab T4.

1. Install the packages listed in requirements.txt.
2. Run these scripts in order with python. They are src/build_dataset.py, src/preprocess.py, src/baselines.py, src/train.py and src/rq2_diagnostics.py. The train script takes the options model, mode, seeds, max_len and holdout_domain, which you can see at the top of the file.
3. For RQ3, run src/train.py with the holdout_domain option set to each of evol_instruct, flan, sharegpt and ultrachat, then run src/rq3_analysis.py.
4. For RQ4, run src/rq4_analysis.py and then src/rq4_figures.py.
5. Run src/summarize.py to collect the results into one table.
6. Run src/make_figures.py to draw the figures.
7. Run src/make_report.py to build report.pdf.

## Files

* src/build_dataset.py builds the dataset from UltraFeedback
* src/preprocess.py removes name leaks and makes the splits
* src/baselines.py majority class and TF IDF baselines
* src/common.py tokenizer, vocabulary and encoding
* src/models.py the CNN and the BiLSTM
* src/train.py training and evaluation
* src/summarize.py collects the results into one table
* src/rq2_diagnostics.py checks on the input only result
* src/rq3_analysis.py tables and figures for RQ3
* src/rq4_analysis.py feature statistics and signal removal tests for RQ4
* src/rq4_figures.py figure for RQ4
* src/make_figures.py draws the figures
* src/make_report.py builds report.pdf
* results has the result tables
* figures has the figures used in the report
