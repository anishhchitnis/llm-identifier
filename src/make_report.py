from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FONTS = Path("/usr/share/fonts/truetype/dejavu")

pdfmetrics.registerFont(TTFont("Body", str(FONTS / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("Bold", str(FONTS / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("Ital", str(FONTS / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold", italic="Ital", boldItalic="Bold")

INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
GRID = colors.HexColor("#c3c2b7")
HEAD = colors.HexColor("#eceae2")

body = ParagraphStyle("body", fontName="Body", fontSize=9.6, leading=14.2, textColor=INK, spaceAfter=7, alignment=TA_LEFT)
h1 = ParagraphStyle("h1", parent=body, fontName="Bold", fontSize=13.5, leading=17, spaceBefore=12, spaceAfter=6, keepWithNext=1)
h2 = ParagraphStyle("h2", parent=body, fontName="Bold", fontSize=10.8, leading=14, spaceBefore=8, spaceAfter=4, keepWithNext=1)
title = ParagraphStyle("title", parent=body, fontName="Bold", fontSize=19, leading=24, spaceAfter=4)
sub = ParagraphStyle("sub", parent=body, fontSize=10, leading=14, textColor=INK2, spaceAfter=2)
cap = ParagraphStyle("cap", parent=body, fontSize=8.4, leading=11.5, textColor=INK2, spaceAfter=10)
cell = ParagraphStyle("cell", parent=body, fontSize=8.4, leading=11, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Bold")
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=3, spaceAfter=3)


def p(text):
    return Paragraph(text, body)


def bl(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def table(rows, widths, caption, header=True):
    data = []
    for r, row in enumerate(rows):
        data.append([Paragraph(str(c), cellb if (r == 0 and header) else cell) for c in row])
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, GRID),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK2),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK2),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), HEAD))
    t.setStyle(TableStyle(style))
    return KeepTogether([t, Spacer(1, 3), Paragraph(caption, cap)])


def fig(name, caption, width=6.3):
    from PIL import Image as PILImage

    w, h = PILImage.open(FIGURES / name).size
    img = Image(str(FIGURES / name), width=width * inch, height=width * inch * h / w)
    return KeepTogether([img, Spacer(1, 3), Paragraph(caption, cap)])


def f3(x):
    return f"{x:.3f}"


def pm(m, s):
    return f"{m:.3f} ± {s:.3f}"


summary = pd.read_csv(RESULTS / "summary_table.csv").set_index("run")
base = pd.read_csv(RESULTS / "baselines.csv").set_index("model")
chi = pd.read_csv(RESULTS / "prompt_length_chi2.csv")
quart = pd.read_csv(RESULTS / "flan_prompt_length_quartiles.csv")
prompt_only = pd.read_csv(RESULTS / "prompt_only_baselines.csv").set_index("classifier")
perseed = pd.read_csv(RESULTS / "neural_output_per_seed.csv")
stats = pd.read_csv(RESULTS / "dataset_stats.csv")
cells = pd.read_csv(RESULTS / "rows_per_family_domain.csv")
splits = pd.read_csv(RESULTS / "split_sizes.csv")
leaks = pd.read_csv(RESULTS / "leak_counts.csv")
charrep = pd.read_csv(RESULTS / "baseline_char_report.csv").set_index("family")

story = []

story.append(Paragraph("Which LLM wrote this text? Fingerprinting LLM families with a CNN and an RNN", title))
story.append(Paragraph("Anish Chitnis, Penn State. Individual project.", sub))
story.append(Paragraph("Code and results: https://github.com/anishhchitnis/llm-identifier", sub))
story.append(Spacer(1, 8))

story.append(Paragraph("Summary", h1))
story.append(p(
    "I trained a CNN and a BiLSTM to guess which LLM family wrote an answer. The data has five families, "
    "which are GPT, Llama 2, Bard, Falcon and MPT. When the models see only the answer text, they get about "
    "0.74 (CNN) and 0.71 (BiLSTM) macro-F1 on a test set where random guessing gives 0.20. A simple TF-IDF and "
    "logistic regression baseline did slightly better, at 0.77. When the models see only the prompt, they are "
    "close to chance at 0.22. Adding the prompt to the answer did not help. I also found that the small gap "
    "above chance for prompt-only models comes from a data quirk and not from real fingerprints in the prompt."
))

story.append(Paragraph("1. Problem definition and dataset", h1))
story.append(Paragraph("1.1 The problem", h2))
story.append(p(
    "Many LLMs write fluent text, so it is hard to tell by eye who wrote what. The task here is to look at a "
    "text and predict the LLM family that produced it. This is a 5 class classification problem. "
    "I answer these two research questions."
))
story.extend(bl([
    "<b>RQ1.</b> Can we identify the LLM family from the output text alone?",
    "<b>RQ2.</b> How does the result change if the model sees only the input (the prompt), only the output, or both?",
]))

story.append(Paragraph("1.2 Where the data comes from", h2))
story.append(p(
    "I used the public UltraFeedback dataset from Hugging Face (openbmb/UltraFeedback, MIT license). It has "
    "63,967 prompts. Each prompt has four answers, written by four of 17 different models. Each row has a "
    "prompt, an answer, the name of the model that wrote it, and the name of the source dataset the prompt came from. "
    "This fits the project format directly, because each row is (LLM name, input, output). "
    "I did not need any API keys, so all data collection was free."
))
story.append(p(
    "I grouped the models into families. GPT is GPT-4 and GPT-3.5-turbo. Llama 2 is the 7B, 13B and 70B chat models. "
    "Bard, Falcon (40B instruct) and MPT (30B chat) each have one model. That gives five families, which is more "
    "than the three the project asks for. I used the source dataset of each prompt as its domain. There are four "
    "domains, which are Evol-Instruct, Flan, ShareGPT and UltraChat. The Flan source is split into several parts "
    "in the data, and I merged them into one domain."
))

story.append(Paragraph("1.3 Cleaning and balancing", h2))
story.extend(bl([
    "I kept answers between 20 and 8000 characters and dropped duplicates.",
    "I capped every family and domain pair at 1000 rows, so no single family or domain can dominate. "
    "Bard only has 701 Flan answers in the whole dataset, so that one pair has 701 rows.",
    "Some answers name the model, for example \"As ChatGPT\" or \"I am OpenAI\". A classifier could just read the name. "
    "I found these with a pattern match and removed them. This was about 1 percent of the rows, or 183 rows in total.",
    "I split the data into train, validation and test with an 80, 10, 10 split. The split is grouped by prompt text, "
    "so the same prompt never appears in two splits. I checked that the overlap is zero. This matters because "
    "answers to the same prompt are alike, and a random split would leak information.",
]))

rows = [["Family", "Rows kept", "Mean words", "Median words", "Self-naming rows removed"]]
names = {"sharegpt": "ShareGPT", "evol_instruct": "Evol-Instruct", "ultrachat": "UltraChat", "flan": "Flan", "bard": "Bard", "falcon": "Falcon", "gpt": "GPT", "llama2": "Llama 2", "mpt": "MPT"}
lk = leaks.set_index("llm_name")
for _, r in stats.iterrows():
    rows.append([names[r["llm_name"]], f"{r['rows']:,}", f"{r['mean_words']:.0f}", f"{r['median_words']:.0f}", int(lk.loc[r["llm_name"], "self_naming_rows"])])
rows.append(["Total", f"{int(stats['rows'].sum()):,}", "", "", int(leaks["self_naming_rows"].sum())])
story.append(table(rows, [1.2, 1.1, 1.1, 1.2, 1.9], "Table 1. The dataset before the leak rows were removed. Falcon answers are much shorter than the others."))

story.append(fig("fig_length.png", "Figure 1. Mean and median answer length for each family. Length is one clear difference between families."))

rows = [["Group", "Train", "Validation", "Test"]]
for _, r in splits.iterrows():
    rows.append([names.get(r["group"], r["group"].replace("_", " ").title() if r["group"] != "all" else "All rows"), f"{r['train']:,}", f"{r['val']:,}", f"{r['test']:,}"])
story.append(table(rows, [2.0, 1.2, 1.2, 1.2], "Table 2. Rows in each split. All rows is 19,518 after the leak rows were removed."))

story.append(Paragraph("2. Models and training", h1))
story.append(Paragraph("2.1 Input to the networks", h2))
story.append(p(
    "Both networks use the same tokenizer. It splits text into words, punctuation marks and newlines. It does not "
    "lowercase, because capital letters and formatting like bullet points and line breaks can be part of a model's style. "
    "The vocabulary has the 30,000 most common tokens from the training set only. Each answer is cut or padded to 384 tokens. "
    "For RQ2, the input and output are joined into one sequence. The input is cut to 128 tokens, then a separator token, "
    "then the output."
))
story.append(Paragraph("2.2 The CNN", h2))
story.append(p(
    "The CNN is a TextCNN. It has a 200 dimension embedding layer, then three parallel 1D convolutions with kernel sizes "
    "3, 4 and 5 and 128 filters each. After each convolution it uses a ReLU and a max pool over the whole text. Padding "
    "is masked so it cannot win the max pool. The three pooled vectors are joined, passed through dropout of 0.5, "
    "and then a linear layer gives five scores. It has about 6.31 million parameters."
))
story.append(Paragraph("2.3 The RNN", h2))
story.append(p(
    "The RNN is a bidirectional LSTM with hidden size 128 in each direction and the same 200 dimension embedding. "
    "It uses packed sequences so padding is ignored. I join the mean pool and the max pool of the hidden states, "
    "apply dropout of 0.5, and use a linear layer. It has about 6.34 million parameters."
))
story.append(Paragraph("2.4 Training details", h2))
story.append(table([
    ["Setting", "Value"],
    ["Optimizer", "Adam, learning rate 0.001"],
    ["Batch size", "64"],
    ["Loss", "Cross entropy with class weights"],
    ["Gradient clipping", "1.0"],
    ["Max epochs", "20"],
    ["Early stopping", "Stop when validation macro-F1 does not improve for 3 epochs, keep the best epoch"],
    ["Seeds", "3 seeds for every neural result. I report mean and standard deviation"],
    ["Hardware", "One T4 GPU on Google Colab"],
], [1.6, 4.9], "Table 3. Training settings, the same for the CNN and the BiLSTM."))
story.append(p(
    "I only used validation data to pick the best epoch and to pick the regularization strength C for the baselines. "
    "The test set was used once per final model. Macro-F1 is my main number because it treats all five families the same. "
    "I also report accuracy."
))
story.append(Paragraph("2.5 Baselines", h2))
story.append(p(
    "I added three reference points. The first is a majority class guess. The second is TF-IDF with word 1 and 2 grams plus "
    "logistic regression. The third is TF-IDF with character 2 to 5 grams plus logistic regression. The character version "
    "keeps case, punctuation and newlines. These baselines help show how much the neural networks add."
))

story.append(Paragraph("3. Results", h1))
story.append(Paragraph("3.1 RQ1, output only", h2))
rows = [["Model", "Test accuracy", "Test macro-F1"]]
rows.append(["Majority class", f3(base.loc["majority_class", "test_acc"]), f3(base.loc["majority_class", "test_macro_f1"])])
rows.append(["TF-IDF words + logistic regression", f3(base.loc["tfidf_word_output", "test_acc"]), f3(base.loc["tfidf_word_output", "test_macro_f1"])])
rows.append(["TF-IDF characters + logistic regression", f3(base.loc["tfidf_char_output", "test_acc"]), f3(base.loc["tfidf_char_output", "test_macro_f1"])])
rows.append(["CNN (3 seeds)", pm(summary.loc["cnn_output", "acc_mean"], summary.loc["cnn_output", "acc_std"]), pm(summary.loc["cnn_output", "f1_mean"], summary.loc["cnn_output", "f1_std"])])
rows.append(["BiLSTM (3 seeds)", pm(summary.loc["lstm_output", "acc_mean"], summary.loc["lstm_output", "acc_std"]), pm(summary.loc["lstm_output", "f1_mean"], summary.loc["lstm_output", "f1_std"])])
story.append(table(rows, [3.1, 1.7, 1.7], "Table 4. RQ1 results on the test set with 384 tokens of the output. Random guessing among five classes is 0.20 accuracy."))
story.append(fig("fig_models.png", "Figure 2. Test macro-F1 for each model. The grey line is chance at 0.20."))
story.append(p(
    "The answer to RQ1 is yes. Every real model is far above the 0.20 chance line. The CNN reaches 0.743 macro-F1 and the BiLSTM "
    "reaches 0.708. The TF-IDF baselines are a bit better at 0.773 and 0.769. So on this data, the neural networks do not "
    "beat a simple linear model on word counts. I think this is because the style signals are strong and local, "
    "like the words \"Certainly\" or \"happy to help\", and a linear model can use those well. "
    "The CNN is 3.5 points ahead of the BiLSTM, but the BiLSTM changes a lot between seeds, as Table 5 shows, "
    "so I would not call that gap firm."
))
rows = [["Model", "Seed", "Test accuracy", "Test macro-F1", "Epochs run"]]
for _, r in perseed.iterrows():
    rows.append([r["model"].upper() if r["model"] == "cnn" else "BiLSTM", int(r["seed"]), f3(r["test_acc"]), f3(r["test_macro_f1"]), int(r["epochs_run"])])
story.append(table(rows, [1.4, 0.8, 1.5, 1.5, 1.2], "Table 5. Each seed for the output only models at 384 tokens. BiLSTM seed 1 ran all 20 epochs. The other runs stopped early."))

story.append(Paragraph("3.2 Which families are easy and which are hard", h2))
story.append(fig("fig_recall.png", "Figure 3. Recall for each true family. Recall is the share of that family's answers that were found."))
story.append(fig("fig_confusion.png", "Figure 4. Confusion matrices for the CNN and the BiLSTM. Each row adds to 100 percent. The diagonal is the correct answers."))
story.append(p(
    "Bard is the easiest family. The BiLSTM finds 94 percent of Bard answers and the CNN finds 85 percent. Llama 2 is next. "
    "MPT is the hardest. The CNN finds 62 percent of MPT answers and the BiLSTM only 41 percent. MPT answers are most "
    "often mistaken for GPT and Falcon. GPT and Falcon are also mixed up with each other. The BiLSTM sent many MPT "
    "answers to Bard, which is why its MPT recall is so low."
))

story.append(Paragraph("4. Analysis and experiments", h1))
story.append(Paragraph("4.1 How much text does the model need? (truncation)", h2))
story.append(p(
    "I trained both networks again with the output cut to 16, 32, 64, 128, 256 and 384 tokens. Each setting used 3 seeds."
))
rows = [["Tokens kept", "CNN macro-F1", "BiLSTM macro-F1"]]
for n, suffix in [(16, "_len16"), (32, "_len32"), (64, "_len64"), (128, "_len128"), (256, "_len256"), (384, "")]:
    c = summary.loc[f"cnn_output{suffix}"]
    l = summary.loc[f"lstm_output{suffix}"]
    rows.append([n, pm(c["f1_mean"], c["f1_std"]), pm(l["f1_mean"], l["f1_std"])])
story.append(table(rows, [1.6, 2.2, 2.2], "Table 6. Test macro-F1 when only the first N tokens of the answer are kept."))
story.append(fig("fig_truncation.png", "Figure 5. Macro-F1 against tokens kept. The shaded band is one standard deviation. The grey line at the top is the best TF-IDF baseline."))
story.append(p(
    "Even 16 tokens, which is about one short sentence, gives 0.567 for the CNN and 0.561 for the BiLSTM. That is almost three "
    "times chance. This means the start of an answer already shows a lot of style, for example how a model begins with "
    "\"Sure\" or \"Certainly\". The score rises quickly until about 128 tokens and then flattens. The CNN stays slightly ahead of the "
    "BiLSTM at every length, but at 16 tokens they are about the same. The BiLSTM at 384 is lower than at 256, but its spread is large, so I would not "
    "read much into that drop."
))

story.append(Paragraph("4.2 RQ2, input only, output only, or both", h2))
rows = [["Model", "Output only", "Input only", "Input + output"]]
for name, k in [("CNN", "cnn"), ("BiLSTM", "lstm")]:
    rows.append([name] + [pm(summary.loc[f"{k}_{m}", "f1_mean"], summary.loc[f"{k}_{m}", "f1_std"]) for m in ["output", "input", "both"]])
rows.append(["TF-IDF words", f3(base.loc["tfidf_word_output", "test_macro_f1"]), f3(base.loc["tfidf_word_input_only", "test_macro_f1"]), f3(base.loc["tfidf_word_input_plus_output", "test_macro_f1"])])
story.append(table(rows, [1.5, 1.7, 1.7, 1.7], "Table 7. Test macro-F1 for RQ2. Chance is 0.20."))
story.append(fig("fig_rq2.png", "Figure 6. RQ2 results. The grey line is chance at 0.20."))
story.extend(bl([
    "<b>Output only</b> works well, as in RQ1.",
    "<b>Input only</b> is near chance, at 0.219 for the CNN and 0.217 for the BiLSTM. This is what I expected, because "
    "the prompt is written before the model answers.",
    "<b>Input and output together</b> is not better than output only. The CNN goes from 0.743 to 0.732 and the BiLSTM from "
    "0.708 to 0.704. The TF-IDF model drops more, from 0.773 to 0.722. The prompt adds noise and nothing useful.",
]))
story.append(p(
    "So the answer to RQ2 is that the output carries almost all of the signal, and the input adds nothing."
))

story.append(Paragraph("4.3 Why is input only a little above chance?", h2))
story.append(p(
    "Input only scores 0.22, a little above 0.20. The BiLSTM input only model also finds 55 percent of Bard answers, but only 12 to 22 percent "
    "of the answers from each other family. That looked strange, since the prompt should not say who answers. I looked for a cause."
))
story.append(p(
    "The cause is a data quirk. Bard has only 701 Flan rows, while every other family has 1000 in each domain. "
    "This means that a Flan prompt is a bit more likely to be from a non-Bard family overall. But Bard's Flan rows are not spread evenly over "
    "prompt lengths. I split the Flan prompts into four equal groups by length and measured the share of Bard rows in each. "
    "Bard's overall share of Flan rows is 15 percent, but it is 26 percent for the shortest prompts and only 4 percent for the second shortest."
))
story.append(fig("fig_bard_flan.png", "Figure 7. Bard share of Flan rows by prompt length group. If prompt length told us nothing, every bar would be near the 15 percent line."))
rows = [["Domain", "Chi-square", "Degrees of freedom", "p-value"]]
for _, r in chi.iterrows():
    pv = "below 0.0001" if r["p_value"] < 0.0001 else f"{r['p_value']:.3f}"
    rows.append([{"evol_instruct": "Evol-Instruct", "flan": "Flan", "sharegpt": "ShareGPT", "ultrachat": "UltraChat"}[r["domain"]], f"{r['chi2']:.1f}", int(r["dof"]), pv])
story.append(table(rows, [1.8, 1.4, 1.6, 1.4], "Table 8. Chi-square test of prompt length group against family, for each domain. Only Flan shows a link."))
rows = [["Classifier that sees only prompt length and domain", "Test accuracy", "Test macro-F1"]]
rows.append(["Majority class", f3(prompt_only.loc["majority_class", "test_acc"]), f3(prompt_only.loc["majority_class", "test_macro_f1"])])
rows.append(["Domain only", f3(prompt_only.loc["domain_only", "test_acc"]), f3(prompt_only.loc["domain_only", "test_macro_f1"])])
rows.append(["Prompt length and domain", f3(prompt_only.loc["prompt_length_and_domain", "test_acc"]), f3(prompt_only.loc["prompt_length_and_domain", "test_macro_f1"])])
story.append(table(rows, [3.6, 1.4, 1.4], "Table 9. A simple classifier that cannot read any words gets the same score as the neural input only models."))
story.append(p(
    "The chi-square test is strongly significant for Flan and not for the other three domains. A simple classifier that only knows the "
    "prompt length and the domain gets 0.216 macro-F1. That is the same as the CNN and BiLSTM input only results of 0.219 and 0.217. "
    "So the neural models are not finding any fingerprint in the prompt words. They found the Bard and Flan length pattern. "
    "This is a good example of a shortcut, and it shows why checking the data matters as much as the model."
))

story.append(Paragraph("4.4 What the baselines learned", h2))
story.append(p(
    "The top logistic regression features match what I see in real chat output. Llama 2 uses exclamation marks and phrases like "
    "\"happy to help\" and \"respectful\". Bard uses bullet stars and blank lines. MPT often says \"Certainly\" and "
    "ends with \"Overall\". GPT uses phrases like \"Confidence: 95\", which probably come from the way some prompts ask for a confidence level. "
    "Falcon features include the words \"assistant\" and \"user\" and leftover tag pieces like \"&lt; /\". "
    "Falcon answers are also much shorter, with a median of 70 words against 161 to 245 for the others."
))
story.append(p(
    "Some of these features say more about the training data and chat template of each model than about the model's deep style. "
    "That is still a valid fingerprint, but it may be easy to hide. It also means these results may not carry over to other prompts."
))

story.append(Paragraph("5. Limits of this work", h1))
story.extend(bl([
    "All text comes from one dataset, UltraFeedback. The results may not hold for other prompts or for newer models.",
    "A family can have more than one model, such as GPT-4 and GPT-3.5, and Llama 2 in three sizes. I did not test whether the "
    "classifiers could tell models in the same family apart.",
    "In UltraFeedback each prompt gets four of the 17 models, so not every family answered every prompt.",
    "Bard has fewer rows than the other families because of how much Bard data exists in the source.",
    "I used 3 seeds, so the standard deviations are rough. I did not do a heavy search for hyperparameters.",
]))

story.append(Paragraph("6. Lessons and experience", h1))
story.append(p(
    "The biggest thing I learned is that the data work took more thinking than the models. Early on I thought "
    "the hard part would be the CNN and the LSTM, but the CNN and the LSTM were short to write and trained without trouble. "
    "Getting the data right took much longer. I had to find that the Flan prompts were under a different name, so my first "
    "dataset only had three domains. I also had to find the self-naming answers where a model says its own name, since a "
    "classifier would just read that and look smart for the wrong reason."
))
story.append(p(
    "I also learned to split by prompt. At first it seemed fine to split rows at random, but then the same prompt would be in train and test "
    "with different answers, and the scores would be too high. Grouping by prompt fixed that."
))
story.append(p(
    "A surprise was that the simple TF-IDF baseline beat my neural networks. At first I felt bad about that, but I think it is a useful result. "
    "It told me that the style signal is mostly in which words and symbols appear, and not in long range structure. If I had skipped the baseline "
    "I would have thought 0.74 was great without knowing that a much simpler model gets 0.77."
))
story.append(p(
    "The input only result taught me the most. A score of 0.22 looked tiny, and I almost wrote it off as noise. But it was "
    "consistent across both models, so I dug in. The real cause was that Bard has fewer Flan rows. It would have been easy to "
    "claim that the prompt leaks a bit of information about the model. That would have been wrong. "
    "Next time I will check simple explanations before I make a claim."
))
story.append(p(
    "On the practical side, next time I will save and download result files after every step. If I had more time, I would "
    "test the models on a family that was not in training, and look at the wrong predictions one by one to see what the MPT answers "
    "that look like GPT have in common."
))

story.append(Paragraph("Reproducing this", h1))
story.append(p(
    "All code is in the repo under src. The file run_all.sh runs the full pipeline, which is dataset building, preprocessing, baselines, "
    "the CNN and BiLSTM runs, the truncation runs, the RQ2 runs and the diagnostics. It needs a GPU and takes about an hour and a half on a "
    "Colab T4. The file src/make_figures.py draws the figures from the files in the results folder."
))


fixed = []
for item in story:
    if isinstance(item, Paragraph) and item.style in (h1, h2):
        fixed.append(CondPageBreak(3.2 * inch))
    fixed.append(item)
story = fixed


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 8)
    canvas.setFillColor(INK2)
    canvas.drawString(0.9 * inch, 0.55 * inch, "Anish Chitnis. LLM fingerprinting")
    canvas.drawRightString(letter[0] - 0.9 * inch, 0.55 * inch, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(
    str(ROOT / "report.pdf"), pagesize=letter, leftMargin=0.9 * inch, rightMargin=0.9 * inch,
    topMargin=0.85 * inch, bottomMargin=0.9 * inch, title="LLM fingerprinting", author="Anish Chitnis",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", ROOT / "report.pdf")
