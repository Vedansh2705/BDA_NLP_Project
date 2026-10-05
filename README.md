# Hadoop-based Twitter Sentiment Analysis: Replication

Replication of **"Apache Hadoop based effective sentiment analysis on demonetization and covid-19 tweets"**
(S. Anitha, M. Metilda, *Global Transitions Proceedings* 3 (2022) 338-342, doi:10.1016/j.gltp.2022.03.021).

Big Data Analytics & NLP project, SVKM's NMIMS (MPSTME), Indore.

## What was done
1. Pre-processing: RT/URL/mention removal, lower-casing, punctuation removal, tokenization, stop-word removal, Porter stemming.
2. **Algorithm 2 (TSA)** in **Apache Pig 0.12** on a Cloudera QuickStart VM (CDH 5.13): AFINN dictionary join, `Avg_R` per tweet, polarity (positive / negative / neutral).
3. Classification with SVM (LinearSVC), Naive Bayes and an LSTM-based RNN on TF-IDF / token sequences.
4. Comparison with the paper's Tables 3-6.

## Key results (20% hold-out, accuracy %)
| Dataset | SVM | Naive Bayes | RNN |
|---|---|---|---|
| Demonetization (14,912 tweets) | 94.37 | 88.94 | 92.16 |
| COVID-19 (143,480 tweets) | 87.68 | 75.13 | 89.75 |

Pig output on the full demonetization data matches the Python/AFINN computation exactly
(4,228 negative / 4,933 neutral / 5,751 positive). For COVID-19, 60,000 tweets were processed via Pig;
the rest with the equivalent Python implementation.

> Labels come from an AFINN lexicon rule, so accuracy here means agreement with lexicon labels,
> not accuracy against human sentiment judgement. See the report, Section 6.

## Repository layout
```
sentiment_replication.ipynb   EDA, preprocessing, scoring, ML
pig/                          Pig script (tsa_algorithm.pig)
docs/                         Project report and Cloudera evidence
results/                      Figures
data/                         Put Kaggle CSVs here (git-ignored)
```

## Data
Download from Kaggle and place in `data/` (filenames used in the notebook):
- `demonetization-tweets.csv`
- `Covid-19 Twitter Dataset (Apr-Jun 2020).csv`

## Run
```bash
pip install -r requirements.txt
jupyter notebook sentiment_replication.ipynb
```
Note: RNN results vary slightly between runs (no fixed TensorFlow seed).
