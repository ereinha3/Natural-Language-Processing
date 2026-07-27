# CS 510 — Natural Language Processing

Coursework from CS 510 (NLP), Spring 2025 — University of Oregon.
Three assignments, each a self-contained classical-NLP system built from scratch
(no ML frameworks): distributional embeddings, an HMM POS tagger with Viterbi
decoding, and a MEMM named-entity recognizer.

| # | Topic | Code | Write-up | Result (dev) |
|---|-------|------|----------|--------------|
| 1 | TF-IDF, Naive Bayes, Word2Vec similarity | [`assignment1.py`](assignment1.py) | [`CS510_HW1.pdf`](CS510_HW1.pdf) | qualitative |
| 2 | HMM POS tagger + Viterbi | [`HMM_and_Viterbi/`](HMM_and_Viterbi) | [`NLP_Assignment_1.pdf`](HMM_and_Viterbi/NLP_Assignment_1.pdf) | **94.58%** token accuracy |
| 3 | MEMM named-entity recognition | [`NER_MEMM/`](NER_MEMM) | [`CS_510_HW2.pdf`](NER_MEMM/CS_510_HW2.pdf) | **84.67** entity F1 / **97.23%** token accuracy |

All numbers above were reproduced from a clean run of the code in this repo, not
copied from the reports. See [Reproducing the results](#reproducing-the-results).

---

## Repository layout

```
.
├── assignment1.py              # HW1: Word2Vec similarity (Brown corpus vs. GoogleNews)
├── CS510_HW1.pdf               # HW1 written solutions
│
├── HMM_and_Viterbi/            # HW2: POS tagging
│   ├── main.py                 #   HMM training + Viterbi decoding + tagging driver
│   ├── scorer.py               #   accuracy scorer (course-provided)
│   ├── NLP_Assignment_1.pdf    #   report
│   └── data/                   #   WSJ-style .pos / .words corpora (course-provided)
│
└── NER_MEMM/                   # HW3: named-entity recognition (CoNLL-2003 BIO)
    ├── train.sh / dev.sh / test.sh
    ├── code/
    │   ├── feature_builder.py  #   token → feature-vector extraction (the actual assignment)
    │   └── score.name.py       #   chunk-level P/R/F scorer (course-provided)
    └── maxent/
        ├── MEtrain.java        #   OpenNLP maxent training wrapper (course-provided)
        ├── MEtag.java          #   OpenNLP maxent decoding wrapper (course-provided)
        ├── maxent-3.0.0.jar    #   OpenNLP MaxEnt + Trove dependency
        ├── trove.jar
        ├── *.pos-chunk*        #   token/POS/chunk inputs, and gold tags for train + dev
        └── dist.all.last.txt,  #   gazetteers: surnames, large cities, Brown word freqs
            LargestCity.txt,
            brown-c1000-freq1.txt
```

Generated artifacts (feature files, the trained model, tagger output) are **not**
committed — every one of them is rebuilt by the run commands below in under three
minutes. See [`.gitignore`](.gitignore).

---

## Assignment 1 — Distributional semantics

`assignment1.py` trains a Word2Vec CBOW model (100-dim, window 5) on the NLTK
Brown corpus and compares its nearest neighbours for the target words
`rebellion` and `slave` against Google's pre-trained `word2vec-google-news-300`
vectors. The point of the exercise is the contrast: a 1M-token corpus produces
topically-associated neighbours, while the 100B-token model produces genuinely
substitutable ones.

The written portion (TF/TF-IDF cosine similarity, Naive Bayes with add-1
smoothing) is in `CS510_HW1.pdf`.

```bash
pip install nltk gensim
python3 assignment1.py
```

> Note: the first run downloads the Brown corpus (~3 MB) and the GoogleNews
> vectors (~1.6 GB) into `~/gensim-data/`.

---

## Assignment 2 — HMM POS tagger with Viterbi

A bigram Hidden Markov Model tagger built from raw counts, decoded with Viterbi.

**Implementation notes**

- **Log space throughout.** Initial, transition and emission probabilities are
  stored as logs (`log_pi`, `log_A`, `log_B`) so long sentences can't underflow.
- **Add-α smoothing** (α = 0.1) on all three distributions, with an explicit
  `<UNK>` emission mass per tag for words never seen in training.
- **Tag-set pruning.** Each known word is restricted to the tags it was actually
  observed with (`word_tag`); only unknown words consider the full tag set. This
  is what keeps decoding to ~5 s over the full corpus instead of |T|² per token.
- The final test predictions are produced by retraining on train + dev
  concatenated, which is why `main.py` trains twice.

The report traces the accuracy climb from a 22.73% baseline to 94.58% through
smoothing, unknown-word handling and pruning.

```bash
cd HMM_and_Viterbi
python3 main.py                                    # writes data/my_dev.pos and my_test.pos
python3 scorer.py data/POS_dev.pos data/my_dev.pos
```

```
31072 out of 32853 tags correct
  accuracy: 94.578882
```

---

## Assignment 3 — MEMM named-entity recognition

A Maximum Entropy Markov Model over the CoNLL-2003 BIO tag set (`PER`, `ORG`,
`LOC`, `MISC` × `B-`/`I-`, plus `O`). The pipeline is:

```
 .pos-chunk  ──feature_builder.py──▶  .feat  ──MEtrain──▶  NER-MEMM.model
                                        │
                                        └────MEtag──────▶  predictions ──score.name.py──▶ P/R/F
```

The maxent classifier itself is Apache OpenNLP; **the assignment is the feature
function**, which lives in `code/feature_builder.py`:

- *Orthographic* — lowercased form, initial-cap, all-caps, has-digit,
  has-hyphen, and a word-shape signature (`Xxxx`, `dd-dd`, …).
- *Morphological* — 1/2/3-character prefixes and suffixes.
- *Syntactic* — the token's POS and chunk tag, the previous POS, and the POS
  bigram.
- *Contextual* — previous and next word, and the previous **predicted** tag
  (`PREV_TAG`), which is the "Markov" half of the MEMM.
- *Gazetteer* — membership in a US-census surname list, a large-cities list, and
  the Brown top-1000 frequency list (plus the raw Brown frequency as a feature),
  and an `Inc`/`Ltd`/`Corp`/`University` organisation-suffix flag.

```bash
cd NER_MEMM
./train.sh   # builds train.feat, then trains maxent/NER-MEMM.model (100 GIS iterations)
./dev.sh     # tags the dev set and scores it
./test.sh    # tags the blind test set
```

`./dev.sh` prints:

```
5917 groups in key
5922 groups in response
5012 correct groups
  precision: 84.63
  recall:    84.71
  F1:        84.67
```

Requires a JRE (tested on OpenJDK 21) — the two `.class` files are committed so
no JDK is needed. To rebuild them:

```bash
cd NER_MEMM/maxent && javac -cp maxent-3.0.0.jar:trove.jar MEtrain.java MEtag.java
```

---

## Reproducing the results

Verified on Linux with Python 3 and OpenJDK 21 (July 2026):

| Check | Command | Result |
|-------|---------|--------|
| HMM dev accuracy | `cd HMM_and_Viterbi && python3 main.py` | 94.578882% — byte-identical output across runs |
| MEMM dev F1 | `cd NER_MEMM && ./train.sh && ./dev.sh` | P 84.63 / R 84.71 / **F1 84.67** |
| MEMM dev token accuracy | — | 97.23% |

One caveat worth stating plainly: `CS_510_HW2.pdf` reports **84.79** F1 / 97.24%
token accuracy. That run included two features — the raw `WORD=` identity and a
binned `LENGTH_BIN=` — that are commented out in the current
`feature_builder.py`. Uncommenting them reproduces the report's configuration;
the checked-in code is the slightly leaner variant at 84.67.

---

## Data provenance

Everything under `HMM_and_Viterbi/data/` and `NER_MEMM/maxent/*.pos-chunk*`,
along with the gazetteers, the OpenNLP jars, and both scorers, was distributed
with the course. The `MEtrain.java` / `MEtag.java` wrappers are adapted from
Ralph Grishman's NLP course materials. Original work in this repository:
`assignment1.py`, `HMM_and_Viterbi/main.py`, `NER_MEMM/code/feature_builder.py`,
the shell drivers, and the three PDF write-ups.
