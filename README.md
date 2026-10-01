# SHL Kaggle Competition: Grammar Scoring from Speech

Each clip is a person speaking for about a minute. The task is to predict their grammar score (1 to 5), as given by human raters. Submissions are scored by RMSE.

**Result:** public leaderboard RMSE **0.3492** (cross-validation RMSE 0.478).

## Approach

Two notebooks:

1. [`notebooks/1_feature_extraction.ipynb`](notebooks/1_feature_extraction.ipynb) (GPU, ~1 h) turns every clip into features.
2. [`notebooks/2_model.ipynb`](notebooks/2_model.ipynb) (CPU, ~1 min) trains five small models on those features and blends them.

### 1. Features

From the **audio**:
- **WavLM-large** embeddings (a speech model), averaged over time, for every layer
- **Whisper-large-v3 encoder** embeddings, mean and standard deviation over time, for every layer
- Speech timing: how much of the clip is speech, pauses per minute, speaking rate

From the **transcript**:
- Whisper transcript. Whisper tends to quietly fix grammar mistakes, so we prompt it with a disfluent, ungrammatical sentence to keep the speaker's errors. A few transcripts that came out broken (prompt copied in, loops) are redone without the prompt.
- Fillers ("um", "uh") and exact repeats are removed, since the raters score grammar, not hesitation.
- **RoBERTa-large** and **Qwen2.5-1.5B** embeddings of the cleaned transcript
- Handcrafted features: sentence length and structure (spaCy), grammar errors (LanguageTool), how "surprising" the text is to GPT-2, a grammaticality classifier (RoBERTa trained on CoLA), and a second, letter-by-letter transcript from wav2vec2 that cannot correct grammar

### 2. Model

| Model | Input |
|---|---|
| SVR | WavLM, layer 21 |
| SVR | Whisper encoder, layer 31 |
| Ridge | RoBERTa, layer 19 |
| Ridge | Qwen, layer 17 |
| Ridge | handcrafted features |

The layers were picked by cross-validation. The five predictions are combined with a linear regression whose weights must be non-negative. The two audio models carry about two thirds of the total weight.

The 37 training clips labelled 0 are pure noise and are left out of training. The test set has none of them.

## Maths and statistics

**Metric.** RMSE = √(mean of (predicted − true)²). Always predicting the average score gives RMSE 1.014 (the standard deviation of the labels), so that is the baseline to beat.

**Validation.** 5-fold cross-validation, stratified by score level so every fold has the same mix of scores, repeated with 3 different shuffles. Every training clip gets an out-of-fold prediction from models that never saw it, and these are averaged over the 3 repeats. The test prediction is the average of all 15 fitted models.

**Choosing layers.** Each embedding model has 25 to 33 layers. For each layer we fitted a Ridge model and kept the layer with the lowest cross-validated RMSE. Picking the best layer on the same data makes its score slightly optimistic (about 0.01).

**Models.**
- **SVR** (support vector regression) with an RBF kernel, on standardised features: C = 10, ε = 0.1. It ignores errors smaller than ε and can fit curved relationships.
- **Ridge**: linear regression with an L2 penalty. The penalty strength α is chosen from 10⁻¹ to 10⁶ by leave-one-out cross-validation.

**Blending.** Non-negative least squares on the five out-of-fold predictions: score = Σ wᵢ · predictionᵢ + b, with every wᵢ ≥ 0. The blend's own score comes from a separate 5-fold split, so it is also judged on clips it did not see.

**Feature checks.** Spearman correlation of each handcrafted feature with the score. Strongest: grammaticality from the CoLA classifier (ρ = 0.50), vocabulary variety (0.41), and disagreement between the Whisper and wav2vec2 transcripts (−0.42).

**Train vs test.** A classifier trained to tell training clips from test clips does so with AUC 0.84 on audio embeddings and 0.78 on text embeddings, so the test set clearly differs from the training set. By resampling the training predictions, we estimated that with 216 test clips, the leaderboard gap between two similar versions can swing by about ±0.01 by chance. Because of both, cross-validation alone could not choose between versions that were close, so the final choice was made on the Kaggle score.

## Experiments

**1. Grammar error correction (GEC).** Our own idea: if the raters score grammar, count how many fixes the transcript needs. Each transcript was corrected by two models (CoEdIT, from Grammarly, and the LLM Qwen2.5-7B), and ERRANT counted the edits by type (verb tense, articles, prepositions…), ignoring spelling and punctuation. The edit rate correlates with the score (ρ ≈ −0.54), and on its own it predicts the score with RMSE 0.81. But on top of the other models it improved cross-validation by only 0.001, so we dropped it.

**2. Transcript only.** Since only grammar is scored, we tried using nothing but the transcript. Cross-validation RMSE got worse: 0.60, against 0.48 with the audio. Speech recognition quietly fixes some of the speaker's mistakes, while the audio models still hear what was actually said, so we kept the audio.

**3. Whisper encoder and extra layers.** Adding Whisper encoder (and Qwen) embeddings next to WavLM and RoBERTa brought the Kaggle score to 0.3492, our best. Adding a second, middle layer of each audio model looked even better in cross-validation (0.4695), but scored 0.3653 on Kaggle. A simpler 3-model average also looked fine in cross-validation but scored 0.3563. Based on the Kaggle scores, we kept the five-model version.

## How to run

1. Run `1_feature_extraction.ipynb` on Kaggle with GPU T4 and internet on. Attach the competition data (and optionally a `clips.csv` from an earlier run to reuse the Whisper transcripts, which saves about 1.5 h).
2. Save its output as a Kaggle dataset.
3. Run `2_model.ipynb` with the competition data and that dataset attached. It writes `submission.csv`.
