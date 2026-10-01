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

## What we learned

- **Audio matters even though only grammar is scored.** Transcript-only models reached CV RMSE 0.60; adding audio brought it to 0.48. The audio models hear what was actually said, before speech recognition tidies it up.
- **Grammar-correction features did not help.** We corrected each transcript with CoEdIT and an LLM and counted the edits. The counts correlate with the score, but they added nothing on top of the embeddings.
- **More models was not better.** A 13-model stack gave the same predictions as these five. Several changes improved cross-validation slightly but scored worse on the leaderboard, so we kept the simplest version that did well on both.

## How to run

1. Run `1_feature_extraction.ipynb` on Kaggle with GPU T4 and internet on. Attach the competition data (and optionally a `clips.csv` from an earlier run to reuse the Whisper transcripts, which saves about 1.5 h).
2. Save its output as a Kaggle dataset.
3. Run `2_model.ipynb` with the competition data and that dataset attached. It writes `submission.csv`.
