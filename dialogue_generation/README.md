
# Persona-Grounded Dialogue Modeling

This folder contains the source code of our experiments in **PeaCoK Section 7: Enhancing Dialogue Systems**.
The code of our baseline dialogue system is modified from the [P^2Bot repository](https://github.com/SivilTaram/Persona-Dialogue-Generation).

## Development Environment

- OS: Ubuntu 20.04 LTS (64bit)
- GPU: Nvidia Titan X
- CUDA 11.6
- Python 3.7

## Requirements

Please follow the **Install Dependencies** in [P^2Bot repository](https://github.com/SivilTaram/Persona-Dialogue-Generation) to set up the environment.

## Knowledge Linking

Following the [ComFact](https://github.com/Silin159/ComFact) benchmark, we train a DeBERTa (large) **entity** linker to retrieve PeaCoK facts that are relevant to each [PersonaChat](https://arxiv.org/abs/1801.07243) dialogues.
Our developed DeBERTa **entity** linker trained on ComFact data can be downloaded from [this link](https://drive.google.com/file/d/1GHa3N7AbHLQSnhIsywmRLn003mN5NLTo/view?usp=sharing).
Our preprocessed [ConvAI2 PersonaChat](https://arxiv.org/pdf/1902.00098v1.pdf) dataset with linked relevant PeaCoK facts can be downloaded from [this link](https://drive.google.com/file/d/158NluWqUSVMEUUjKh1CxuG5IKAK8VO9r/view?usp=sharing).

## Model Training

Our final preprocessed data for P^2Bot dialogue model training can be downloaded from [this link](https://drive.google.com/file/d/1kdE5aW9o2Lz58CaD2uLnTuKdzsLcxwPQ/view?usp=sharing), please place `data_peacok_p2bot_convai2.zip` under this directory and unzip the file.
The P^2Bot dialogue model needs three-step training, where we use different training data augmented by PeaCoK, see `data/README.md` in the unzipped file for more details.

### Step 1: Transmitter Training

Copy the `ConvAI2` data folder in `data/transmitter` and paste it under `data/`, then run:
```
python train_transmitter.py
```

### Step 2: Receiver Training

Copy the `ConvAI2` data folder in `data/receiver` and paste it (with replacement) under `data/`, then run:
```
python train_receiver.py
```

### Step 3: P^2Bot Training

Copy the `ConvAI2` data folder in `data/psquare` and paste it (with replacement) under `data/`, then run:
```
python train_psquare.py
```

## Evaluation

Copy the `ConvAI2` data folder in `data/transmitter` and paste it (with replacement) under `data/`.

For BLEU and word-level F1 evaluation:
```
python eval_f1.py
```

For HITS@1 evaluation:
```
python eval_hits.py
```

For PPL evaluation, we follow [P^2Bot repository](https://github.com/SivilTaram/Persona-Dialogue-Generation) to run `train_psquare.py` on a trained model to fake the continuation of training.
The restoring will first validate and report PPL on the validation dataset.

## Model Checkpoints

We provide our trained model checkpoints and evaluation samples [here](https://drive.google.com/file/d/1e-PXv-w7ODUA_xuIUBsaqw0mjxuWOLGt/view?usp=sharing).

Please upzip the file to get the `checkpoint` folder:
- `checkpoint/original/` includes models trained on the original ConvAI2 PersonaChat profiles.
- `checkpoint/revised/` includes models trained on the revised ConvAI2 PersonaChat profiles.

Under `checkpoint/original/` or `checkpoint/revised/`:
- `p2bot`: our reproduced P2Bot baseline model
- `p2bot_atomic`: P2Bot model augmented with (Comet-)Atomic2020 knowledge
- `p2bot_peacok`: P2Bot model augmented with PeaCoK persona knowledge
