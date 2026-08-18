# Maritime RE Baselines

This folder contains BERT/RoBERTa baselines for the MarRED dataset.

There are now two different training styles:

1. Document-batch baseline: one DataLoader sample is one document; all visible entity pairs are predicted inside the document.
2. Entity-pair baseline: one DataLoader sample is one head-tail pair. This is only a weak sentence/entity-pair classifier baseline.


## Recommended: Document-Batch Baseline

Default document-level RoBERTa:

```powershell
python F:\海事数据集构建\baseline_experiments\train_document_classifier_hydra.py
```

Select document-level BERT:

```powershell
python F:\海事数据集构建\baseline_experiments\train_document_classifier_hydra.py --config-name doc_bert
```

Select document-level RoBERTa explicitly:

```powershell
python F:\海事数据集构建\baseline_experiments\train_document_classifier_hydra.py --config-name doc_roberta
```

Hydra overrides also work:

```powershell
python F:\海事数据集构建\baseline_experiments\train_document_classifier_hydra.py --config-name doc_roberta train.epochs=5 train.device=cuda:1
```


## Weak Entity-Pair/Sentence Baseline

Use this only when you want a weak baseline that expands documents into entity-pair samples.

```powershell
python F:\海事数据集构建\baseline_experiments\train_pair_classifier_hydra.py --config-name sent_roberta
```

```powershell
python F:\海事数据集构建\baseline_experiments\train_pair_classifier_hydra.py --config-name sent_bert
```

## Configs

Available configs:

- `baseline_experiments/configs/doc_roberta.yaml`: document-batch RoBERTa
- `baseline_experiments/configs/doc_bert.yaml`: document-batch BERT
- `baseline_experiments/configs/sent_roberta.yaml`: sentence/entity-pair RoBERTa
- `baseline_experiments/configs/sent_bert.yaml`: sentence/entity-pair BERT

The RoBERTa configs use the local model directory:

```text
F:\海事数据集构建\baseline_experiments\roberta-base
```

The BERT configs currently use `bert-base-uncased`. If the machine cannot access HuggingFace, download BERT locally and replace `model.model_name_or_path` with that local path.

## Progress Logs

The training scripts print:

- PID, config, model path, data paths, output path
- train/dev/test document counts and relation distributions
- document-batch steps per epoch
- train tqdm for each epoch
- step-level loss every `train.log_steps` steps
- dev evaluation after every epoch
- final dev/test metrics and output path

Set the frequency in the YAML file:

```yaml
train:
  log_steps: 100
```

## Output

Each run writes to the `output.output_dir` configured in the YAML file:

- `best_model.pt`
- `effective_args.json`
- `dev_predictions.json`
- `test_predictions.json`
- `dev_metrics.json`
- `test_metrics.json`

Use `test_metrics.json` for the main result table and `per_relation` inside it for relation-wise analysis.
