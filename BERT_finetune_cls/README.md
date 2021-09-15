## Model Architecture

- Predict `intent` from **one BERT model** (=Joint model)
- total_loss = intent_loss 

## Dependencies

- python>=3.6
- torch==1.6.0
- transformers==3.0.2
- seqeval==0.0.12
- pytorch-crf==0.7.2

## Dataset

|       | Train  | Dev | Test | Intent Labels | Slot Labels |
| ----- | ------ | --- | ---- | ------------- | ----------- |
| ATIS  | 4,478  | 500 | 893  | 21            | 120         |

- The number of labels are based on the _train_ dataset.
- Add `UNK` for labels (For intent which are only shown in _dev_ and _test_ dataset)

## Training & Evaluation

```bash

# For ATIS intent classification
python bert_finetune_cls/main.py --data_dir bert_finetune_cls/data/ --task atis --model_type bert --model_dir bert_finetune_cls/experiments/outputs/clsbert_0 --do_train --do_eval --train_batch_size 8 --num_train_epochs 2
```


## Prediction

```bash
$ python3 predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}

# e.g.
python bert_finetune_cls/predict.py --input_file bert_finetune_cls/data/atis/test/seq.in --output_file bert_finetune_cls/experiments/outputs/clsbert_0/atis_test_predicted.txt --model_dir bert_finetune_cls/experiments/outputs/clsbert_0


```

## Results

- Run 5 ~ 10 epochs (Record the best result)
- Only test with `uncased` model

|           |                  | Intent acc (%) | 
| --------- | ---------------- | -------------- | 
| **ATIS**  | BERT-base        | 97.87          | 
|           | BERT-tiny (2 layers)        | 0.8275          | 


