## Model Architecture

- Predict `slot label` from **one BERT model** 
- total_loss = slot_loss 

## Dependencies

- python>=3.6
- torch==1.6.0
- transformers==3.0.2
- seqeval==0.0.12
- pytorch-crf==0.7.2

## Dataset

|         | Train  | Dev | Test |  Slot (NER) Labels |
|  -----  | ------ | --- | ---- |  ----------- |
|  ATIS   | 4,478  | 500 | 893  |  120         |
| MEDICAL | 6,832  | 929 | 755  |  11          |

- The number of labels are based on the _train_ dataset.

## Training & Evaluation

```bash

############################################################
# For ATIS slot filling, which is essentially a NER task
############################################################


# DEFAULT VERSION

python bert_finetune_ner/main.py --data_dir ./bert_finetune_ner/data/ --task atis --model_type bert --model_dir bert_finetune_ner/experiments/outputs/nerbert_0 --do_train --do_eval --train_batch_size 8 --num_train_epochs 8 --learning_rate 5e-5 --warmup_steps 600 --ignore_index -100

# on test
sementic_frame_acc = 0.6103023516237402
slot_f1 = 0.8044086773967809

python bert_finetune_ner/main.py --data_dir ./bert_finetune_ner/data/ --task atis --model_type bert --model_dir bert_finetune_ner/experiments/outputs/nerbert_1 --do_train --do_eval --train_batch_size 8 --num_train_epochs 8 --use_crf --learning_rate 5e-5 --warmup_steps 600 --ignore_index -100

# on test
sementic_frame_acc = 0.6226203807390818
slot_f1 = 0.8127837932238909


```




```bash

# Ablation studies on hyper-params

# 1. set higher learning rate for linear layer:
python bert_finetune_ner/main.py --data_dir ./bert_finetune_ner/data/ --task atis --model_type bert --model_dir bert_finetune_ner/experiments/outputs/nerbert_3 --do_train --do_eval --train_batch_size 8 --num_train_epochs 8 --learning_rate 5e-5 --linear_learning_rate 5e-4 --warmup_steps 600 --ignore_index -100

# on test
sementic_frame_acc = 0.8073908174692049
slot_f1 = 0.9181532004197271

# 2. set higher learning rate for crf layer:
python bert_finetune_ner/main.py --data_dir ./bert_finetune_ner/data/ --task atis --model_type bert --model_dir bert_finetune_ner/experiments/outputs/nerbert_1 --do_train --do_eval --train_batch_size 8 --num_train_epochs 8 --use_crf --crf_learning_rate 5e-3 --learning_rate 5e-5 --linear_learning_rate 5e-4 --warmup_steps 600 --ignore_index -100

# on test
sementic_frame_acc = 0.8365061590145577
slot_f1 = 0.9351427564328516

@ 2.1. even higher learning rate for crf
python bert_finetune_ner/main.py --data_dir ./bert_finetune_ner/data/ --task atis --model_type bert --model_dir bert_finetune_ner/experiments/outputs/nerbert_2 --do_train --do_eval --train_batch_size 8 --num_train_epochs 8 --use_crf --crf_learning_rate 1e-1 --learning_rate 5e-5 --linear_learning_rate 5e-5 --warmup_steps 600 --ignore_index -100

# on test
sementic_frame_acc = 0.7077267637178052
slot_f1 = 0.8759744861800142


```


## Prediction

```bash
$ python3 predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}

# e.g.
python bert_finetune_ner/predict.py --input_file bert_finetune_ner/data/atis/test/seq.in --output_file bert_finetune_ner/experiments/outputs/nerbert_0/atis_test_predicted.txt --model_dir bert_finetune_ner/experiments/outputs/nerbert_0


```




## Results

- Run 5 ~ 10 epochs (Record the best result)
- Only test with `uncased` model

|           |                  | Intent acc (%) | 
| --------- | ---------------- | -------------- | 
| **ATIS**  | BERT-base        | 97.87          | 
|           | BERT-tiny (2 layers)        | 0.8275          | 


