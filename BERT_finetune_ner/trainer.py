import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

from utils import MODEL_CLASSES, compute_metrics, get_slot_labels

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.slot_label_lst = get_slot_labels(args)  # ner标签。list
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        # 加载模型的config，model本身
        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                                      config=self.config,
                                                      args=args,
                                                      slot_label_lst=self.slot_label_lst)

        # 将模型放到GPU，如果有的话
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        # 加载训练数据
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        # 计算训练的总的更新步数，用于learning rate的schedule (不是迭代步数)
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
                        len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # 看看模型参数
        for n, p in self.model.named_parameters():
            print(n)

        # 准备优化器、学习率调度器（线性层warmup预热和decay）
        optimizer_grouped_parameters = []
        # BERT部分参数，设置一个较低的学习率
        bert_params = list(self.model.bert.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']  # bias和层归一化操作中的参数不做weight decay，否则会导致模型学偏
        optimizer_grouped_parameters += [
            {
                'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
                # 等价于：
                # for n,p in bert_params:
                #     if any(nd in n for nd in no_decay): 判断所有nd是否在n中
                #         print(p)

                'weight_decay': self.args.weight_decay,
                "lr": self.args.learning_rate,
            },
            {
                'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.args.learning_rate,
            }
        ]

        # 线性层参数
        linear_params = list(self.model.slot_classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters += [
            {
                'params': [p for n, p in linear_params if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay,
                "lr": self.args.linear_learning_rate,
            },
            {
                'params': [p for n, p in linear_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.args.linear_learning_rate,
            }
        ]

        # crf层参数
        if self.args.use_crf:
            crf_params = list(self.model.crf.named_parameters())
            no_decay = ['start_transitions', 'end_transitions']
            optimizer_grouped_parameters += [
                {
                    'params': [p for n, p in crf_params if not any(nd in n for nd in no_decay)],
                    'weight_decay': self.args.weight_decay,
                    "lr": self.args.crf_learning_rate,
                },
                {
                    'params': [p for n, p in crf_params if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': self.args.crf_learning_rate,
                }
            ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=t_total)  # lr warmup 参考：https://www.cnblogs.com/douzujun/p/13868472.html

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d",
                    self.args.gradient_accumulation_steps)  # 梯度累加次数，详见：https://www.cnblogs.com/sddai/p/14598018.html
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)  # 计算dev performance
        logger.info("  Save steps = %d", self.args.save_steps)  # 保存model checkpoint

        global_step = 0
        tr_loss = 0.0

        # # 神经网络训练通常的步骤：
        self.model.zero_grad()  # 1.训练前清空梯度

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # 将数据传到设备上面：GPU or CPU

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'slot_labels_ids': batch[3]}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)  # 2.正向传播
                loss = outputs[0]  # 3.计算损失

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()  # 4.反向传播，计算梯度

                tr_loss += loss.item()

                # 设置梯度清空、更新参数的间隔步数，如gradient_accumulation_steps = 3，则每隔3个batch清空一次梯度
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.args.max_grad_norm)  # 梯度裁剪，当梯度过大时裁剪，防止梯度爆炸

                    optimizer.step()  # 5.更新参数
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()  # 6.清空梯度
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        self.evaluate("dev")

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        slot_preds = None
        out_slot_labels_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'slot_labels_ids': batch[3]}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                tmp_eval_loss, slot_logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Slot prediction
            if slot_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(self.model.crf.decode(slot_logits))  # crf解码模块，维特比算法
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()

                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

                out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(),
                                                axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Slot result
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)  # (n, L, NUM_OF_LABELS) --> (n, L, 1)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        total_result = compute_metrics(slot_preds_list, out_slot_label_list)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                          args=self.args,
                                                          slot_label_lst=self.slot_label_lst)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
