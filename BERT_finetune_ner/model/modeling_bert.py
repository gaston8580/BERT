import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF
from .module import SlotClassifier


class NerBERT(BertPreTrainedModel):
    def __init__(self, config, args, slot_label_lst):
        super().__init__(config)
        self.args = args
        self.num_slot_labels = len(slot_label_lst)

        self.bert = BertModel(config=config)  # Load pretrained bert

        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, slot_labels_ids):
        # input_ids: (B, L)
        # attention_mask: (B, L)
        # token_type_ids: (B, L)
        # slot_labels_ids: (B, L)

        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)

        sequence_output = outputs[0]  # [B, L, D] 输出句向量, size:[32, 50, 128], 32:batch size, 50:最大句子长度, 128:词向量维度

        pooled_output = outputs[1]  # [CLS]向量,size:[32, 128], 32:batch size, 128:词向量维度

        slot_logits = self.slot_classifier(sequence_output)  # (B, L, num_slot_labels)
        outputs = (slot_logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Slot Softmax
        if slot_labels_ids is not None:
            # 如果使用crf层，损失函数使用crf层自带的loss，否则使用交叉熵
            if self.args.use_crf:
                slot_loss = self.crf(
                    slot_logits,
                    slot_labels_ids,
                    mask=attention_mask.byte(),
                    reduction='mean'
                )
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)  # default -100
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1  # (B * L, )

                    # slot_logits: (B, L, num_slot_labels) --> (B * L, num_slot_labels)
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]  # (有效长度， num_slot_labels)
                    active_labels = slot_labels_ids.view(-1)[active_loss]  # (有效长度， )
                    slot_loss = slot_loss_fct(active_logits, active_labels)

                else:

                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))

            outputs = (slot_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
