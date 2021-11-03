import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF
from .module import IntentClassifier

class ClsBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst):
        super().__init__(config)  # 声明继承父类的属性
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.bert = BertModel(config=config)  # 加载bert预训练模型
        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)  # 分类MLP层

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # 输出包含：sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]  # 输出句向量, size:[32, 250, 768], 32:batch size, 250:最大句子长度, 768:字向量维度
        pooled_output = outputs[1]  # [CLS]向量, size:[32, 768], 32:batch size, 768:字向量维度
        intent_logits = self.intent_classifier(pooled_output)  # MLP层分类结果，size:[32, 3], 32:batch size, 3:标签类别数量
        outputs = ((intent_logits),) + outputs[2:]  # add hidden states and attention if they are here

        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))

            outputs = (intent_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
