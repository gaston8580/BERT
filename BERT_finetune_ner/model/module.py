import torch.nn as nn

# ner识别的MLP全连接层
class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)  # nn.Linear(input神经元数量，output神经元数量)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
