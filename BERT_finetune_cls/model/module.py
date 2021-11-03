import torch.nn as nn

# intent分类的MLP全连接层
class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)  # nn.Linear(input神经元数量，output神经元数量)

    def forward(self, x):
        x = self.dropout(x)  # x.size: [batch_size, input_dim]
        return self.linear(x)
