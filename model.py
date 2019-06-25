import torch.nn.functional as F
import torch


class ClassifyModel(torch.nn.Module):

    def __init__(self, num_features=316, num_hidden=[316,316], num_classes=6,dropout_prob=0.7):
        super(ClassifyModel, self).__init__()
        num_hidden_1, num_hidden_2 = num_hidden

        self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)        
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_hidden_2)        
        self.linear_out = torch.nn.Linear(num_hidden_2, num_classes)
        self.dropout_prob = dropout_prob
        
    def forward(self, x):
        out = self.linear_1(x) + x
        out = torch.sigmoid(out)
        # out = F.dropout(out, p=self.dropout_prob, training=self.training)
        
        out = self.linear_2(out) + out
        out = torch.sigmoid(out)
        # out = F.dropout(out, p=self.dropout_prob, training=self.training)
        
        logits = self.linear_out(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas
