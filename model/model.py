import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class MyDNN(BaseModel):
    def __init__(self, feature_num, dropout=0.1):
        super().__init__()
        self.feature_num = feature_num

        self.prediction_layer = nn.Sequential(
            nn.Linear(in_features=self.feature_num,
                      out_features=int(self.feature_num/8)),
            nn.BatchNorm1d(num_features=int(self.feature_num/8)),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(in_features=int(self.feature_num/8),
                      out_features=int(self.feature_num/8)),
            nn.BatchNorm1d(num_features=int(self.feature_num/8)),
            nn.Dropout(p=dropout),
            nn.ReLU(),

            nn.Linear(in_features=int(self.feature_num/8),
                      out_features=1))
            # nn.Sigmoid()
            # nn.Linear(in_features=int(self.feature_num/8),
            #           out_features=1))

    def forward(self, x):
        output = self.prediction_layer(x)
        output = torch.squeeze(output)
        return output

class PCADNN(BaseModel):
    def __init__(self, feature_num, dropout=0.1):
        super().__init__()
        self.feature_num = feature_num

        self.prediction_layer = nn.Sequential(
            nn.Linear(in_features=self.feature_num,
                      out_features=int(self.feature_num*2)),
            nn.BatchNorm1d(num_features=int(self.feature_num*2)),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(in_features=int(self.feature_num*2),
                      out_features=int(self.feature_num)),
            nn.BatchNorm1d(num_features=int(self.feature_num)),
            nn.Dropout(p=dropout),
            nn.ReLU(),

            nn.Linear(in_features=int(self.feature_num),
                      out_features=1))
            # nn.Sigmoid()
            # nn.Linear(in_features=int(self.feature_num/8),
            #           out_features=1))

    def forward(self, x):
        output = self.prediction_layer(x)
        output = torch.squeeze(output)
        return output