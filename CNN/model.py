import torch
import torch.nn as nn

class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        self.embed = nn.Embedding(args.embed_num, args.embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, args.kernel_num, (size, args.embed_dim)) for size in args.kernel_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(args.kernel_sizes) * args.kernel_num, args.class_num)

    def forward(self, x):
        embedded = self.embed(x).unsqueeze(1)
        convoluted = [nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        convoluted_processed = self.dropout(torch.cat([nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in convoluted], 1))
        return self.fc(convoluted_processed)

