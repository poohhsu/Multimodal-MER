import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation
from transformers import AutoModel


class MyMERT(nn.Module):
    def __init__(self, model_name, n_class=2):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        self.layer_weights = nn.Parameter(torch.ones(13) / 13)
        self.projector = nn.Linear(768, 256)
        self.classifier = nn.Linear(256, n_class)

    def forward(self, freeze=True, **x):
        for k, v in x.items():
            x[k] = x[k].squeeze(1)
        outputs = self.model(**x, output_hidden_states=True)

        hidden_states = torch.stack(outputs.hidden_states, dim=1)
        if freeze:
            hidden_states = hidden_states.detach()
        norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
        hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)

        hidden_states = self.projector(hidden_states)
        pooled_output = hidden_states.mean(dim=1)

        logits = self.classifier(pooled_output)

        return logits


class MyCRNN(nn.Module):
    def __init__(self, n_channels=64, n_class=2):
        super().__init__()
        self.bn = nn.BatchNorm2d(1)

        self.conv = nn.Sequential(
            nn.Conv2d(1, n_channels, 3, padding=1),
            nn.BatchNorm2d(n_channels),
            nn.ELU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Dropout(0.1),
            nn.Conv2d(n_channels, n_channels*2, 3, padding=1),
            nn.BatchNorm2d(n_channels*2),
            nn.ELU(),
            nn.MaxPool2d((4, 2), stride=(4, 2)),
            nn.Dropout(0.1),
            nn.Conv2d(n_channels*2, n_channels*2, 3, padding=1),
            nn.BatchNorm2d(n_channels*2),
            nn.ELU(),
            nn.MaxPool2d((4, 2), stride=(4, 2)),
            nn.Dropout(0.1),
            nn.Conv2d(n_channels*2, n_channels*2, 3, padding=1),
            nn.BatchNorm2d(n_channels*2),
            nn.ELU(),
            nn.MaxPool2d((4, 2), stride=(4, 2)),
            nn.Dropout(0.1),
        )

        self.gru = nn.GRU(n_channels*2, n_channels//2, num_layers=2, batch_first=True, dropout=0.3)

        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_channels//2, n_class),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.bn(x)

        x = self.conv(x)

        x = x.transpose(1, 3)
        x = x.reshape((x.size(0), x.size(1), -1))
        
        x, _ = self.gru(x)
        x = x[:, -1, :]

        x = self.fc(x)

        return x


class MyShortChunkCNN_Res(nn.Module):
    def __init__(self, n_channels=128, n_class=2):
        super().__init__()
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn = nn.BatchNorm2d(1)

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, n_channels, 3, padding=1),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            nn.BatchNorm2d(n_channels),
        )
        self.layer1d = nn.Sequential(
            nn.Conv2d(1, n_channels, 1),
            nn.BatchNorm2d(n_channels),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            nn.BatchNorm2d(n_channels),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels*2, 3, padding=1),
            nn.BatchNorm2d(n_channels*2),
            nn.ReLU(),
            nn.Conv2d(n_channels*2, n_channels*2, 3, padding=1),
            nn.BatchNorm2d(n_channels*2),
        )
        self.layer3d = nn.Sequential(
            nn.Conv2d(n_channels, n_channels*2, 1),
            nn.BatchNorm2d(n_channels*2),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(n_channels*2, n_channels*2, 3, padding=1),
            nn.BatchNorm2d(n_channels*2),
            nn.ReLU(),
            nn.Conv2d(n_channels*2, n_channels*2, 3, padding=1),
            nn.BatchNorm2d(n_channels*2),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(n_channels*2, n_channels*2, 3, padding=1),
            nn.BatchNorm2d(n_channels*2),
            nn.ReLU(),
            nn.Conv2d(n_channels*2, n_channels*2, 3, padding=1),
            nn.BatchNorm2d(n_channels*2),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(n_channels*2, n_channels*2, 3, padding=1),
            nn.BatchNorm2d(n_channels*2),
            nn.ReLU(),
            nn.Conv2d(n_channels*2, n_channels*2, 3, padding=1),
            nn.BatchNorm2d(n_channels*2),
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(n_channels*2, n_channels*4, 3, padding=1),
            nn.BatchNorm2d(n_channels*4),
            nn.ReLU(),
            nn.Conv2d(n_channels*4, n_channels*4, 3, padding=1),
            nn.BatchNorm2d(n_channels*4),
        )
        self.layer7d = nn.Sequential(
            nn.Conv2d(n_channels*2, n_channels*4, 1),
            nn.BatchNorm2d(n_channels*4),
        )

        self.fc = nn.Sequential(
            nn.Linear(n_channels*4, n_channels*4),
            nn.BatchNorm1d(n_channels*4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(n_channels*4, n_class),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.bn(x)
        
        if self.training:
            x = self.spec_augmenter(x.transpose(2, 3)).transpose(2, 3)

        x = F.max_pool2d(F.relu(self.layer1(x) + self.layer1d(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.layer2(x) + x), kernel_size=2)
        x = F.max_pool2d(F.relu(self.layer3(x) + self.layer3d(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.layer4(x) + x), kernel_size=2)
        x = F.max_pool2d(F.relu(self.layer5(x) + x), kernel_size=2)
        x = F.max_pool2d(F.relu(self.layer6(x) + x), kernel_size=2)
        x = F.max_pool2d(F.relu(self.layer7(x) + self.layer7d(x)), kernel_size=2)

        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = x.squeeze((2, 3))

        x = self.fc(x)

        return x


class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class CRNN(nn.Module):
    '''
    Choi et al. 2017
    Convolution recurrent neural networks for music classification.
    Feature extraction with CNN + temporal summary with RNN
    '''
    def __init__(self, n_channels=64, n_class=2):
        super(CRNN, self).__init__()

        # Spectrogram
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Conv_2d(1, n_channels, pooling=(2,2))
        self.layer2 = Conv_2d(n_channels, n_channels*2, pooling=(3,3))
        self.layer3 = Conv_2d(n_channels*2, n_channels*2, pooling=(4,4))
        self.layer4 = Conv_2d(n_channels*2, n_channels*2, pooling=(4,4))

        # RNN
        self.layer5 = nn.GRU(n_channels*2, n_channels//2, 2, batch_first=True)

        # Dense
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(n_channels//2, n_class)

    def forward(self, x):
        # Spectrogram
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # RNN
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.layer5(x)
        x = x[:, -1, :]

        # Dense
        x = self.dropout(x)
        x = self.dense(x)

        return x


class Res_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=2):
        super(Res_2d, self).__init__()
        # convolution
        self.conv_1 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, shape, padding=shape//2)
        self.bn_2 = nn.BatchNorm2d(output_channels)

        # residual
        self.diff = False
        if (stride != 1) or (input_channels != output_channels):
            self.conv_3 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True
        self.relu = nn.ReLU()

    def forward(self, x):
        # convolution
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))

        # residual
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.relu(out)
        return out


class ShortChunkCNN_Res(nn.Module):
    '''
    Short-chunk CNN architecture with residual connections.
    '''
    def __init__(self, n_channels=128, n_class=2):
        super(ShortChunkCNN_Res, self).__init__()

        # Spectrogram
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Res_2d(1, n_channels, stride=2)
        self.layer2 = Res_2d(n_channels, n_channels, stride=2)
        self.layer3 = Res_2d(n_channels, n_channels*2, stride=2)
        self.layer4 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer5 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer6 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer7 = Res_2d(n_channels*2, n_channels*4, stride=2)

        # Dense
        self.dense1 = nn.Linear(n_channels*4, n_channels*4)
        self.bn = nn.BatchNorm1d(n_channels*4)
        self.dense2 = nn.Linear(n_channels*4, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Spectrogram
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)

        return x