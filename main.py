import pandas as pd
import torch.nn as nn
import torch.utils.data as data_util
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import os
from PIL import Image
from transformers import BertTokenizer, BertModel
import pickle
from collections import namedtuple
from torchvision.models import vgg
from torchvision.utils import save_image


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

tokenize = lambda txt: tokenizer(txt,padding='max_length', max_length = 30,
                       truncation=True, return_tensors="pt")


class PokemonDataset(data_util.Dataset):
    def __init__(self, img_folder, pokedex, transforms=None):
        self.img_folder = img_folder
        self.relation_folder = "relation_vecs"
        self.pokedex = pokedex
        self.transforms = transforms

    def __len__(self):
        return self.pokedex.shape[0]

    def __getitem__(self, item):
        name = self.pokedex.iloc[item]["name"]
        pokedex_entry = self.pokedex.iloc[item]["entry"]

        dex = tokenize(pokedex_entry)

        img = Image.open(f"{self.img_folder}/{name}.jpg")
        if(self.transforms is not None):
            img = self.transforms(img)

        relation_vector = pickle.load(open(f"{self.relation_folder}/{name}.txt", "rb"))

        return (pokedex_entry, dex["input_ids"], dex["attention_mask"]), img, torch.tensor(relation_vector, dtype=torch.float32)


class LatentSpaceModel(nn.Module):
    def __init__(self, dropout=0.5):

        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 898)
        self.sig = nn.Sigmoid()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.sig(linear_output)

        return final_layer

class ReconstructImageModel(nn.Module):
    def __init__(self, w=16, h=16):
        super().__init__()
        self.w = w
        self.h = h

        self.matrix_representation = torch.randn(1, 898, w, h, requires_grad=True)
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(898, 200, (4, 4)),
            nn.ReLU(),
            nn.ConvTranspose2d(200, 50, (4, 4), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(50, 10, (4, 4)),
            nn.ReLU(),
            nn.ConvTranspose2d(10, 3, (4, 4), stride=2),
            nn.ReLU(),
        )

    def params(self):
        return [self.matrix_representation]

    def forward(self, x):
        mulled = torch.mul(torch.reshape(x, [x.shape[0], x.shape[1], 1, 1]), self.matrix_representation)
        conv_out = self.conv_block(mulled)

        return conv_out

# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)

LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
vgg_model = vgg.vgg16(pretrained=True)
loss_network = LossNetwork(vgg_model)

crit = nn.MSELoss()

def perceptual_loss(reconstructed, true):
    r = loss_network(reconstructed)
    t = loss_network(true)

    loss = 0
    for r1, t1 in zip(r, t):
        loss += crit(r1, t1)

    return loss


pokedex = pd.read_csv("inverted_dex.csv")
img_folder = "pokemon"

trans = transforms.Compose([
    transforms.Resize(88),
    transforms.CenterCrop(88),
    transforms.ToTensor()
])

alpha = 0.5
beta = 1

data = PokemonDataset(img_folder, pokedex, transforms=trans)
dataset = data_util.DataLoader(data, batch_size=16, shuffle=True)

latent_model = LatentSpaceModel()
reco = ReconstructImageModel()
criterion = nn.MSELoss()
params = list(latent_model.parameters())+list(reco.parameters())+reco.params()
optimizer = optim.Adam(params, lr=1e-6)

new_pkm = "it has a red bulb on its back that is made from magma. It likes to eat at night."

for epoch in range(100):
    for i, ((entry, input_id, mask), img, relation) in enumerate(dataset):
        optimizer.zero_grad()

        latent = latent_model(input_id.squeeze(1), mask)
        rec = reco(latent)

        loss = perceptual_loss(rec, img) * alpha + criterion(latent, relation) * beta
        loss.backward()

        print(f"epoch: {epoch}, {i*16}/{len(data)}, loss: {loss}")

        optimizer.step()

    dex = tokenize(new_pkm)
    input_id = dex["input_ids"]
    mask = dex["attention_mask"]

    latent = latent_model(input_id.squeeze(1), mask)
    rec = reco(latent)
    save_image(rec, f"prediction_{epoch}.png")
    torch.save(latent_model, f"latent_{epoch}.pth")
    torch.save(reco, f"reconstructor_{epoch}.pth")


