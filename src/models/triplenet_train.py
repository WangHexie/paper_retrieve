import torch
from torch import optim
from torch.optim import lr_scheduler

from src.models.datasets import TripletText
from src.models.losses import TripletLoss, MyOnlineTripletLoss
from src.models.networks import EmbeddingNet, TripletNet
from src.models.trainer import fit


batch_size = 128
sample_number = 8
triplet_train_dataset = TripletText(16, sample_number)

# Set up the network and training parameters

device = torch.device('cuda')

margin = 1.
embedding_net = EmbeddingNet(300)
embedding_net.to(device)

model = TripletNet(embedding_net)
model.to(device)
loss_fn = MyOnlineTripletLoss(margin, sample_number)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 20

fit(triplet_train_dataset, model, loss_fn, optimizer, scheduler, n_epochs, log_interval)

# fit(TripletText(16, 4), TripletNet(EmbeddingNet(300)), MyOnlineTripletLoss(1, 4), )
