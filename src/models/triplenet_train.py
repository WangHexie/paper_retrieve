import torch
from torch import optim
from torch.optim import lr_scheduler

from src.data.read_data import load_file_or_model
from src.models.datasets import TripletText
from src.models.losses import TripletLoss, MyOnlineTripletLoss
from src.models.networks import EmbeddingNet, TripletNet
from src.models.trainer import fit


batch_size = 16
sample_number = 32
triplet_train_dataset = TripletText(batch_size, sample_number, random=False, hard=-1, max_len=100)

# Set up the network and training parameters

device = torch.device('cuda')

margin = 1.
embedding_net = EmbeddingNet(300)
embedding_net.to(device)

model = TripletNet(embedding_net)
# model = load_file_or_model("modelhardest2_abs_loss_hign_learning_100_length.pk")

model.to(device)
loss_fn = MyOnlineTripletLoss(margin, sample_number, absolute=True)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 20

fit(triplet_train_dataset, model, loss_fn, optimizer, scheduler, n_epochs, log_interval,
    name="hardest2_abs_loss_hign_learning_rate", start_epoch=2)

# fit(TripletText(16, 4), TripletNet(EmbeddingNet(300)), MyOnlineTripletLoss(1, 4), )
