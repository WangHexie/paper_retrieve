import torch
from torch import optim
from torch.optim import lr_scheduler

from src.config.configs import default_train_config
from src.data.read_data import load_file_or_model
from src.models.datasets import TripletText
from src.models.losses import MyOnlineTripletLoss
from src.models.networks import EmbeddingNet, TripletNet
from src.models.trainer import fit

triplet_train_dataset = TripletText(default_train_config.batch_size,
                                    default_train_config.sample_number,
                                    random=default_train_config.random,
                                    hard=default_train_config.hard,
                                    max_len=default_train_config.max_len,
                                    use_idf=default_train_config.use_idf,
                                    use_self_train=default_train_config.use_self_train)

triplet_train_dataset.shuffle()
# Set up the network and training parameters

device = torch.device('cuda')

margin = 1.
embedding_net = EmbeddingNet(default_train_config.embedding_size)
embedding_net.to(device)

# model = TripletNet(embedding_net)
model = load_file_or_model("modelhardest2_abs_loss_idf4.pk")

model.to(device)
loss_fn = MyOnlineTripletLoss(margin, default_train_config.sample_number, absolute=default_train_config.absolute,
                              soft_margin=default_train_config.soft_margin)
lr = 5e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 2, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 10

fit(triplet_train_dataset, model, loss_fn, optimizer, scheduler, n_epochs, log_interval,
    name="hardest2_abs_loss_idf", start_epoch=6)

# fit(TripletText(16, 4), TripletNet(EmbeddingNet(300)), MyOnlineTripletLoss(1, 4), )
