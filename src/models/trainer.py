import torch
import numpy as np

from src.models.datasets import TripletText
from src.models.losses import MyOnlineTripletLoss
from src.models.networks import TripletNet, EmbeddingNet, save_model


def fit(train_loader, model, loss_fn, optimizer, scheduler, n_epochs, log_interval, metrics=[],
        start_epoch=0, name=""):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        # for metric in metrics:
        #     message += '\t{}: {}'.format(metric.name(), metric.value())
        #
        # val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        # val_loss /= len(val_loader)

        # message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
        #                                                                          val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        save_model(model, name+str(epoch))

        print(message)


def train_epoch(train_loader, model, loss_fn, optimizer, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx in range(int(len(train_loader)/train_loader.batch_size)):
        train_loader.shuffle()

        data = train_loader[batch_idx]
        optimizer.zero_grad()
        outputs = model(*data)

        loss_inputs = outputs

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * train_loader.batch_size, len(train_loader),
                100. * batch_idx * train_loader.batch_size / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics


if __name__ == '__main__':
    fit(TripletText(16, 4), TripletNet(EmbeddingNet(300)), MyOnlineTripletLoss(1, 4), )