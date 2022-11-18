import datetime
from typing import Tuple

import numpy as np
import sklearn.metrics  # type: ignore
import tensorflow as tf  # type: ignore
import torch
import tqdm  # type: ignore
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import DhSegment
from news_dataset import NewsDataset

EPOCHS = 1
BATCH_SIZE = 32
DATALOADER_WORKER = 4
IN_CHANNELS, OUT_CHANNELS = 3, 10
LEARNING_RATE = 0.01  # 0,0001 seems to work well
LOSS_WEIGHTS = [1.0, 10.0, 10.0, 10.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0]  # 1 and 5 seems to work well

# set random seed for reproducibility
torch.manual_seed(42)


def get_normalization_parameters(data: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    calculate mean and standard deviation
    :param: data: dataloader
    :return: mean and std values for each channel
    """
    batch_means = []
    batch_stds = []

    for batch in data:
        images = batch[0]
        mean = images.mean((0, 2, 3))
        std = images.std((0, 2, 3))

        batch_means.append(mean)
        batch_stds.append(std)

    channel_means = torch.stack(batch_means).mean(0)
    channel_stds = torch.stack(batch_stds).mean(0)
    return channel_means, channel_stds


def train(load_model=None, save_model=None):
    """
    train function. Initializes dataloaders and optimzer.
    :param load_model: (default: None) path to model to load
    :param save_model: (default: None) path to save the model
    :return: None
    """
    # create model
    model = DhSegment([3, 4, 6, 4], in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS, load_resnet_weights=True)

    model = model.float()

    # load model if argument is None it does nothing
    model.load(load_model)

    # load data
    dataset = NewsDataset()

    # splitting with fractions should work according to pytorch doc, but it does not
    train_set, validation_set, _ = dataset.random_split([.9, .05, .05])
    print(f"train size: {len(train_set)}, test size: {len(validation_set)}")

    print(f"ration between classes: {train_set.class_ratio(OUT_CHANNELS)}")

    # set optimizer and loss_fn
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE) # weight_decay=1e-4
    loss_fn = CrossEntropyLoss()  # weight=torch.tensor(LOSS_WEIGHTS)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=DATALOADER_WORKER, drop_last=True)
    val_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=DATALOADER_WORKER)

    means, stds = get_normalization_parameters(train_loader)
    model.means = means
    model.stds = stds

    train_loop(train_loader, len(train_set), model, loss_fn, optimizer, val_loader)

    model.save(save_model)


def train_loop(train_loader: DataLoader, n_train: int, model: torch.nn.Module, loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer, val_loader: DataLoader):
    """
    executes all training epochs. After each epoch a validation round is performed.
    :param train_loader: Dataloader object
    :param n_train: size of train data
    :param model: model to train
    :param loss_fn: loss function to optimize
    :param optimizer: optimizer to use
    :param val_loader: dataloader with validation data
    :return: None
    """

    model.to(DEVICE)
    loss_fn.to(DEVICE)

    step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()

        with tqdm.tqdm(total=(n_train//BATCH_SIZE), desc=f'Epoch {epoch}/{EPOCHS}', unit='batches') as pbar:
            for images, targets in train_loader:
                images = images.to(DEVICE)
                targets = targets.to(DEVICE)

                # Compute prediction and loss
                pred = model(images)
                loss = loss_fn(pred, targets)

                # Backpropagation
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # update description
                pbar.update(1)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # update tensor board logs
                step += 1
                with summary_writer.as_default():
                    tf.summary.scalar('train loss', loss.item(), step=step)

                # delete data from gpu cache
                del images, targets, pred, loss
                torch.cuda.empty_cache()

        validation(val_loader, model, loss_fn, epoch, step)


def validation(val_loader: DataLoader, model, loss_fn, epoch: int, step: int):
    """
    Executes one validation round, containing the evaluation of the current model on the entire validation set.
    :param model: model to validate
    :param loss_fn: loss_fn to validate with
    :param val_loader: dataloader with validation data
    :param epoch: current epoch value for logging
    :param step: current batch related step value for logging. Count of batches that have been loaded.
    :return: None
    """

    model.eval()

    size = len(val_loader)

    loss_sum = 0
    jaccard_sum = 0
    accuracy_sum = 0
    for images, targets in tqdm.tqdm(val_loader, desc='validation_round', total=size):
        # Compute prediction and loss
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        pred = model(images)
        loss = loss_fn(pred, targets)

        pred = pred.detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()

        targets = targets.detach().cpu().numpy()

        loss_sum += loss
        pred = np.argmax(pred, axis=1)
        jaccard_sum += sklearn.metrics.jaccard_score(targets.flatten(), pred.flatten(), average='macro')
        accuracy_sum += sklearn.metrics.accuracy_score(targets.flatten(), pred.flatten())

        del images, targets, pred, loss
        torch.cuda.empty_cache()

    image, target = val_loader.dataset[0]
    image = torch.unsqueeze(image.to(DEVICE), 0)
    pred = model(image).argmax(dim=1).float()

    # update tensor board logs
    with summary_writer.as_default():
        tf.summary.scalar('val loss', loss_sum / size, step=step)
        tf.summary.scalar('val accuracy', accuracy_sum / size, step=step)
        tf.summary.scalar('val jaccard score', jaccard_sum / size, step=step)
        tf.summary.scalar('epoch', epoch, step=step)
        tf.summary.image('val image', torch.transpose(image.cpu(), 3, 1),
                         step=step)
        tf.summary.image('val target', torch.unsqueeze(
            torch.unsqueeze(target.float().cpu() / OUT_CHANNELS, 0), 3), step=step)
        tf.summary.image('val prediction', torch.unsqueeze(pred.float().cpu() / OUT_CHANNELS, 3), step=step)

    print(f"average loss: {loss_sum / size}")
    print(f"average accuracy: {accuracy_sum / size}")
    print(f"average jaccard score: {jaccard_sum / size}")  # Intersection over Union


if __name__ == '__main__':
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {DEVICE} device")

    # setup tensor board
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/runs/' + current_time
    summary_writer = tf.summary.create_file_writer(train_log_dir)

    train(load_model=None, save_model='Models/model.pt')
