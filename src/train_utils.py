import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau


def _recompute_sample_weights(iterator, model):
    # add cuda here
    x = Variable(iterator.dataset.data_tensor, volatile=True)
    y = Variable(iterator.dataset.target_tensor, volatile=True)
    logits = model(x)
    predictions = torch.gather(F.softmax(logits), 1, y.view(-1, 1)).cpu()
    iterator.dataset.sample_weights_tensor = predictions.data


def _optimization_step(model, criterion, optimizer, x_batch, y_batch, w_batch):
    # add cuda here
    x_batch = Variable(x_batch)
    y_batch = Variable(y_batch)
    w_batch = Variable(w_batch)
    logits = model(x_batch)

    # compute logloss and weighted logloss
    loss, weighted_loss = criterion(logits, y_batch, w_batch)
    batch_loss = loss.data[0]
    batch_weighted_loss = weighted_loss.data[0]

    # compute accuracy
    pred = F.softmax(logits)
    batch_accuracy = _accuracy(y_batch, pred)

    # compute gradients
    optimizer.zero_grad()
    weighted_loss.backward()

    # update params
    optimizer.step()

    return batch_loss, batch_weighted_loss, batch_accuracy


def train(model, criterion, optimizer,
          train_iterator, n_epochs, steps_per_epoch,
          val_iterator, n_validation_batches,
          reweight_epoch,
          patience=10, threshold=0.01, lr_scheduler=None):

    # collect losses and accuracies here
    all_losses = []

    is_reduce_on_plateau = isinstance(lr_scheduler, ReduceLROnPlateau)

    running_loss = 0.0
    running_weighted_loss = 0.0
    running_accuracy = 0.0
    start_time = time.time()
    model.train()

    for epoch in range(0, n_epochs):

        # main training loop
        step_iterator = enumerate(train_iterator, 1 + epoch*steps_per_epoch)
        for step, (x_batch, y_batch, w_batch) in step_iterator:

            batch_loss, batch_weighted_loss, batch_accuracy = _optimization_step(
                model, criterion, optimizer, x_batch, y_batch, w_batch
            )
            running_loss += batch_loss
            running_weighted_loss += batch_weighted_loss
            running_accuracy += batch_accuracy

        # evaluation
        model.eval()
        val_loss, val_weighted_loss, val_accuracy = _evaluate(
            model, criterion, val_iterator, n_validation_batches
        )

        all_losses += [(
            epoch,
            running_loss/steps_per_epoch, val_loss,
            running_weighted_loss/steps_per_epoch, val_weighted_loss,
            running_accuracy/steps_per_epoch, val_accuracy
        )]
        print('{0}  {1:.3f} {2:.3f}  {3:.3f} {4:.3f}  {5:.3f} {6:.3f}  {7:.3f}'.format(
            *all_losses[-1], time.time() - start_time
        ))

        # it watches test accuracy
        # and if accuracy isn't improving then training stops
        if _is_early_stopping(all_losses, patience, threshold):
            print('early stopping!')
            break

        if lr_scheduler is not None:
            # change learning rate
            if not is_reduce_on_plateau:
                lr_scheduler.step()
            else:
                lr_scheduler.step(val_accuracy)

        if (epoch + 1) % reweight_epoch == 0:
            print('reweighting!')
            # compute new weights for training data
            _recompute_sample_weights(train_iterator, model)
            # compute new weights for val data
            _recompute_sample_weights(val_iterator, model)

        running_loss = 0.0
        running_weighted_loss = 0.0
        running_accuracy = 0.0
        start_time = time.time()
        model.train()

    return all_losses


def _accuracy(true, pred):
    _, argmax = torch.max(pred, dim=1)
    correct = true.eq(argmax)
    return correct.float().mean().data[0]


def _evaluate(model, criterion, val_iterator, n_validation_batches):

    loss = 0.0
    weighted_loss = 0.0
    accuracy = 0.0
    total_samples = 0

    for j, (x_batch, y_batch, w_batch) in enumerate(val_iterator):
        # add cuda here
        x_batch = Variable(x_batch, volatile=True)
        y_batch = Variable(y_batch, volatile=True)
        w_batch = Variable(w_batch, volatile=True)
        n_batch_samples = y_batch.size()[0]
        logits = model(x_batch)

        # compute logloss and weighted logloss
        loss_var, weighted_loss_var = criterion(logits, y_batch, w_batch)
        batch_loss = loss_var.data[0]
        batch_weighted_loss = weighted_loss_var.data[0]

        # compute accuracy
        pred = F.softmax(logits)
        batch_accuracy = _accuracy(y_batch, pred)

        loss += batch_loss*n_batch_samples
        weighted_loss += batch_weighted_loss*n_batch_samples
        accuracy += batch_accuracy*n_batch_samples
        total_samples += n_batch_samples

        if j >= n_validation_batches:
            break

    return loss/total_samples, weighted_loss/total_samples, accuracy/total_samples


# it decides if training must stop
def _is_early_stopping(all_losses, patience, threshold):

    # get current and all past (validation) accuracies
    accuracies = [x[6] for x in all_losses]

    if len(all_losses) > (patience + 4):

        # running average with window size 5
        average = (accuracies[-(patience + 4)] +
                   accuracies[-(patience + 3)] +
                   accuracies[-(patience + 2)] +
                   accuracies[-(patience + 1)] +
                   accuracies[-patience])/5.0

        # compare current accuracy with
        # running average accuracy 'patience' epochs ago
        return accuracies[-1] < (average + threshold)
    else:
        # if not enough epochs to compare with
        return False
