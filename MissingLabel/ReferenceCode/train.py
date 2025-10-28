import numpy as np
from sklearn import metrics
import torch
import datasets
import models
from instrumentation import compute_metrics
import losses
import datetime
import os
from tqdm import tqdm

# define run_train function
def run_train(P):
    # load dataset using get_data method which takes in input P? what is P?
    dataset = datasets.get_data(P)
    # create dataloader dictionary
    dataloader = {}
    # for load the train, validation and test datasets
    for phase in ['train', 'val', 'test']:
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase],
            batch_size=P['bsize'],
            shuffle=phase == 'train',
            sampler=None,
            num_workers=P['num_workers'],
            drop_last=False,
            pin_memory=True
        )
    # define ImageClassifier taking in input P. What is P?
    model = models.ImageClassifier(P)
    # extract feature_extractor_params if the param requires_grad (if gradients need to be computed for this Tensor)
    # and store it in list
    feature_extractor_params = [param for param in list(model.feature_extractor.parameters()) if param.requires_grad]
    # get linear classifer params if the param requires Grad True
    linear_classifier_params = [param for param in list(model.linear_classifier.parameters()) if param.requires_grad]
    # create a list of dictionaries that stores feature extractor params and linear classifier params
    # and learning rates
    opt_params = [
        {'params': feature_extractor_params, 'lr': P['lr']},
        {'params': linear_classifier_params, 'lr': P['lr_mult'] * P['lr']}
    ]

    if P['optimizer'] == 'adam':
        # set optimizer as ADAM
        optimizer = torch.optim.Adam(opt_params, lr=P['lr'])

    elif P['optimizer'] == 'sgd':
        # set optimizer as Stochastic Gradient Descent
        optimizer = torch.optim.SGD(opt_params, lr=P['lr'], momentum=0.9, weight_decay=0.001)

    # training loop
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    bestmap_val = 0
    bestmap_test = 0

    for epoch in range(1, P['num_epochs'] + 1):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                y_pred = np.zeros((len(dataset[phase]), P['num_classes']))
                y_true = np.zeros((len(dataset[phase]), P['num_classes']))
                batch_stack = 0
            # with torch set_grad_enabled during training - context manaeger that sets gradient calculation to on or off.
            with torch.set_grad_enabled(phase == 'train'):
                # for every batch loaded from data phase
                for batch in tqdm(dataloader[phase]):
                    # Move data to GPU
                    image = batch['image'].to(device, non_blocking=True)
                    label_vec_obs = batch['label_vec_obs'].to(device, non_blocking=True)
                    label_vec_true = batch['label_vec_true'].clone().numpy()
                    idx = batch['idx']

                    # Forward pass
                    optimizer.zero_grad()
                    # get model predoctions frm image
                    logits = model(image)
                    # if the logits are 1-dimensional
                    if logits.dim() == 1:
                        # if the logits are 1-dimensional then
                        # return a new tensor with a dimension size of 0
                        logits = torch.unsqueeze(logits, 0)
                    # get the predictions of the logits using sigmoid activation function
                    preds = torch.sigmoid(logits)
                    # if we are in training
                    if phase == 'train':
                        # calculate batch loss
                        loss, correction_idx = losses.compute_batch_loss(logits, label_vec_obs, P)
                        # Computes the gradient of current tensor w.r.t. graph leaves.
                        loss.backward()
                        # Performs a single optimization step (parameter update)
                        optimizer.step()

                        if P['mod_scheme'] is 'LL-Cp' and correction_idx is not None:
                            dataset[phase].label_matrix_obs[idx[correction_idx[0].cpu()], correction_idx[1].cpu()] = 1.0

                    else:
                        preds_np = preds.cpu().numpy()
                        this_batch_size = preds_np.shape[0]
                        # indixces
                        y_pred[batch_stack: batch_stack + this_batch_size] = preds_np
                        y_true[batch_stack: batch_stack + this_batch_size] = label_vec_true
                        batch_stack += this_batch_size
        metrics = compute_metrics(y_pred, y_true)
        del y_pred
        del y_true
        map_val = metrics['map']

        P['clean_rate'] -= P['delta_rel']

        print(f"Epoch {epoch} : val mAP {map_val:.3f}")
        if bestmap_val < map_val:
            bestmap_val = map_val
            bestmap_epoch = epoch

            print(f'Saving model weight for best val mAP {bestmap_val:.3f}')
            path = os.path.join(P['save_path'], 'bestmodel.pt')
            torch.save((model.state_dict(), P), path)

        elif bestmap_val - map_val > 3:
            print('Early stopped.')
            break

    # Test phase

    model_state, _ = torch.load(path)
    model.load_state_dict(model_state)

    phase = 'test'
    model.eval()
    y_pred = np.zeros((len(dataset[phase]), P['num_classes']))
    y_true = np.zeros((len(dataset[phase]), P['num_classes']))
    batch_stack = 0
    with torch.set_grad_enabled(phase == 'train'):
        for batch in tqdm(dataloader[phase]):
            # Move data to GPU
            image = batch['image'].to(device, non_blocking=True)
            label_vec_obs = batch['label_vec_obs'].to(device, non_blocking=True)
            label_vec_true = batch['label_vec_true'].clone().numpy()
            idx = batch['idx']

            # Forward pass
            optimizer.zero_grad()

            logits = model(image)

            if logits.dim() == 1:
                logits = torch.unsqueeze(logits, 0)
            preds = torch.sigmoid(logits)

            preds_np = preds.cpu().numpy()
            this_batch_size = preds_np.shape[0]
            y_pred[batch_stack: batch_stack + this_batch_size] = preds_np
            y_true[batch_stack: batch_stack + this_batch_size] = label_vec_true
            batch_stack += this_batch_size

    metrics = compute_metrics(y_pred, y_true)
    map_test = metrics['map']

    print('Training procedure completed!')
    print(f'Test mAP : {map_test:.3f} when trained until epoch {bestmap_epoch}')
