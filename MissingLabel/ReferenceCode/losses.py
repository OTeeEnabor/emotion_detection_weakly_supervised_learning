from typing_extensions import final
import torch
from torch._C import ThroughputBenchmark
import torch.nn.functional as F
import math

'''
loss functions
'''


def loss_an(logits, observed_labels, P):
    # take in logits, observed labels, p
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_matrix = F.binary_cross_entropy_with_logits(logits, observed_labels, reduction='none') # calculate BCE loss matrix
    # calculate corrected binary cross entropy loss
    # what does torch logical_not(observed_labels).float() do?
    # returns the binary opposite of the observed labels
    # e.g. observed_labels = [1,0,0,0,0,0,0,0,0,0,0,0] - logical not becomes -> [0,1,1,1,1,1,1,1,1,1,1]
    # here, their 0s could be false negatives and turn it to 1 would be the true positive
    # with us - we have false positives 1s and false negatives 0 -
    # we turn high loss 1s to 0s - false positive to true negatives
    # we turn high loss 0s to 1s - false negatives to true positives
    corrected_loss_matrix = F.binary_cross_entropy_with_logits(logits, torch.logical_not(observed_labels).float(),
                                                               reduction='none')
    return loss_matrix, corrected_loss_matrix


'''
top-level wrapper
'''


def compute_batch_loss(preds, label_vec, P):  # "preds" are actually logits (not sigmoid activated !)

    assert preds.dim() == 2
    # get batch size from the first dimension of pred matrix - i.e. no of rows
    batch_size = int(preds.size(0))
    # num of classes in multi-label problem from the number of columns
    num_classes = int(preds.size(1))
    # get all label_vec == 0
    unobserved_mask = (label_vec == 0)

    # compute loss for each image and class:
    # label_vec.clip(0) - get all labels that are set to 0
    loss_matrix, corrected_loss_matrix = loss_an(preds, label_vec.clip(0), P)

    correction_idx = None

    if P['clean_rate'] == 1:  # if epoch is 1, do not modify losses
        final_loss_matrix = loss_matrix
    else:
        if P['mod_scheme'] is 'LL-Cp':
            # round the product of batch size * num labels * delta rel to upward to its nearest integer
            k = math.ceil(batch_size * num_classes * P['delta_rel'])
        else:
            # # round the product of batch_size * num labels * 1 - clean rate to beares integear
            k = math.ceil(batch_size * num_classes * (1 - P['clean_rate']))
        # unobserved loss = multiply loss matrix by loss_matrix
        unobserved_loss = unobserved_mask.bool() * loss_matrix

        # get the k largest elements of the given input tensor along a given dimension
        topk = torch.topk(unobserved_loss.flatten(), k)
        # get the largest element in topk
        topk_lossvalue = topk.values[-1]
        # return a tensor of elements where where the unobserved loss is greater than largest element in topk
        correction_idx = torch.where(unobserved_loss > topk_lossvalue)
        #
        if P['mod_scheme'] in ['LL-Ct', 'LL-Cp']:
            # if LL-Ct and LL-Cp
            # return a tensor of elements from -- loss matrix -- if - unobserved_loss < topk_lossvalue true
            # return a tensor of elements from -- corrected loss matrix -- if - unobserved_loss < topk_lossvalue false
            final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue, loss_matrix, corrected_loss_matrix)
        else:
            # if not LL-CT or LL-Cp
            #
            zero_loss_matrix = torch.zeros_like(loss_matrix)
            # return a tensor of elements from -- loss matrix -- if - unobserved_loss < topk_lossvalue true
            # return a tensor of elements from -- ero loss matrix -- if - unobserved_loss < topk_lossvalue false
            final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue, loss_matrix, zero_loss_matrix)

    main_loss = final_loss_matrix.mean()

    return main_loss, correction_idx