import tensorflow

"""
Define loss functions
"""

def loss_an(logits,noisy_labels,P):
    BCE = tensorflow.keras.losses.BinaryCrossentropy(
        from_logits=True,
        reduction ="none"
    )
    # compute assumed negative loss
    loss_matrix = BCE(logits, noisy_labels)
    corrected_loss_matrix = BCE(logits, tensorflow.math.logical_not(noisy_labels))
    return loss_matrix, corrected_loss_matrix


def compute_batch_loss(preds, label_vec,P):
    batch_size = int(preds.size(0))
    num_classes = int(preds.size(1))

    unobserved_mask = (label_vec == 0)
    #
    loss_matrix, corrected_loss_matrix = loss_an(preds, label_vec.clip(0))

    correction_indices = None

    if P['clean_rate'] == 1:
        final_loss_matrix = loss_matrix
    else:
        if P['mod_scheme'] is 'LL-Cp':
            k = tensorflow.math.ceil(batch_size * num_classes * P['delta_rel'])
        else:
            k = tensorflow.math.ceil(batch_size * num_classes * (1-P['clean_rate']))
        unobserved_loss = unobserved_mask.bool() * loss_matrix
        topk = tensorflow.math.top_k(tensorflow.keras.layers.Flatten(unobserved_loss),k)
        topk_lossvalue = topk.values[-1]

        correction_indices = tensorflow.where(unobserved_loss> topk_lossvalue)
        if P['mod_scheme'] in ['LL-Ct','LL-Cp']:
            final_loss_matrix = tensorflow.where(unobserved_loss < topk_lossvalue, loss_matrix, corrected_loss_matrix)
        else:
            zero_loss_matrix = tensorflow.zeros_like(loss_matrix)
            final_loss_matrix = tensorflow.where(unobserved_loss < topk_lossvalue, loss_matrix, zero_loss_matrix)
    main_loss = final_loss_matrix

    return main_loss, correction_indices



