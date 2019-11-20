from __future__ import print_function

import random

import numpy as np
import torch
from torch.autograd import Variable

from rougefonc import from_summary_index_compute_rouge


method = 'herke'

# loss cumputed by one sentence each time:
def extract_sent_compute_loss(probs,mask,epsilon = 0.1):
    """
    extract one sentence based on e-greedy
    return selected_idx, loss_i
    """
    # self.probs_torch = probs
    # self.probs_torch = torch.clamp(probs, 1e-6, 1 - 1e-6)  # this just make sure no zero
    # print('%'*50)
    # print(probs.shape,probs)
    probs_torch = probs * 0.9999 + 0.00005  # this just make sure no zero
    probs_torch = probs_torch.squeeze()
    # probs_numpy = probs.data.cpu().numpy()
    # probs_numpy = np.reshape(probs_numpy, len(probs_numpy))

    # herke's method
    p_masked = probs_torch * mask

    if random.uniform(0, 1) <= epsilon:  # explore
        selected_idx = torch.multinomial(mask, 1)
    else:
        selected_idx = torch.multinomial(p_masked, 1)
    loss_i = (epsilon / mask.sum() + (1 - epsilon) * p_masked[selected_idx] / p_masked.sum()).log()

    return selected_idx,loss_i