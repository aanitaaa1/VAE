import torch
import math
from date import X_train  # the data: 6270 x 21
from model import encoder, reparameterize, decoder, total_loss, params  # the model

# =====================================================
# SETTINGS
# =====================================================
N = X_train.shape[0]  # number of training hours: 6270
B = 256  # batch size
epochs = 50  # how many times we go through all the hours

n_batches_per_epoch = math.ceil(N / B)  # 6270 / 256 = 24.49 -> rounded UP = 25
print("batches per epoch:", n_batches_per_epoch)

# the optimizer (notes, section 5): Adam, learning rate eta = 0.001
optimizer = torch.optim.Adam(params, lr=1e-3)

# =====================================================
# TRAINING
# =====================================================
for epoch in range(epochs):  # outer loop: one epoch = all the hours once
    perm = torch.randperm(N)  # a new random order at every epoch
    X_shuffled = X_train[perm]

    epoch_loss = 0.0  # we add up the loss of each batch here
    n_batches = 0  # and count the batches

    for start in range(0, N, B):  # inner loop: the 25 batches
        X = X_shuffled[start : start + B]  # one batch (the last one has 126 hours)

        # forward: X -> encoder -> z -> decoder -> loss
        mu_z, logvar_z = encoder(X)
        z = reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = decoder(z)
        loss = total_loss(X, mu_x, logvar_x, mu_z, logvar_z)

        # one optimization step (notes, section 6)
        optimizer.zero_grad()  # erase the old gradient
        loss.backward()  # compute grad J(theta)
        optimizer.step()  # move theta (Adam update)

        epoch_loss = epoch_loss + loss.item()
        n_batches = n_batches + 1

    print("epoch", epoch + 1, "  average loss:", epoch_loss / n_batches)
