import torch

torch.manual_seed(0)  # fix the random numbers -> the same result at every run
# (must be BEFORE importing model, because model.py creates the random weights)

import math
from date import X_train, X_test, m, s  # the data: 6270 x 21 (train), 4841 x 21 (test)
from model import encoder, reparameterize, decoder, total_loss, params, L  # the model

# =====================================================
# SETTINGS
# =====================================================
N = X_train.shape[0]  # number of training hours: 6270
B = 256  # batch size
epochs = 200  # how many times we go through all the hours

n_batches_per_epoch = math.ceil(N / B)  # 6270 / 256 = 24.49 -> rounded UP = 25
print("batches per epoch:", n_batches_per_epoch)

# the optimizer (notes, section 5): Adam, learning rate eta = 0.001
optimizer = torch.optim.Adam(params, lr=1e-3)


# =====================================================
# EVALUATION: average loss, Recon and KL on a whole data set
# (defined BEFORE training, because we use it during training)
# =====================================================
def evaluate(data):
    with torch.no_grad():  # only checking, no training
        mu_z, logvar_z = encoder(data)
        z = reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = decoder(z)
        loss, recon, kl = total_loss(data, mu_x, logvar_x, mu_z, logvar_z)
    return loss.item(), recon.item(), kl.item()


# =====================================================
# TRAINING
# =====================================================
for epoch in range(epochs):  # outer loop: one epoch = all the hours once
    perm = torch.randperm(N)  # a new random order at every epoch
    X_shuffled = X_train[perm]  # the hours in that random order

    epoch_loss = 0.0  # we add up the loss of each batch here
    epoch_recon = 0.0  # we add up the Recon of each batch here
    epoch_kl = 0.0  # we add up the KL of each batch here
    n_batches = 0  # and count the batches

    for start in range(
        0, N, B
    ):  # inner loop: the 25 batches (start = 0, 256, ..., 6144)
        X = X_shuffled[start : start + B]  # one batch (the last one has 126 hours)

        # forward: X -> encoder -> z -> decoder -> loss
        mu_z, logvar_z = encoder(X)  # give X, receive mu_z and logvar_z
        z = reparameterize(mu_z, logvar_z)  # give mu_z and logvar_z, receive z
        mu_x, logvar_x = decoder(z)  # give z, receive mu_x and logvar_x
        loss, recon, kl = total_loss(
            X, mu_x, logvar_x, mu_z, logvar_z
        )  # receive L = Recon + beta*KL, Recon, KL

        # one optimization step (notes, section 6) - only the total loss is used for the gradient
        optimizer.zero_grad()  # erase the old gradient
        loss.backward()  # compute grad J(theta)
        optimizer.step()  # move theta (Adam update)

        epoch_loss = epoch_loss + loss.item()
        epoch_recon = epoch_recon + recon.item()
        epoch_kl = epoch_kl + kl.item()
        n_batches = n_batches + 1

    # every 10 epochs: print the training averages AND the test loss
    if (epoch + 1) % 10 == 0:
        loss_te, recon_te, kl_te = evaluate(X_test)
        print(
            "epoch",
            epoch + 1,
            "  train loss:",
            round(epoch_loss / n_batches, 2),
            "  recon:",
            round(epoch_recon / n_batches, 2),
            "  kl:",
            round(epoch_kl / n_batches, 2),
            "  | test loss:",
            round(loss_te, 2),
        )

# =====================================================
# FINAL EVALUATION
# =====================================================
loss_tr, recon_tr, kl_tr = evaluate(X_train)  # hours seen during training (2023)
loss_te, recon_te, kl_te = evaluate(X_test)  # hours never seen (2024)
print(
    "train (2023):  loss",
    round(loss_tr, 2),
    "  recon",
    round(recon_tr, 2),
    "  kl",
    round(kl_tr, 2),
)
print(
    "test  (2024):  loss",
    round(loss_te, 2),
    "  recon",
    round(recon_te, 2),
    "  kl",
    round(kl_te, 2),
)

# =====================================================
# ERROR PER STATION, on all the test hours (2024)
# =====================================================
s_t = torch.tensor(s.values, dtype=torch.float32)  # 21 standard deviations
names = list(s.index)  # the 21 station names

with torch.no_grad():
    mu_z, logvar_z = encoder(X_test)
    mu_x, logvar_x = decoder(mu_z)  # decode the center of each zone
    sigma_x = torch.exp(0.5 * logvar_x)

err = (X_test - mu_x) * s_t  # error in ug/m3, 4841 x 21
rmse = torch.sqrt(torch.mean(err**2, dim=0))  # one number per station
within = torch.mean(
    (torch.abs(X_test - mu_x) < sigma_x).float(), dim=0
)  # fraction inside +/- sigma

print()
print("station              RMSE (ug/m3)   relative   inside +/-sigma")
for j in range(21):
    print(
        names[j].ljust(20),
        str(round(rmse[j].item(), 1)).ljust(14),
        str(round(rmse[j].item() / s_t[j].item(), 2)).ljust(10),
        round(100 * within[j].item()),
        "%",
    )

# =====================================================
# KL PER LATENT DIRECTION (which directions does the model use?)
# =====================================================
with torch.no_grad():
    mu_z, logvar_z = encoder(X_test)
    kl_per_dim = 0.5 * torch.mean(
        mu_z**2 + torch.exp(logvar_z) - logvar_z - 1, dim=0
    )  # L numbers

print()
print("direction   KL")
for k in range(L):
    print(str(k + 1).ljust(11), round(kl_per_dim[k].item(), 2))

torch.save(params, "vae_params.pt")
