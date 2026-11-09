import torch  # biblioteca PyTorch

torch.manual_seed(
    0
)  # aceleasi numere aleatoare la fiecare rulare (INAINTE de import model)

import math
from date import X_train, X_test, m, s  # datele
from model_mask import (
    encoder,
    reparameterize,
    decoder,
    total_loss,
    params,
    d,
)  # NOU: modelul cu masca

# =====================================================
# SETARI
# =====================================================
N = X_train.shape[0]  # numarul de ore de antrenare: 6270
B = 256  # marimea batch-ului
epochs = 200  # de cate ori trecem prin toate orele
p_missing = 0.15  # NOU: probabilitatea ca o statie sa fie ascunsa in timpul antrenarii

optimizer = torch.optim.Adam(params, lr=1e-3)  # optimizatorul Adam, eta = 0.001


# =====================================================
# EVALUARE
# cu masca aleatoare: unele statii ascunse (ca la antrenare)
# cu masca plina de 1: ore complete (ca la modelul vechi)
# =====================================================
def evaluate(data, mask):
    with torch.no_grad():  # doar verificam
        mu_z, logvar_z = encoder(
            data * mask, mask
        )  # encoderul vede statiile vizibile + masca
        z = reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = decoder(z)
        loss, recon, kl = total_loss(
            data, mu_x, logvar_x, mu_z, logvar_z
        )  # Recon pe TOATE statiile
    return loss.item(), recon.item(), kl.item()


# masti FIXE pentru test, ca pierderea pe test sa fie comparabila de la o epoca la alta
mask_test_random = (
    torch.rand(X_test.shape[0], d) > p_missing
).float()  # cam 15% ascunse
mask_test_full = torch.ones(X_test.shape[0], d)  # nimic ascuns

# =====================================================
# ANTRENAREA
# =====================================================
for epoch in range(epochs):  # bucla exterioara: epocile
    perm = torch.randperm(N)  # o ordine aleatoare noua la fiecare epoca
    X_shuffled = X_train[perm]

    epoch_loss = 0.0  # adunam pierderile batch-urilor
    epoch_recon = 0.0
    epoch_kl = 0.0
    n_batches = 0

    for start in range(0, N, B):  # bucla interioara: cele 25 de batch-uri
        X = X_shuffled[start : start + B]  # orele COMPLETE din acest batch

        mask = (
            torch.rand(X.shape[0], d) > p_missing
        ).float()  # NOU: o masca aleatoare noua pentru fiecare ora, la fiecare pas
        X_tilde = X * mask  # NOU: ascundem statiile (devin 0)

        # forward
        mu_z, logvar_z = encoder(
            X_tilde, mask
        )  # NOU: encoderul vede doar statiile vizibile + masca
        z = reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = decoder(z)
        loss, recon, kl = total_loss(
            X, mu_x, logvar_x, mu_z, logvar_z
        )  # costul fata de X COMPLET -> invata sa ghiceasca statiile ascunse

        # un pas de optimizare
        optimizer.zero_grad()  # stergem gradientul vechi
        loss.backward()  # calculam gradientul
        optimizer.step()  # mutam parametrii (Adam)

        epoch_loss = epoch_loss + loss.item()
        epoch_recon = epoch_recon + recon.item()
        epoch_kl = epoch_kl + kl.item()
        n_batches = n_batches + 1

    if (epoch + 1) % 10 == 0:  # o data la 10 epoci
        loss_te, recon_te, kl_te = evaluate(X_test, mask_test_random)
        print(
            "epoch",
            epoch + 1,
            "  train loss:",
            round(epoch_loss / n_batches, 2),
            "  recon:",
            round(epoch_recon / n_batches, 2),
            "  kl:",
            round(epoch_kl / n_batches, 2),
            "  | test loss (15% hidden):",
            round(loss_te, 2),
        )

# =====================================================
# EVALUAREA FINALA
# =====================================================
loss_r, recon_r, kl_r = evaluate(X_test, mask_test_random)  # test cu 15% ascunse
loss_f, recon_f, kl_f = evaluate(
    X_test, mask_test_full
)  # test fara nimic ascuns (comparabil cu modelul vechi: 8.43)
print(
    "test, 15% hidden:     loss",
    round(loss_r, 2),
    "  recon",
    round(recon_r, 2),
    "  kl",
    round(kl_r, 2),
)
print(
    "test, nothing hidden: loss",
    round(loss_f, 2),
    "  recon",
    round(recon_f, 2),
    "  kl",
    round(kl_f, 2),
)

# =====================================================
# SALVAREA
# =====================================================
torch.save(
    params, "vae_mask_params.pt"
)  # NOU: alt fisier, modelul vechi ramane in vae_params.pt
print("model saved to vae_mask_params.pt")
