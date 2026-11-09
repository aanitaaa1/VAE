import torch  # biblioteca PyTorch
from date import (
    X_train,
    X_test,
    m,
    s,
)  # datele (standardizate), media si deviatia statiilor
import model as old  # modelul VECHI  -> old.encoder, old.decoder, old.params
import model_mask as new  # modelul cu MASCA -> new.encoder, new.decoder, new.params

d = 21  # numarul de statii
names = list(s.index)  # numele celor 21 de statii


# =====================================================
# BLOCUL 1: INCARCAM AMBELE MODELE ANTRENATE
# =====================================================
def load(module, filename):  # module = old sau new, filename = fisierul cu parametrii
    saved = torch.load(filename)  # citim matricele antrenate
    with torch.no_grad():
        for p, p_saved in zip(module.params, saved):  # parametrii doi cate doi
            p.copy_(p_saved)  # punem numerele antrenate
            p.requires_grad_(False)  # ponderile raman fixe


load(old, "vae_params.pt")  # modelul vechi
load(new, "vae_mask_params.pt")  # modelul cu masca


# =====================================================
# BLOCUL 2: REGRESIA LINIARA (referinta)
# =====================================================
def impute_linear(hidden):  # hidden = lista statiilor lipsa, de ex. [4, 8]
    observed = [k for k in range(d) if k not in hidden]  # statiile care raman
    A_train = torch.cat(
        [X_train[:, observed], torch.ones(X_train.shape[0], 1)], dim=1
    )  # statiile ramase + coloana de 1 (2023)
    A_test = torch.cat(
        [X_test[:, observed], torch.ones(X_test.shape[0], 1)], dim=1
    )  # la fel, 2024
    w = torch.linalg.lstsq(
        A_train, X_train[:, hidden]
    ).solution  # ponderile pentru toate statiile lipsa
    return A_test @ w  # ghicirile liniare (4841 x numarul de statii lipsa)


# =====================================================
# BLOCUL 3: MODELUL VECHI + METODA C
# =====================================================
def impute_old_C(X, mask, n_iter_B=50, n_steps=100, lr=0.05):
    X_fill = X * mask  # 0 la statiile lipsa
    with torch.no_grad():
        for t in range(n_iter_B):  # metoda B: 50 de repetari
            mu_z, logvar_z = old.encoder(X_fill)
            mu_x, logvar_x = old.decoder(mu_z)
            X_fill = X * mask + mu_x * (1 - mask)  # inlocuim doar statiile lipsa
        z_start, _ = old.encoder(X_fill)  # punctul de pornire
    z = z_start.clone().requires_grad_()  # z-ul pe care il miscam
    optimizer_z = torch.optim.Adam([z], lr=lr)
    for step in range(n_steps):  # metoda C: cautam z-ul
        mu_x, logvar_x = old.decoder(z)
        recon_obs = 0.5 * torch.sum(
            mask * (logvar_x + (X - mu_x) ** 2 / torch.exp(logvar_x)), dim=1
        )  # Recon doar pe statiile observate
        prior = 0.5 * torch.sum(z**2, dim=1)  # tine z pe harta
        loss_z = torch.sum(recon_obs + prior)
        optimizer_z.zero_grad()
        loss_z.backward()
        optimizer_z.step()
    with torch.no_grad():
        mu_x, logvar_x = old.decoder(z)
    return mu_x  # ghicirile pentru toate cele 21 de statii


# =====================================================
# BLOCUL 4: MODELUL CU MASCA (o trecere + 50 de esantioane pentru incertitudine)
# =====================================================
def impute_new(X, mask, S=50):  # S = cate puncte z extragem din zona
    M = mask.expand(X.shape[0], d)  # aceeasi masca pentru toate orele (4841 x 21)
    with torch.no_grad():
        mu_z, logvar_z = new.encoder(
            X * M, M
        )  # encoderul vede statiile vizibile + masca

        sum_mu = torch.zeros_like(X)  # adunam ghicirile
        sum_mu2 = torch.zeros_like(X)  # adunam ghicirile la patrat
        sum_var = torch.zeros_like(X)  # adunam sigma_x^2

        for _ in range(S):  # 50 de esantioane
            z = new.reparameterize(mu_z, logvar_z)  # un punct z din zona
            mu_x, logvar_x = new.decoder(z)
            sum_mu = sum_mu + mu_x
            sum_mu2 = sum_mu2 + mu_x**2
            sum_var = sum_var + torch.exp(logvar_x)

        guess = sum_mu / S  # media ghicirilor
        var_total = sum_var / S + (
            sum_mu2 / S - guess**2
        )  # zgomotul statiei + nesiguranta despre z
    return guess, torch.sqrt(var_total)


# =====================================================
# BLOCUL 5: 1, 2, 3, 5 STATII LIPSA, ALESE ALEATOR, 20 DE INCERCARI
# =====================================================
g = torch.Generator().manual_seed(
    1
)  # generator separat -> aceleasi combinatii ca data trecuta
n_trials = 20

print()
print(
    "missing   linear   old (C)   new (mask)   new better than linear   inside +/-sigma (new)"
)
for k in [1, 2, 3, 5]:
    sum_lin, sum_old, sum_new, sum_inside = 0.0, 0.0, 0.0, 0.0
    wins = 0

    for trial in range(n_trials):
        hidden = torch.randperm(d, generator=g)[:k].tolist()  # k statii alese aleator
        mask = torch.ones(d)
        mask[hidden] = 0

        real = X_test[:, hidden]  # valorile reale
        pred_lin = impute_linear(hidden)  # regresia liniara
        pred_old = impute_old_C(X_test, mask)[:, hidden]  # modelul vechi + C
        guess_new, sigma_new = impute_new(X_test, mask)  # modelul cu masca
        pred_new = guess_new[:, hidden]

        e_lin = torch.sqrt(torch.mean((real - pred_lin) ** 2)).item()
        e_old = torch.sqrt(torch.mean((real - pred_old) ** 2)).item()
        e_new = torch.sqrt(torch.mean((real - pred_new) ** 2)).item()
        inside = torch.mean(
            (torch.abs(real - pred_new) < sigma_new[:, hidden]).float()
        ).item()

        sum_lin = sum_lin + e_lin
        sum_old = sum_old + e_old
        sum_new = sum_new + e_new
        sum_inside = sum_inside + inside
        if e_new < e_lin:
            wins = wins + 1

    print(
        str(k).ljust(10),
        str(round(sum_lin / n_trials, 3)).ljust(9),
        str(round(sum_old / n_trials, 3)).ljust(10),
        str(round(sum_new / n_trials, 3)).ljust(13),
        (str(wins) + " / " + str(n_trials)).ljust(25),
        round(100 * sum_inside / n_trials),
        "%",
    )

# =====================================================
# BLOCUL 6: STATII VECINE LIPSA DEODATA (Gracia NO2 + Eixample NO2 + Gracia O3)
# =====================================================
hidden = [4, 8, 18]
mask = torch.ones(d)
mask[hidden] = 0

pred_lin = impute_linear(hidden)
pred_old = impute_old_C(X_test, mask)
guess_new, sigma_new = impute_new(X_test, mask)

print()
print("Hiding together:", [names[k] for k in hidden])
print("station              linear   old (C)   new (mask)   inside +/-sigma (new)")
for i, k in enumerate(hidden):
    real = X_test[:, k]
    e_lin = torch.sqrt(torch.mean((real - pred_lin[:, i]) ** 2)).item()
    e_old = torch.sqrt(torch.mean((real - pred_old[:, k]) ** 2)).item()
    e_new = torch.sqrt(torch.mean((real - guess_new[:, k]) ** 2)).item()
    inside = torch.mean(
        (torch.abs(real - guess_new[:, k]) < sigma_new[:, k]).float()
    ).item()
    print(
        names[k].ljust(20),
        str(round(e_lin, 2)).ljust(8),
        str(round(e_old, 2)).ljust(9),
        str(round(e_new, 2)).ljust(12),
        round(100 * inside),
        "%",
    )
