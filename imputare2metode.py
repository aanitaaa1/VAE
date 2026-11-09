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
        for p, p_saved in zip(module.params, saved):
            p.copy_(p_saved)  # punem numerele antrenate
            p.requires_grad_(False)  # ponderile raman fixe


load(old, "vae_params.pt")  # modelul vechi
load(new, "vae_mask_params.pt")  # modelul cu masca


# =====================================================
# BLOCUL 2: REGRESIA LINIARA
# =====================================================
def impute_linear(hidden):  # hidden = lista statiilor lipsa
    observed = [k for k in range(d) if k not in hidden]  # statiile care raman
    A_train = torch.cat([X_train[:, observed], torch.ones(X_train.shape[0], 1)], dim=1)
    A_test = torch.cat([X_test[:, observed], torch.ones(X_test.shape[0], 1)], dim=1)
    w = torch.linalg.lstsq(A_train, X_train[:, hidden]).solution
    return A_test @ w  # ghicirile liniare (4841 x numarul de statii lipsa)


# =====================================================
# BLOCUL 3: CAUTAREA LUI z (metoda C), pentru ORICE decoder
# o folosim si pentru modelul vechi, si pentru hibrid
# =====================================================
def search_z(decoder_fn, X, mask, z_start, n_steps=100, lr=0.05):
    z = z_start.clone().requires_grad_()  # z-ul pe care il miscam
    optimizer_z = torch.optim.Adam([z], lr=lr)  # optimizator doar pentru z
    for step in range(n_steps):
        mu_x, logvar_x = decoder_fn(z)  # ce spune decoderul din z-ul curent
        recon_obs = 0.5 * torch.sum(
            mask * (logvar_x + (X - mu_x) ** 2 / torch.exp(logvar_x)), dim=1
        )  # Recon doar pe statiile observate
        prior = 0.5 * torch.sum(z**2, dim=1)  # tine z pe harta
        loss_z = torch.sum(recon_obs + prior)
        optimizer_z.zero_grad()
        loss_z.backward()
        optimizer_z.step()
    with torch.no_grad():
        mu_x, logvar_x = decoder_fn(z)  # decodam z-ul gasit
    return mu_x  # ghicirile pentru toate cele 21 de statii


# =====================================================
# BLOCUL 4: MODELUL VECHI + C (pornire din metoda B)
# =====================================================
def impute_old_C(X, mask):
    X_fill = X * mask  # 0 la statiile lipsa
    with torch.no_grad():
        for t in range(50):  # metoda B: 50 de repetari
            mu_z, logvar_z = old.encoder(X_fill)
            mu_x, logvar_x = old.decoder(mu_z)
            X_fill = X * mask + mu_x * (1 - mask)
        z_start, _ = old.encoder(X_fill)  # punctul de pornire
    return search_z(old.decoder, X, mask, z_start)  # cautarea cu decoderul VECHI


# =====================================================
# BLOCUL 5: MODELUL CU MASCA, O TRECERE + 50 DE ESANTIOANE
# =====================================================
def impute_new(X, mask, S=50):
    M = mask.expand(X.shape[0], d)  # aceeasi masca pentru toate orele
    with torch.no_grad():
        mu_z, logvar_z = new.encoder(X * M, M)  # zona pe harta
        sum_mu = torch.zeros_like(X)
        sum_mu2 = torch.zeros_like(X)
        sum_var = torch.zeros_like(X)
        for _ in range(S):  # 50 de esantioane
            z = new.reparameterize(mu_z, logvar_z)
            mu_x, logvar_x = new.decoder(z)
            sum_mu = sum_mu + mu_x
            sum_mu2 = sum_mu2 + mu_x**2
            sum_var = sum_var + torch.exp(logvar_x)
        guess = sum_mu / S  # media ghicirilor
        var_total = sum_var / S + (sum_mu2 / S - guess**2)  # varianta totala
    return (
        guess,
        torch.sqrt(var_total),
        mu_z,
    )  # intoarcem si mu_z: e punctul de pornire pentru hibrid


# =====================================================
# BLOCUL 6 (NOU): HIBRID = encoderul cu masca da pornirea, cautarea rafineaza
# =====================================================
def impute_hybrid(
    X, mask, mu_z_new
):  # mu_z_new = centrul zonei dat de encoderul cu masca
    return search_z(
        new.decoder, X, mask, mu_z_new
    )  # cautarea cu decoderul NOU, pornind de la encoderul nou


# =====================================================
# BLOCUL 7: 1, 2, 3, 5 STATII LIPSA, 20 DE INCERCARI
# =====================================================
g = torch.Generator().manual_seed(1)  # aceleasi combinatii de statii ca data trecuta
n_trials = 20

print()
print(
    "missing   linear   old (C)   new      hybrid   ensemble   inside +/-sigma (hybrid)"
)
for k in [1, 2, 3, 5]:
    sums = {
        "lin": 0.0,
        "old": 0.0,
        "new": 0.0,
        "hyb": 0.0,
        "ens": 0.0,
    }  # aici adunam erorile fiecarei metode
    sum_inside = 0.0

    for trial in range(n_trials):
        hidden = torch.randperm(d, generator=g)[:k].tolist()  # k statii alese aleator
        mask = torch.ones(d)
        mask[hidden] = 0

        real = X_test[:, hidden]  # valorile reale
        pred_lin = impute_linear(hidden)  # regresia
        pred_old = impute_old_C(X_test, mask)[:, hidden]  # modelul vechi + C
        guess_new, sigma_new, mu_z_new = impute_new(X_test, mask)  # modelul cu masca
        pred_new = guess_new[:, hidden]
        M = mask.expand(X_test.shape[0], d)
        pred_hyb = impute_hybrid(X_test, M, mu_z_new)[:, hidden]  # hibridul
        pred_ens = (pred_lin + pred_old + pred_new) / 3  # media celor trei

        for name, pred in [
            ("lin", pred_lin),
            ("old", pred_old),
            ("new", pred_new),
            ("hyb", pred_hyb),
            ("ens", pred_ens),
        ]:
            sums[name] = (
                sums[name] + torch.sqrt(torch.mean((real - pred) ** 2)).item()
            )  # eroarea relativa a fiecarei metode

        inside = torch.mean(
            (torch.abs(real - pred_hyb) < sigma_new[:, hidden]).float()
        ).item()  # incertitudinea modelului cu masca, pentru ghicirile hibridului
        sum_inside = sum_inside + inside

    print(
        str(k).ljust(10),
        str(round(sums["lin"] / n_trials, 3)).ljust(9),
        str(round(sums["old"] / n_trials, 3)).ljust(10),
        str(round(sums["new"] / n_trials, 3)).ljust(9),
        str(round(sums["hyb"] / n_trials, 3)).ljust(9),
        str(round(sums["ens"] / n_trials, 3)).ljust(11),
        round(100 * sum_inside / n_trials),
        "%",
    )
