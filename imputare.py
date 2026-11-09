import torch  # biblioteca PyTorch (tensori, retele neuronale)
from date import (
    X_train,
    X_test,
    m,
    s,
)  # din date.py: orele din 2023 si 2024 (standardizate), media si deviatia statiilor
from model import (
    encoder,
    reparameterize,
    decoder,
    total_loss,
    params,
)  # din model.py: functiile si lista cu cei 12 parametri

# =====================================================
# BLOCUL 1: INCARCAM MODELUL ANTRENAT                                        (ACTIV)
# =====================================================
saved = torch.load("vae_params.pt")  # citim din fisier cele 12 matrice antrenate
with torch.no_grad():  # nu antrenam aici
    for p, p_saved in zip(
        params, saved
    ):  # luam parametrii doi cate doi: W_h cu W_h antrenat, b_h cu b_h antrenat, ...
        p.copy_(p_saved)  # punem numerele antrenate in locul celor aleatoare
        p.requires_grad_(False)  # ponderile raman fixe (la metoda C se misca doar z)

# verificare: pierderea pe test trebuie sa fie in jur de 8.4, ca la finalul antrenarii
with torch.no_grad():
    mu_z, logvar_z = encoder(
        X_test
    )  # encoderul: cele 4841 de ore din 2024 -> centrul si latimea zonei
    z = reparameterize(mu_z, logvar_z)  # alegem un punct z din fiecare zona
    mu_x, logvar_x = decoder(z)  # decoderul: z -> ora reconstruita si incertitudinea
    loss, recon, kl = total_loss(
        X_test, mu_x, logvar_x, mu_z, logvar_z
    )  # pierderea (Recon + KL)
print("test loss of the loaded model:", round(loss.item(), 2))

s_t = torch.tensor(
    s.values, dtype=torch.float32
)  # deviatiile celor 21 de statii (pentru ug/m3)
m_t = torch.tensor(m.values, dtype=torch.float32)  # mediile celor 21 de statii
names = list(s.index)  # numele celor 21 de statii

# =====================================================
# BLOCUL 2: METODA A, CA FUNCTIE                                             (COMENTAT)
# =====================================================
# def impute_A(X, j):                             # X = orele, j = numarul statiei ascunse
#     mask = torch.ones(21)                       # masca: 1 = observata
#     mask[j] = 0                                 # statia j: 0 = lipsa
#     X_tilde = X * mask                          # statia j devine 0 (= media ei)
#     with torch.no_grad():
#         mu_z, logvar_z = encoder(X_tilde)       # encoderul vede 20 de statii reale + 0 la statia j
#         mu_x, logvar_x = decoder(mu_z)          # decodam centrul zonei
#     guess = mu_x[:, j]                          # ghicirea pentru statia j
#     sigma = torch.exp(0.5 * logvar_x[:, j])     # incertitudinea
#     return guess, sigma

# =====================================================
# BLOCUL 3: METODA A, GRACIA, PRIMELE 5 ORE                                  (COMENTAT)
# =====================================================
# j = 4
# guess, sigma = impute_A(X_test, j)
# print()
# print("Hiding", names[j], "- first 5 hours of 2024 (ug/m3):")
# print("real    guess   +/-")
# for n in range(5):
#     real_ug = X_test[n, j] * s_t[j] + m_t[j]
#     guess_ug = guess[n] * s_t[j] + m_t[j]
#     sigma_ug = sigma[n] * s_t[j]
#     print(round(real_ug.item(), 1), "  ", round(guess_ug.item(), 1), "  ", round(sigma_ug.item(), 1))

# =====================================================
# BLOCUL 4: METODA A, TOATE STATIILE (linear vs A)                           (COMENTAT)
# =====================================================
# print()
# print("station              linear   VAE (A)   inside +/-sigma")
# for j in range(21):
#     guess, sigma = impute_A(X_test, j)
#     rel_vae = torch.sqrt(torch.mean((X_test[:, j] - guess) ** 2))
#     inside = torch.mean((torch.abs(X_test[:, j] - guess) < sigma).float())
#     others = [k for k in range(21) if k != j]
#     A_train = torch.cat([X_train[:, others], torch.ones(X_train.shape[0], 1)], dim=1)
#     A_test = torch.cat([X_test[:, others], torch.ones(X_test.shape[0], 1)], dim=1)
#     w = torch.linalg.lstsq(A_train, X_train[:, j:j+1]).solution
#     pred_lin = (A_test @ w)[:, 0]
#     rel_lin = torch.sqrt(torch.mean((X_test[:, j] - pred_lin) ** 2))
#     print(names[j].ljust(20), str(round(rel_lin.item(), 2)).ljust(8),
#           str(round(rel_vae.item(), 2)).ljust(9), round(100 * inside.item()), "%")

# =====================================================
# BLOCUL 5: METODA B, O SINGURA STATIE                                       (COMENTAT)
# =====================================================
# def impute_B(X, j, n_iter=50):
#     mask = torch.ones(21)
#     mask[j] = 0
#     X_fill = X * mask
#     with torch.no_grad():
#         for t in range(n_iter):
#             mu_z, logvar_z = encoder(X_fill)
#             mu_x, logvar_x = decoder(mu_z)
#             X_fill = X * mask + mu_x * (1 - mask)
#     guess = mu_x[:, j]
#     sigma = torch.exp(0.5 * logvar_x[:, j])
#     return guess, sigma

# =====================================================
# BLOCUL 6: METODA B, CUM SE SCHIMBA GHICIREA (Gracia, ora 2)                (COMENTAT)
# =====================================================
# j = 4
# n = 2
# x_one = X_test[n:n+1]
# print()
# print("Method B on", names[j], "- hour", n, "- real value:", round((X_test[n, j] * s_t[j] + m_t[j]).item(), 1))
# print("iteration   guess (ug/m3)")
# for n_iter in [1, 2, 5, 10, 20, 50, 100]:
#     guess, sigma = impute_B(x_one, j, n_iter)
#     guess_ug = guess[0] * s_t[j] + m_t[j]
#     print(str(n_iter).ljust(11), round(guess_ug.item(), 1))

# =====================================================
# BLOCUL 7: TOATE STATIILE (linear vs A vs B)                                (COMENTAT)
# =====================================================
# print()
# print("station              linear   VAE (A)   VAE (B)   inside +/-sigma (B)")
# for j in range(21):
#     guess_A, sigma_A = impute_A(X_test, j)
#     rel_A = torch.sqrt(torch.mean((X_test[:, j] - guess_A) ** 2))
#     guess_B, sigma_B = impute_B(X_test, j)
#     rel_B = torch.sqrt(torch.mean((X_test[:, j] - guess_B) ** 2))
#     inside_B = torch.mean((torch.abs(X_test[:, j] - guess_B) < sigma_B).float())
#     others = [k for k in range(21) if k != j]
#     A_train = torch.cat([X_train[:, others], torch.ones(X_train.shape[0], 1)], dim=1)
#     A_test = torch.cat([X_test[:, others], torch.ones(X_test.shape[0], 1)], dim=1)
#     w = torch.linalg.lstsq(A_train, X_train[:, j:j+1]).solution
#     pred_lin = (A_test @ w)[:, 0]
#     rel_lin = torch.sqrt(torch.mean((X_test[:, j] - pred_lin) ** 2))
#     print(names[j].ljust(20), str(round(rel_lin.item(), 2)).ljust(8), str(round(rel_A.item(), 2)).ljust(9),
#           str(round(rel_B.item(), 2)).ljust(9), round(100 * inside_B.item()), "%")

# =====================================================
# BLOCUL 8: METODA C, O SINGURA STATIE                                       (COMENTAT)
# =====================================================
# def impute_C(X, j, n_steps=100, lr=0.05):
#     mask = torch.ones(21)
#     mask[j] = 0
#     X_fill = X * mask
#     with torch.no_grad():
#         for t in range(50):
#             mu_z, logvar_z = encoder(X_fill)
#             mu_x, logvar_x = decoder(mu_z)
#             X_fill = X * mask + mu_x * (1 - mask)
#         z_start, _ = encoder(X_fill)
#     z = z_start.clone().requires_grad_()
#     optimizer_z = torch.optim.Adam([z], lr=lr)
#     for step in range(n_steps):
#         mu_x, logvar_x = decoder(z)
#         recon_obs = 0.5 * torch.sum(mask * (logvar_x + (X - mu_x) ** 2 / torch.exp(logvar_x)), dim=1)
#         prior = 0.5 * torch.sum(z ** 2, dim=1)
#         loss_z = torch.sum(recon_obs + prior)
#         optimizer_z.zero_grad()
#         loss_z.backward()
#         optimizer_z.step()
#     with torch.no_grad():
#         mu_x, logvar_x = decoder(z)
#     guess = mu_x[:, j]
#     sigma = torch.exp(0.5 * logvar_x[:, j])
#     return guess, sigma

# =====================================================
# BLOCUL 9: METODA C, CUM SE SCHIMBA GHICIREA (Gracia, ora 2)                (COMENTAT)
# =====================================================
# j = 4
# n = 2
# x_one = X_test[n:n+1]
# print()
# print("Method C on", names[j], "- hour", n, "- real value:", round((X_test[n, j] * s_t[j] + m_t[j]).item(), 1))
# print("search steps   guess (ug/m3)")
# for n_steps in [0, 10, 50, 100, 200]:
#     guess, sigma = impute_C(x_one, j, n_steps)
#     guess_ug = guess[0] * s_t[j] + m_t[j]
#     print(str(n_steps).ljust(14), round(guess_ug.item(), 1))

# =====================================================
# BLOCUL 10: TOATE STATIILE (linear vs B vs C), O STATIE LIPSA               (COMENTAT)
# =====================================================
# print()
# print("station              linear   VAE (B)   VAE (C)   inside +/-sigma (C)")
# for j in range(21):
#     guess_B, sigma_B = impute_B(X_test, j)
#     rel_B = torch.sqrt(torch.mean((X_test[:, j] - guess_B) ** 2))
#     guess_C, sigma_C = impute_C(X_test, j)
#     rel_C = torch.sqrt(torch.mean((X_test[:, j] - guess_C) ** 2))
#     inside_C = torch.mean((torch.abs(X_test[:, j] - guess_C) < sigma_C).float())
#     others = [k for k in range(21) if k != j]
#     A_train = torch.cat([X_train[:, others], torch.ones(X_train.shape[0], 1)], dim=1)
#     A_test = torch.cat([X_test[:, others], torch.ones(X_test.shape[0], 1)], dim=1)
#     w = torch.linalg.lstsq(A_train, X_train[:, j:j+1]).solution
#     pred_lin = (A_test @ w)[:, 0]
#     rel_lin = torch.sqrt(torch.mean((X_test[:, j] - pred_lin) ** 2))
#     print(names[j].ljust(20), str(round(rel_lin.item(), 2)).ljust(8), str(round(rel_B.item(), 2)).ljust(9),
#           str(round(rel_C.item(), 2)).ljust(9), round(100 * inside_C.item()), "%")


# =====================================================
# BLOCUL 11 (NOU): FUNCTII CARE PRIMESC O MASCA (oricate statii lipsa)       (ACTIV)
# =====================================================
def impute_BC(
    X, mask, n_iter_B=50, n_steps=100, lr=0.05
):  # mask: 1 = observata, 0 = lipsa (oricate zerouri)
    # --- metoda B: umplem, decodam, reumplem ---
    X_fill = X * mask  # pornim cu 0 la toate statiile lipsa
    with torch.no_grad():  # aici doar folosim modelul
        for t in range(n_iter_B):  # 50 de repetari
            mu_z, logvar_z = encoder(X_fill)  # encoderul vede ghicirea curenta
            mu_x, logvar_x = decoder(
                mu_z
            )  # decoderul da o ghicire noua pentru toate statiile
            X_fill = X * mask + mu_x * (
                1 - mask
            )  # pastram statiile observate, inlocuim TOATE statiile lipsa
        mu_x_B = mu_x  # rezultatul metodei B (toate cele 21 de statii)
        z_start, _ = encoder(X_fill)  # punctul de pornire pentru metoda C

    # --- metoda C: cautam z-ul care explica statiile observate ---
    z = z_start.clone().requires_grad_()  # copie a lui z, marcata "de miscat"
    optimizer_z = torch.optim.Adam([z], lr=lr)  # optimizator doar pentru z
    for step in range(n_steps):  # pasii de cautare
        mu_x, logvar_x = decoder(z)  # ce spune decoderul din z-ul curent
        recon_obs = 0.5 * torch.sum(
            mask * (logvar_x + (X - mu_x) ** 2 / torch.exp(logvar_x)), dim=1
        )  # Recon DOAR pe statiile observate
        prior = 0.5 * torch.sum(z**2, dim=1)  # 1/2 ||z||^2: tine z pe harta
        loss_z = torch.sum(
            recon_obs + prior
        )  # adunam pe ore (fiecare ora se cauta independent)
        optimizer_z.zero_grad()  # stergem gradientul vechi
        loss_z.backward()  # gradientul fata de z
        optimizer_z.step()  # mutam z putin

    with torch.no_grad():
        mu_x_C, logvar_x_C = decoder(z)  # decodam z-ul gasit
    sigma_C = torch.exp(0.5 * logvar_x_C)  # incertitudinea pentru toate statiile
    return (
        mu_x_B,
        mu_x_C,
        sigma_C,
    )  # intoarcem: ghicirile B, ghicirile C, incertitudinea C (toate 21 de statii)


def impute_linear(hidden):  # hidden = lista statiilor lipsa, de ex. [4, 8]
    observed = [k for k in range(21) if k not in hidden]  # statiile care raman
    A_train = torch.cat(
        [X_train[:, observed], torch.ones(X_train.shape[0], 1)], dim=1
    )  # statiile ramase + coloana de 1 (2023)
    A_test = torch.cat(
        [X_test[:, observed], torch.ones(X_test.shape[0], 1)], dim=1
    )  # la fel, 2024
    w = torch.linalg.lstsq(
        A_train, X_train[:, hidden]
    ).solution  # ponderile pentru TOATE statiile lipsa deodata (o coloana pe statie)
    return A_test @ w  # ghicirile liniare pe 2024 (4841 x numarul de statii lipsa)


# =====================================================
# BLOCUL 12 (NOU): 1, 2, 3, 5 STATII LIPSA, ALESE ALEATOR, 20 DE INCERCARI    (ACTIV)
# =====================================================
torch.manual_seed(1)  # aceleasi alegeri aleatoare la fiecare rulare
n_trials = 20  # de cate ori alegem alte statii

print()
print("missing   linear   VAE (B)   VAE (C)   C better than linear")
for k in [1, 2, 3, 5]:  # cate statii lipsesc deodata
    sum_lin, sum_B, sum_C = (
        0.0,
        0.0,
        0.0,
    )  # aici adunam erorile din cele 20 de incercari
    wins = 0  # de cate ori C bate regresia liniara

    for trial in range(n_trials):  # 20 de incercari
        hidden = torch.randperm(21)[
            :k
        ].tolist()  # k statii alese aleator, de ex. [4, 17]
        mask = torch.ones(21)  # masca: 1 peste tot
        mask[hidden] = 0  # 0 la toate statiile alese

        mu_x_B, mu_x_C, sigma_C = impute_BC(X_test, mask)  # imputam cu B si C
        pred_lin = impute_linear(hidden)  # imputam cu regresia liniara

        real = X_test[:, hidden]  # valorile reale ale statiilor ascunse
        e_lin = torch.sqrt(
            torch.mean((real - pred_lin) ** 2)
        ).item()  # eroarea relativa liniara
        e_B = torch.sqrt(
            torch.mean((real - mu_x_B[:, hidden]) ** 2)
        ).item()  # eroarea relativa B
        e_C = torch.sqrt(
            torch.mean((real - mu_x_C[:, hidden]) ** 2)
        ).item()  # eroarea relativa C

        sum_lin = sum_lin + e_lin
        sum_B = sum_B + e_B
        sum_C = sum_C + e_C
        if e_C < e_lin:  # daca C a fost mai buna in aceasta incercare
            wins = wins + 1

    print(
        str(k).ljust(10),
        str(round(sum_lin / n_trials, 3)).ljust(9),
        str(round(sum_B / n_trials, 3)).ljust(10),
        str(round(sum_C / n_trials, 3)).ljust(10),
        wins,
        "/",
        n_trials,
    )

# =====================================================
# BLOCUL 13 (NOU): CAZ SPECIAL - STATII DIN ACEEASI ZONA LIPSA DEODATA       (ACTIV)
# Gracia NO2 (4) + Eixample NO2 (8) + Gracia O3 (18)
# =====================================================
hidden = [4, 8, 18]  # cele trei statii vecine
mask = torch.ones(21)
mask[hidden] = 0

mu_x_B, mu_x_C, sigma_C = impute_BC(X_test, mask)
pred_lin = impute_linear(hidden)

print()
print("Hiding together:", [names[k] for k in hidden])
print("station              linear   VAE (B)   VAE (C)   inside +/-sigma (C)")
for i, k in enumerate(hidden):  # i = pozitia in lista (0, 1, 2), k = numarul statiei
    real = X_test[:, k]
    e_lin = torch.sqrt(torch.mean((real - pred_lin[:, i]) ** 2)).item()
    e_B = torch.sqrt(torch.mean((real - mu_x_B[:, k]) ** 2)).item()
    e_C = torch.sqrt(torch.mean((real - mu_x_C[:, k]) ** 2)).item()
    inside = torch.mean((torch.abs(real - mu_x_C[:, k]) < sigma_C[:, k]).float()).item()
    print(
        names[k].ljust(20),
        str(round(e_lin, 2)).ljust(8),
        str(round(e_B, 2)).ljust(9),
        str(round(e_C, 2)).ljust(9),
        round(100 * inside),
        "%",
    )
