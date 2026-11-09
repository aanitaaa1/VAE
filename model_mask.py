import torch  # biblioteca PyTorch
import math  # pentru log(2*pi)

# =====================================================
# NOTATII
# =====================================================
d = 21  # numarul de statii
H = 128  # numarul de neuroni ascunsi
L = 10  # dimensiunea spatiului latent

# =====================================================
# PARAMETRII
# =====================================================
# encoder
W_h = torch.randn(H, 2 * d) * 0.1  # NOU: 128 x 42 (21 de valori + 21 de biti ai mastii)
b_h = torch.zeros(H)  # 128
W_mu = torch.randn(L, H) * 0.1  # 10 x 128
b_mu = torch.zeros(L)  # 10
W_sigma = torch.randn(L, H) * 0.1  # 10 x 128
b_sigma = torch.zeros(L)  # 10

# decoder (neschimbat)
W_3 = torch.randn(H, L) * 0.1  # 128 x 10
b_3 = torch.zeros(H)  # 128
W_4mu = torch.randn(d, H) * 0.1  # 21 x 128
b_4mu = torch.zeros(d)  # 21
W_4sigma = torch.randn(d, H) * 0.1  # 21 x 128
b_4sigma = torch.zeros(d)  # 21

# lista cu toti parametrii + ii marcam pentru gradient
params = [
    W_h,
    b_h,
    W_mu,
    b_mu,
    W_sigma,
    b_sigma,
    W_3,
    b_3,
    W_4mu,
    b_4mu,
    W_4sigma,
    b_4sigma,
]
for p in params:
    p.requires_grad_()


# =====================================================
# ENCODER: [x_tilde, masca] -> (mu_z, logvar_z)
# =====================================================
def encoder(X_tilde, mask):
    inp = torch.cat(
        [X_tilde, mask], dim=1
    )  # lipim valorile si masca -> fiecare ora are 42 de numere
    h = torch.relu(inp @ W_h.T + b_h)  # stratul ascuns: B x 128
    mu_z = h @ W_mu.T + b_mu  # media: B x 10
    logvar_z = h @ W_sigma.T + b_sigma  # log-varianta: B x 10
    return mu_z, logvar_z


# =====================================================
# ESANTIONAREA: (mu_z, logvar_z) -> z
# =====================================================
def reparameterize(mu_z, logvar_z):
    sigma_z = torch.exp(0.5 * logvar_z)
    eps = torch.randn_like(sigma_z)
    z = mu_z + sigma_z * eps
    return z


# =====================================================
# DECODER: z -> (mu_x, logvar_x)
# =====================================================
def decoder(z):
    h_dec = torch.relu(z @ W_3.T + b_3)
    mu_x = h_dec @ W_4mu.T + b_4mu
    logvar_x = h_dec @ W_4sigma.T + b_4sigma
    return mu_x, logvar_x


# =====================================================
# COSTUL: L = Recon + beta * KL
# =====================================================
def total_loss(X, mu_x, logvar_x, mu_z, logvar_z, beta=1.0):
    recon = 0.5 * torch.sum(
        math.log(2 * math.pi) + logvar_x + (X - mu_x) ** 2 / torch.exp(logvar_x), dim=1
    )
    kl = 0.5 * torch.sum(mu_z**2 + torch.exp(logvar_z) - logvar_z - 1, dim=1)
    loss = torch.mean(recon + beta * kl)
    return loss, torch.mean(recon), torch.mean(kl)


# =====================================================
# TEST (ruleaza doar cu "python model_mask.py")
# =====================================================
if __name__ == "__main__":
    X_test = torch.randn(4, d)
    mask = (torch.rand(4, d) > 0.15).float()
    mu_z, logvar_z = encoder(X_test * mask, mask)
    z = reparameterize(mu_z, logvar_z)
    mu_x, logvar_x = decoder(z)
    loss, recon, kl = total_loss(X_test, mu_x, logvar_x, mu_z, logvar_z)
    print("mask of the first hour:", mask[0])
    print("mu_z:", mu_z.shape)
    print("mu_x:", mu_x.shape)
    print("loss:", loss.item())
