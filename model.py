import torch
import math

# Notation
d = 21  # numarul de statii (dimensiunea lui x)
H = 128  # numarul de neuroni ascunsi
L = 10  # dimensiunea spatiului latent

# Encoder
W_h = torch.randn(H, d) * 0.1  # 128 x 21: 128 neuroni, fiecare cu 21 de ponderi
b_h = torch.zeros(H)  # 128: un bias pentru fiecare neuron
W_mu = torch.randn(L, H) * 0.1  # 10 x 128: 10 iesiri, fiecare cu 128 de ponderi
b_mu = torch.zeros(L)  # 10: un bias pentru fiecare iesire
W_sigma = torch.randn(L, H) * 0.1  # 10 x 128: la fel ca W_mu
b_sigma = torch.zeros(L)  # 10

# Decoder
W_3 = torch.randn(H, L) * 0.1  # 128 x 10: 128 neuroni, fiecare cu 10 ponderi
b_3 = torch.zeros(H)  # 128: un bias pentru fiecare neuron
W_4mu = torch.randn(d, H) * 0.1  # 21 x 128: 21 statii, fiecare cu 128 de ponderi
b_4mu = torch.zeros(d)  # 21: un bias pentru fiecare statie
W_4sigma = torch.randn(d, H) * 0.1  # 21 x 128: la fel ca W_4mu
b_4sigma = torch.zeros(d)  # 21

# mark the parameters: PyTorch will compute their gradient
W_h.requires_grad_()
b_h.requires_grad_()
W_mu.requires_grad_()
b_mu.requires_grad_()
W_sigma.requires_grad_()
b_sigma.requires_grad_()
W_3.requires_grad_()
b_3.requires_grad_()
W_4mu.requires_grad_()
b_4mu.requires_grad_()
W_4sigma.requires_grad_()
b_4sigma.requires_grad_()

# all 12 parameters in one list (the optimizer needs it)
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


def encoder(X):
    a = X @ W_h.T + b_h
    h = torch.relu(a)
    mu_z = h @ W_mu.T + b_mu
    logvar_z = h @ W_sigma.T + b_sigma
    return mu_z, logvar_z


# Reparameterization: (mu_z, logvar_z) -> z
def reparameterize(mu_z, logvar_z):
    sigma_z = torch.exp(0.5 * logvar_z)  # sigma_z = exp(1/2 * log sigma_z^2)
    eps = torch.randn_like(sigma_z)  # eps ~ N(0, I)
    z = mu_z + sigma_z * eps  # z = mu_z + sigma_z * eps
    return z


# Decoder function: z -> (mu_x, logvar_x)
def decoder(z):
    a_dec = z @ W_3.T + b_3  # partea liniara
    h_dec = torch.relu(a_dec)  # h' = ReLU(W_3 z + b_3)
    mu_x = h_dec @ W_4mu.T + b_4mu  # (2) ora reconstruita
    logvar_x = h_dec @ W_4sigma.T + b_4sigma  # (3) incertitudin
    return (mu_x, logvar_x)


# Loss function: L = Recon + beta * KL, averaged over the batch
def total_loss(X, mu_x, logvar_x, mu_z, logvar_z, beta=1.0):
    # Recon for each hour: sum over the 21 stations
    recon = 0.5 * torch.sum(
        math.log(2 * math.pi) + logvar_x + (X - mu_x) ** 2 / torch.exp(logvar_x), dim=1
    )
    # KL for each hour: sum over the 10 latent directions
    kl = 0.5 * torch.sum(mu_z**2 + torch.exp(logvar_z) - logvar_z - 1, dim=1)
    # total loss: average over the hours in the batch
    loss = torch.mean(recon + beta * kl)
    return (
        loss,
        torch.mean(recon),
        torch.mean(kl),
    )  # total, Recon, KL (each averaged over the hours)


if __name__ == "__main__":
    X_test = torch.randn(4, d)
    mu_z, logvar_z = encoder(X_test)
    z = reparameterize(mu_z, logvar_z)
    mu_x, logvar_x = decoder(z)
    loss = total_loss(X_test, mu_x, logvar_x, mu_z, logvar_z)
    print("mu_z:", mu_z.shape)
    print("z:", z.shape)
    print("mu_x:", mu_x.shape)
    print("loss:", loss.item())
