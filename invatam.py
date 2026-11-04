import torch
import math

# Notation
d = 21  # numarul de statii (dimensiunea lui x)
H = 128  # numarul de neuroni ascunsi
L = 10  # dimensiunea spatiului latent

# (1) stratul ascuns: h = ReLU(W_h x + b_h)
W_h = torch.randn(H, d) * 0.1  # 128 x 21: 128 neuroni, fiecare cu 21 de ponderi
b_h = torch.zeros(H)  # 128: un bias pentru fiecare neuron

# (2) media: mu_z = W_mu h + b_mu
W_mu = torch.randn(L, H) * 0.1  # 10 x 128: 10 iesiri, fiecare cu 128 de ponderi
b_mu = torch.zeros(L)  # 10: un bias pentru fiecare iesire

# (3) log-varianta: log sigma^2 = W_sigma h + b_sigma
W_sigma = torch.randn(L, H) * 0.1  # 10 x 128: la fel ca W_mu
b_sigma = torch.zeros(L)  # 10

# ENCODERUL: x -> h -> mu_z, logvar_z -> z
x = torch.randn(d)
a = W_h @ x + b_h
h = torch.relu(a)
mu_z = W_mu @ h + b_mu
logvar_z = W_sigma @ h + b_sigma
sigma_z = torch.exp(0.5 * logvar_z)  # sigma_z = exp(1/2 * log sigma_z^2)
eps = torch.randn(L)  # eps ~ N(0, I)
z = mu_z + sigma_z * eps  # z = mu_z + sigma_z * eps


# KL = 1/2 * suma_k (mu_k^2 + sigma_k^2 - log sigma_k^2 - 1)
kl = 0.5 * torch.sum(mu_z**2 + sigma_z**2 - logvar_z - 1)

# DECODER
# (1) stratul ascuns: h' = ReLU(W_3 z + b_3)
W_3 = (
    torch.randn(H, L) * 0.1
)  # 128 x 10: 128 neuroni, fiecare cu 10 ponderi (una pe directie latenta)
b_3 = torch.zeros(H)  # 128: un bias pentru fiecare neuron

# (2) media: mu_x = W_4mu h' + b_4mu
W_4mu = torch.randn(d, H) * 0.1  # 21 x 128: 21 statii, fiecare cu 128 de ponderi
b_4mu = torch.zeros(d)  # 21: un bias pentru fiecare statie

# (3) log-varianta: log sigma_x^2 = W_4sigma h' + b_4sigma
W_4sigma = torch.randn(d, H) * 0.1  # 21 x 128: la fel ca W_4mu
b_4sigma = torch.zeros(d)  # 21

# Decoder hidden layer
a_dec = W_3 @ z + b_3  # partea liniara
h_dec = torch.relu(a_dec)  # h' = ReLU(W_3 z + b_3)
mu_x = W_4mu @ h_dec + b_4mu  # (2) ora reconstruita
logvar_x = W_4sigma @ h_dec + b_4sigma  # (3) incertitudinea pe fiecare statie

# Recon = 1/2 * suma_j [ log(2*pi) + log sigma_x,j^2 + (x_j - mu_x,j)^2 / sigma_x,j^2 ]
recon = 0.5 * torch.sum(
    math.log(2 * math.pi) + logvar_x + (x - mu_x) ** 2 / torch.exp(logvar_x)
)

# Total cost
beta = 1.0  # cat de mult conteaza KL (1 = VAE standard)
loss = recon + beta * kl  # L = Recon + beta * KL

# verif encoder
# print(W_h.shape, b_h.shape)
# print(W_mu.shape, b_mu.shape)
# print(W_sigma.shape, b_sigma.shape)
# print(x.shape)
# print(h.shape)
##rint(logvar_z.shape)
# print(logvar_z.shape)
# print("KL:", kl)

# decoder verif
print("W_3:", W_3.shape, " b_3:", b_3.shape)
print("W_4mu:", W_4mu.shape, " b_4mu:", b_4mu.shape)
print("W_4sigma:", W_4sigma.shape, " b_4sigma:", b_4sigma.shape)
print("h_dec:", h_dec.shape)
print("mu_x:", mu_x.shape)
print("logvar_x:", logvar_x.shape)
print("Recon:", recon.item())
print("KL:", kl.item())
print("Loss:", loss.item())
