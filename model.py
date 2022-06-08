import torch
from torch import nn
import torch.distributions as dists
import numpy as np

MNIST_SIZE = 28

class IWAE(nn.Module):
    def __init__(self, k=1):
        super(IWAE, self).__init__()
        self.k = k
        self.q1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(MNIST_SIZE**2, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
        )
        self.q1_mu = nn.Linear(200, 100)
        self.q1_log_std = nn.Linear(200, 100)

        self.q2 = nn.Sequential(
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
        )

        self.q2_mu = nn.Linear(100, 50)
        self.q2_log_std = nn.Linear(100, 50)

        self.p1 = nn.Sequential(
            nn.Linear(50, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
        )
        self.p1_mu = nn.Linear(100, 100)
        self.p1_log_std = nn.Linear(100, 100)

        self.p0 = nn.Sequential(
            nn.Linear(100, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, MNIST_SIZE**2),
            nn.Sigmoid()
        )


    def sample(self, mu, std):
        eps = torch.normal(torch.zeros_like(std), torch.ones_like(std))
        x = mu + eps * std + 1e-6
        return x

    def forward(self, x):
        self.batch = x.shape[0]
        x = torch.repeat_interleave(x, self.k, axis=0)
        # Encode
        l1 = self.q1(x)
        mu_q1, std_q1 = self.q1_mu(l1), torch.exp(self.q1_log_std(l1))
        q1 = self.sample(mu_q1, std_q1)
        l2 = self.q2(q1)
        mu_q2, std_q2 = self.q2_mu(l2), torch.exp(self.q2_log_std(l2))
        q2 = self.sample(mu_q2, std_q2)
        # Decode
        l3 = self.p1(q2)
        mu_p1, std_p1 = self.p1_mu(l3), torch.exp(self.p1_log_std(l3))
        mu_p0 = self.p0(q1)

        dparams = [q1, mu_q1, std_q1, q2, mu_q2, std_q2, mu_p1, std_p1, mu_p0]
        dparams = [i.reshape(self.batch, self.k, -1) for i in dparams]
        return dparams

    def compute_ELBO(self, data, dparams):
        q1, mu_q1, std_q1, q2, mu_q2, std_q2, mu_p1, std_p1, mu_p0 = dparams
        log_q_q1_g_x = dists.Normal(mu_q1, std_q1).log_prob(q1).sum(axis=-1)
        log_q_q2_g_q1 = dists.Normal(mu_q2, std_q2).log_prob(q2).sum(axis=-1)
        log_prior = dists.Normal(0, 1).log_prob(q2).sum(axis=-1)
        log_p_q1_g_q2 = dists.Normal(mu_p1, std_p1).log_prob(q1).sum(axis=-1)
        log_p_x_g_q1 = dists.Bernoulli(mu_p0).log_prob(data).sum(axis=-1)
        log_w = log_prior + log_p_q1_g_q2 + log_p_x_g_q1 - log_q_q1_g_x - log_q_q2_g_q1
        return log_w

    def compute_loss(self, data, dparams):
        data = torch.repeat_interleave(data, self.k, axis=0).reshape(self.batch, self.k, -1)
        log_w = self.compute_ELBO(data, dparams)
        vae_elbo = torch.mean(log_w)
        iwae_elbo = torch.mean(torch.logsumexp(log_w, 1) - np.log(self.k))

        return {"vae": -vae_elbo,
                "iwae": -iwae_elbo
                }
