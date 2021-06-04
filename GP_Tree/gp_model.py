from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from torch import nn
from torch.distributions import MultivariateNormal
from utils import *
from GP_Tree.kernel_class import OneClassGPModel


class Likelihood(nn.Module):
    def __init__(self):
        super().__init__()
        self.quadrature = GaussHermiteQuadrature1D()

    def forward(self, function_samples):
        return torch.sigmoid(function_samples)

    def expected_log_prob(self, function_dist):
        log_prob = self.quadrature(F.logsigmoid, function_dist)
        return log_prob


def _triangular_inverse(A, upper=False):
    eye = torch.eye(A.size(-1), dtype=A.dtype, device=A.device)
    return eye.triangular_solve(A, upper=upper).solution


class VariationalELBO(nn.Module):

    def __init__(self, model, num_data, num_inducing_points, lr=0.1, dtype=torch.float64):
        super().__init__()
        self.model = model
        # local parameters
        self.register_buffer("c", torch.ones(num_data, dtype=dtype))
        self.lr = lr
        self.num_data = num_data

        # global parameters
        self.num_inducing_points = num_inducing_points
        scaled_mean_init = torch.zeros(self.num_inducing_points, dtype=dtype)
        neg_prec_init = torch.eye(self.num_inducing_points, self.num_inducing_points, dtype=dtype).mul(-0.5)
        # eta and H parameterization of the variational distribution
        self.register_buffer("eta", scaled_mean_init)
        self.register_buffer("H", neg_prec_init)

    def forward(self, K, Y, batch_idx):

        mu, Sigma = self.NaturalToMuSigma()
        c = self.c[batch_idx]  # take only batch variables

        Kmm = K[:self.num_inducing_points, :self.num_inducing_points]
        Knm = K[:self.num_inducing_points, self.num_inducing_points:].t()
        Knn = K[self.num_inducing_points:, self.num_inducing_points:]

        L = psd_safe_cholesky(Kmm)
        kappa = torch.cholesky_solve(Knm.t(), L).t()
        theta = 1./(2. * c.unsqueeze(0)) * torch.tanh(c.unsqueeze(0) / 2.)

        logDetSigma = torch.logdet(Sigma)
        logDetKmm = torch.logdet(Kmm)
        TraceKmmInvSigma = torch.trace(torch.cholesky_solve(Sigma, L))
        MuKmmInvMu = mu.unsqueeze(0).matmul(torch.cholesky_solve(mu.unsqueeze(0).t(), L)).squeeze()

        YKappaMu = (Y.unsqueeze(0) @ kappa @ mu.unsqueeze(1)).squeeze()

        K_tilde = torch.diagonal(Knn - Knm.matmul(kappa.t())).unsqueeze(1)
        kappaSigmakappa = torch.diagonal(kappa @ (Sigma @ kappa.t())).unsqueeze(1)
        kappaMu = kappa @ mu.unsqueeze(1)
        muKappa_square = kappaMu.pow(2)
        c_square = c.unsqueeze(1).pow(2)
        theta_summation = (theta @ (K_tilde - kappaSigmakappa - muKappa_square - c_square)).squeeze()

        logCosc = 2 * torch.log(torch.cosh(c / 2.)).sum()

        self.ctx = (Kmm.detach(), kappa.detach(), Y, batch_idx, K_tilde.detach(),
                    kappaSigmakappa.detach(), muKappa_square.detach())
        return 1/2 * (logDetSigma - logDetKmm - TraceKmmInvSigma - MuKmmInvMu + YKappaMu - theta_summation - logCosc)

    def NaturalToMuSigma(self):
        L_inv = psd_safe_cholesky(-2.0 * self.H)
        L = _triangular_inverse(L_inv, upper=False)
        S = L.transpose(-1, -2) @ L
        mu = (S @ self.eta.unsqueeze(-1)).squeeze(-1)
        return mu, S

    def update(self):
        Kmm, kappa, Y, batch_idx, K_tilde, kappaSigmakappa, muKappa_square = self.ctx
        with torch.no_grad():

            s = Y.shape[0]
            c = torch.sqrt(K_tilde + kappaSigmakappa + muKappa_square).squeeze()

            self.c[batch_idx] = c  # update only batch variables
            if c.dim() == 0:
                c = c.unsqueeze(0)

            # update eta
            self.eta += self.lr * ((self.num_data / (2 * s)) * (kappa.t() @ Y.unsqueeze(1)).squeeze() - self.eta)

            # update H
            L_inv = psd_safe_cholesky(Kmm)
            L = _triangular_inverse(L_inv, upper=False)
            Kmm_inv = L.transpose(-1, -2) @ L
            theta = torch.diag_embed((1. / (2. * c.unsqueeze(0)) * torch.tanh(c.unsqueeze(0) / 2.)).squeeze(0))
            self.H += self.lr * ((-1 / 2) * (Kmm_inv + (self.num_data / s) * kappa.t() @ (theta @ kappa)) - self.H)


class GP_Model(nn.Module):
    def __init__(self,
                 kernel_func,
                 num_classes=2,
                 dtype=torch.float64,
                 num_inducing_points=10,  # number of inducing points to use
                 num_data=100,
                 natural_lr=0.1):

        super(GP_Model, self).__init__()
        self.num_classes = num_classes
        self.num_inducing_points = num_inducing_points  # number of inducing points per class
        self.kernel_func = kernel_func

        self.dtype = dtype
        self.loss_fn = nn.CrossEntropyLoss()

        self.model = OneClassGPModel(kernel_func, jitter_val=1e-2)
        self.ELBO = VariationalELBO(self.model, num_data, num_inducing_points, natural_lr, dtype)
        self.likelihood = Likelihood()

    def print_hyperparams(self):
        logging.info(f"output scale: "
                     f"{np.round_(detach_to_numpy(self.model.covar_module.outputscale.squeeze()), decimals=2)}")
        if self.kernel_func == "RBFKernel":
            lengthscale = detach_to_numpy(self.model.covar_module.base_kernel.lengthscale.squeeze())
            logging.info(f"length scale: "
                         f"{np.round_(lengthscale, decimals=2)}")
        elif self.kernel_func == "LinearKernel":
            variance = detach_to_numpy(self.model.covar_module.base_kernel.variance.squeeze())
            logging.info(f"variance: "
                         f"{np.round_(variance, decimals=2)}")

    def forward_mll(self, Points, Y, batch_idx, to_print):
        Y_n = Y.mul(2).sub(1).to(self.dtype)  # [0, 1] -> [-1, 1]
        _, K = self.model(Points)
        mll = self.ELBO(K, Y_n, batch_idx)
        if to_print:
            self.print_hyperparams()
        return mll

    def predictive_posterior(self, Points, jitter_Kmm=False):

        _, K = self.model(Points)
        I = torch.eye(self.num_inducing_points, dtype=K.dtype, device=K.device)

        mu, Sigma = self.ELBO.NaturalToMuSigma()
        Kmm = K[:self.num_inducing_points, :self.num_inducing_points]
        if jitter_Kmm:
            Kmm += 0.03 * I
        Knm = K[:self.num_inducing_points, self.num_inducing_points:].t()
        Knn = K[self.num_inducing_points:, self.num_inducing_points:]
        L = psd_safe_cholesky(Kmm)
        mu_s = Knm.matmul(torch.cholesky_solve(mu.unsqueeze(0).t(), L)).squeeze()
        Sigma_s = torch.diagonal(Knn + Knm.matmul(torch.cholesky_solve(
                                                 torch.cholesky_solve(Sigma, L).t() - I, L)
                                                 ).matmul(Knm.t())
                                 )

        L_s = psd_safe_cholesky(torch.diag_embed(Sigma_s))
        dist = MultivariateNormal(mu_s, scale_tril=L_s)
        log_probs = self.likelihood.expected_log_prob(dist)
        return torch.exp(log_probs)