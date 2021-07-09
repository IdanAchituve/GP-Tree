from collections import namedtuple
import pypolyagamma
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from torch import nn
from GP_Tree.kernel_class import OneClassGPModel
import torch.nn.functional as F

from utils import *

SBGibbsState = namedtuple("SBGibbsState", ["omega", "f"])
SBModelState = namedtuple(
    "SBModelState",
    ["N", "N_sb", "mu", "K", "L", "Kinv", "Kinv_mu", "X", "Y", "C", "kappa"],
)


class GP_Model_Gibbs(nn.Module):
    def __init__(self,
                 kernel_func,
                 num_classes=2,
                 num_steps=10,
                 num_draws=20,
                 num_data=5,
                 num_gammas=100):

        super(GP_Model_Gibbs, self).__init__()
        self.num_classes = num_classes

        self.model = OneClassGPModel(kernel_func)
        self.ppg = pypolyagamma.PyPolyaGamma()
        self.num_data = num_data
        self.kernel_func = kernel_func

        self.num_steps = num_steps  # T in OVE paper (NS)
        self.num_draws = num_draws  # M in OVE paper (ND)
        self.num_gammas = num_gammas
        self.quadrature = GaussHermiteQuadrature1D()

    def to_one_hot(self, y, dtype):
        # convert a single label into a one-hot vector
        y_output_onehot = torch.zeros((y.shape[0], self.num_classes), dtype=dtype, device=y.device)
        return y_output_onehot.scatter_(1, y.unsqueeze(1), 1)

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

    def forward_mll(self, X, Y, to_print=True):

        model_state = self.fit(X, Y)
        gibbs_state = self.gibbs_sample(model_state)

        # average over classes and then average over the number of samples
        nmll = self.marginal_log_likelihood(gibbs_state.omega, model_state)

        if to_print:
            self.print_hyperparams()
            logging.info(f"Loss: {nmll.item():.5f}, Avg. Loss: {(nmll / self.num_data).item():.5f}")

        self.save_state(model_state, gibbs_state)

        return nmll / self.num_data

    def predictive_posterior(self, X_star):
        dist = self.predictive_dist(self.last_model_state, self.last_gibbs_state, X_star)
        probs = self.quadrature(torch.sigmoid, dist).mean(0)
        return probs

    def fit(self, X, Y):
        C = 1
        N = X.shape[0]

        mu, K = self.model(X)
        mu = mu.unsqueeze(1).type(X.dtype)  # N x 1
        Kinv = torch.inverse(K)
        Kinv_mu = Kinv.matmul(mu)
        L = psd_safe_cholesky(K)

        L = L.unsqueeze(0)
        K = K.unsqueeze(0)
        Kinv = Kinv.unsqueeze(0)
        Kinv_mu = Kinv_mu.unsqueeze(0)

        # stick break N vector: (ND * N) x 1
        Y_one_hot = 1 - self.to_one_hot(Y, dtype=X.dtype)
        N_sb = N_vec(Y_one_hot).repeat(self.num_draws, 1)
        kappa = kappa_vec(Y_one_hot)

        return SBModelState(
            N=N,
            N_sb=N_sb,
            mu=mu,
            K=K,
            L=L,
            Kinv_mu=Kinv_mu,
            Kinv=Kinv,
            X=X.clone(),
            Y=Y.clone(),
            C=C,
            kappa=kappa
        )

    def gibbs_sample(self, model_state):

        gibbs_state = self.initial_gibbs_state(model_state)

        # sample next state according to conditional posterior
        for _ in range(self.num_steps):
            gibbs_state = self.next_gibbs_state(model_state, gibbs_state)

        return gibbs_state

    def initial_gibbs_state(self, model_state):

        L = model_state.L
        N = model_state.N
        ND = self.num_draws

        # init to the prior mean
        SN = torch.normal(mean=torch.zeros(1 * ND * N, dtype=L.dtype, device=L.device),
                          std=torch.ones(1 * ND * N, dtype=L.dtype, device=L.device)).view(ND, N, 1)
        f_init = model_state.mu.unsqueeze(0) + L.matmul(SN)
        f_init = f_init.squeeze(-1)

        omega_init = self.sample_omega(f_init, model_state)

        return SBGibbsState(omega_init, f_init)

    def next_gibbs_state(self, model_state, gibbs_state):
        f_new = self.gaussian_conditional(gibbs_state.omega, model_state)
        omega_new = self.sample_omega(f_new, model_state)

        return SBGibbsState(omega_new, f_new)

    # P(ω | Y, f)
    def sample_omega(self, f, model_state):
        """"
        Sample from polya-gamma distribution.
        :parm c - number of observations per sample
        :return flattened array of samples of size C * N * ND
        """
        N = model_state.N

        b = detach_to_numpy(model_state.N_sb).reshape(-1).astype(np.double)
        c = detach_to_numpy(f).reshape(-1).astype(np.double)
        ret = np.zeros_like(c)  # fill with samples

        self.ppg.pgdrawv(b, c, ret)

        omega = torch.tensor(ret, dtype=f.dtype, device=f.device).view(self.num_draws, N)  # [ND, N]
        return omega

    # P(f | Y, ω, X)
    def gaussian_conditional(self, omega, model_state):
        kappa = model_state.kappa.t()
        Ω = torch.diag_embed(omega)

        # Set the precision for invalid points to zero
        Kinv_mu = model_state.Kinv_mu
        Kinv = model_state.Kinv

        sigma_tilde = torch.inverse(Kinv + Ω)
        # upper triangular of covariance matrices, each corresponds to different combination
        # of class and draw
        mu_tilde = sigma_tilde.matmul(kappa.unsqueeze(-1) + Kinv_mu).squeeze(-1)

        L_tilde = psd_safe_cholesky(sigma_tilde)
        fs = torch.distributions.MultivariateNormal(mu_tilde, scale_tril=L_tilde).rsample()
        return fs

    # ∑ (log(P(Y = c| ω, X)))
    def marginal_log_likelihood(self, omega, model_state):
        """
        Compute marginal likelihood with the given values of omega
        :param augmented_data:
        :return: log likelihood per class
        """
        kappa = model_state.kappa.t()  # 1 x N
        N = model_state.N
        N_sb = model_state.N_sb  # (ND * N) x 1
        K = model_state.K

        # prevents division by 0
        omega = omega.clamp(min=1e-16)

        # diagonal matrix for each combination of class, and number of draws [ND, N, N]
        Ω_inv = torch.diag_embed(1.0 / omega)
        z_Sigma = K + Ω_inv
        z_mu = model_state.mu.t()  # 1 x N

        # The "observations" are the effective mean of the Gaussian likelihood given omega
        # when omega is zero kappa should be zero as well
        z = kappa / omega
        L_z = psd_safe_cholesky(z_Sigma)
        p_y = torch.distributions.MultivariateNormal(z_mu, scale_tril=L_z)

        mll = p_y.log_prob(z) \
              + 0.5 * N * np.log(2 * np.pi) \
              - 0.5 * torch.log(omega).sum(-1) \
              + 0.5 * torch.sum((kappa.unsqueeze(1) ** 2) / omega, -1) \
              - torch.sum(N_sb.view(self.num_draws, N, -1) * np.log(2.0), dim=1).t()

        mll = - mll.mean()
        return mll

    # P(f^* | ω, Y, x^*, X)) & P(y^* | f^*)
    def predictive_dist(self, model_state, gibbs_state, X_star):
        """
        Compute predictive posterior mean and covariance
        :param X_star data of shape [L, n]
        :return: multinomial density and per class posterior distribution
        """
        omega = gibbs_state.omega  # ND x N
        kappa = model_state.kappa.t()  # 1 x N
        X = model_state.X
        z_mu = model_state.mu.t()  # 1 x N
        N = model_state.N

        Xon = torch.cat((X, X_star), dim=0)
        mu_on, K_on = self.model(Xon)

        # recompute kernel since it might have changed due to backward operation 1 x N x N
        K = K_on[:N, :N].unsqueeze(0)
        # covariance function between existing and new samples 1 x N x M
        K_s = K_on[:N, N:].unsqueeze(0)
        # covariance function between new samples 1 x M x M
        K_ss = K_on[N:, N:].unsqueeze(0)

        mu_s = mu_on[N:].unsqueeze(1).type(X_star.dtype)  # M x 1

        omega = omega.clamp(min=1e-16)
        z = kappa / omega - z_mu  # N x 1

        # diagonal matrix for each draw ND x N x N
        Ω_inv = torch.diag_embed(1.0 / omega)
        L_noisy = psd_safe_cholesky(K + Ω_inv)

        # mu^* + K^* x ((K + Ω^-1)^-1 x (Ω^-1 x kappa - mu))
        mu_pred = mu_s + K_s.permute(0, 2, 1).matmul(
            torch.cholesky_solve(z.unsqueeze(-1), L_noisy))
        mu_pred = mu_pred.squeeze(-1)  # ND x M

        Sigma_pred = torch.diagonal(K_ss - K_s.permute(0, 2, 1).
                                    matmul(torch.cholesky_solve(K_s, L_noisy)), dim1=1, dim2=2)
        Sigma_pred = torch.diag_embed(Sigma_pred)  # [ND, M, M]
        L_s = psd_safe_cholesky(Sigma_pred)
        dist = torch.distributions.MultivariateNormal(mu_pred, scale_tril=L_s)
        return dist

    def save_state(self, model_state, gibbs_state):

        C = model_state.C
        N = model_state.N

        # mean function - initialized to zero or empirical value for all classes [N x (C-1)]
        mu = model_state.mu.detach().clone()

        # covariance function C x 1 x N x N
        K = model_state.K.detach().clone()
        # stick break N vector: (ND * N) x C
        N_sb = model_state.N_sb.detach().clone()

        L = model_state.L.detach().clone()
        Kinv = model_state.Kinv.detach().clone()
        Kinv_mu = model_state.Kinv_mu.detach().clone()
        kappa = model_state.kappa.detach().clone()

        self.last_model_state = SBModelState(
                                    N=N,
                                    N_sb=N_sb,
                                    mu=mu,
                                    K=K,
                                    L=L,
                                    Kinv=Kinv,
                                    Kinv_mu=Kinv_mu,
                                    X=model_state.X.clone(),
                                    Y=model_state.Y.clone(),
                                    C=C,
                                    kappa=kappa
                                )
        self.last_gibbs_state = SBGibbsState(gibbs_state.omega.detach().clone(),
                                             gibbs_state.f.detach().clone())