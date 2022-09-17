import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import tqdm
import autokoopman.core.system as ksys
import autokoopman.core.estimator as kest
import autokoopman.core.trajectory as ktraj


class DeepKoopman(kest.NextStepEstimator):
    r"""
    Deep Learning Koopman with Inputs

    This is a simple implementation of an AutoEncoder architecture to learn the Koopman observables and
    operator. It allows an input space, which is directly injected into the observables space (like KIC).
    To improve a good fit, a *metric loss* is introduced to make the distances between states approximately
    preserve in the observables space.

    The loss is defined as

    .. math::
        \mathcal L(\mathbf x, \mathbf x_r,\mathbf x', \mathbf x'_r, \mathbf y, \mathbf y') = \left\| \mathbf x-\mathbf x_r \right\|_2^2 + \lambda_p \left\| \mathbf x'-\mathbf x'_r \right\|_2^2 + \lambda_m \left| \left\| \mathbf y' - \mathbf y \right\|_2^2 -  \left\| \mathbf x' - \mathbf x \right\|_2^2 \right|

    where :math:`\mathbf x` is the current state, :math:`\mathbf x_r` is the reconstructed current state,
    :math:`\mathbf x'` is the next state, :math:`\mathbf x'_r` is the reconstructed state,
    :math:`\mathbf y` is the current observable, and :math:`\mathbf y'` is the next observable.

    TODO: implement normalizers

    :param state_dim: system state space dimension
    :param input_dim: system input space dimension
    :param hidden_dim: observables space dimension
    :param pred_loss_weight: weighting factor for prediction loss (of total loss)
    :param metric_loss_weight: weighting factor for metric loss (of total loss)
    :param hidden_enc_dim: dimension of hidden layers in the encoder/decoder
    :param max_iter: maximum number of iterations
    :param lr: learning rate
    :param validation_data: additional uniform time trajectories to validate at each iteration
    :param encoder_module: PyTorch Module Encoder (pass in externally)
    :param decoder_module: PyTorch Module Decoder (pass in externally)
    :param torch_device: device to run PyTorch (if None, it will attempt to use cuda:0)

    References
        Li, Y., He, H., Wu, J., Katabi, D., & Torralba, A. (2019).
        Learning compositional koopman operators for model-based control.
        arXiv preprint arXiv:1910.08264.
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        hidden_dim: int,
        pred_loss_weight: float = 1.0,
        metric_loss_weight: float = 0.2,
        hidden_enc_dim: int = 32,
        max_iter: int = 500,
        lr: float = 1e-3,
        validation_data: Optional[ktraj.UniformTimeTrajectoriesData] = None,
        num_hidden_layers: int = 1,
        encoder_module: Optional[nn.Module] = None,
        decoder_module: Optional[nn.Module] = None,
        torch_device: Optional[str] = None,
    ):
        self.max_iter = max_iter
        self.lr = lr
        self.validation_data = validation_data

        self.has_input = input_dim > 0

        if torch_device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch_device

        self.hidden_dim = hidden_dim

        hidden_layers = [
            nn.Linear(hidden_enc_dim, hidden_enc_dim),
            nn.PReLU(),
        ] * num_hidden_layers

        if encoder_module is None:
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_enc_dim),
                nn.PReLU(),
                *(hidden_layers),
                nn.Linear(hidden_enc_dim, hidden_dim),
                nn.Tanh(),
            )
        else:
            self.encoder = encoder_module

        if decoder_module is None:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_enc_dim),
                nn.PReLU(),
                *(
                    [nn.Linear(hidden_enc_dim, hidden_enc_dim), nn.PReLU()]
                    * num_hidden_layers
                ),
                nn.Linear(hidden_enc_dim, state_dim),
                nn.Tanh(),
            )
        else:
            self.decoder = decoder_module

        self.propagate = nn.Linear(hidden_dim + input_dim, hidden_dim, bias=False)

        # move the modules to the target device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.propagate.to(self.device)

        # TODO
        self.loss_hist = None
        self.metric_loss_weight = metric_loss_weight
        self.pred_loss_weight = pred_loss_weight

    def forward(self, x, u):
        """
        NN Forward Function

        :param x: current system state
        :param u: current system input
        """
        # get current step observable
        y = self.encoder(x)

        # get next step observable
        if u is not None:
            yn = self.propagate(torch.cat((y, u), axis=-1))
        else:
            yn = self.propagate(y)

        # get next step state
        xn = self.decoder(yn)

        # get x reconstruction
        xr = self.decoder(y)

        return x, xn, y, yn, xr

    @property
    def system_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """get the linear system matrices :math:`\mathbf x_{n+1} = A \mathbf x_n + B \mathbf u_n`

        :returns: A, B
        """
        weight = self.propagate.weight.data.numpy()
        A = weight[:, : self.hidden_dim]
        B = weight[:, self.hidden_dim :]
        return A, B

    @property
    def model(self) -> ksys.System:
        """
        packs the autoencoder into a system object
        """

        def step_func(t, x, i):
            with torch.no_grad():
                tx = torch.tensor(x, dtype=torch.float32).to(self.device)
                if self.has_input:
                    tu = torch.tensor(i, dtype=torch.float32).to(self.device)
                    tu /= self.u_mult
                else:
                    tu = None
                tx = (tx - self.xmean) / self.xstd
                _, xn, _, _, _ = self.forward(tx, tu)
                xn = xn * self.xstd + self.xmean
                return xn.cpu().detach().numpy()

        return ksys.StepDiscreteSystem(step_func, self.names)

    def fit_next_step(
        self, X: np.ndarray, Y: np.ndarray, U: Optional[np.ndarray] = None, **kwargs
    ) -> None:
        """fits the discrete system model

        :param X: snapshot of states
        :param Y: snapshot of next states
        :param U: snapshot of inputs corresponding to X
        """
        tX = torch.tensor(X.T, dtype=torch.float32).to(self.device)
        tY = torch.tensor(Y.T, dtype=torch.float32).to(self.device)
        tU = (
            torch.tensor(U.T, dtype=torch.float32).to(self.device)
            if self.has_input
            else None
        )
        self.train(tX, tY, tU, **kwargs)

    def train(
        self,
        X: torch.Tensor,
        Xn: torch.Tensor,
        U: Optional[torch.Tensor],
    ):
        """
        Train the AutoEncoder

        Internally updates the loss_hist field to record training information.

        TODO: implement early stopping

        :param X: snapshot of states
        :param Xn: snapshot of next states
        :param U: snapshot of inputs corresponding to X
        """
        mseloss = nn.MSELoss()
        l1loss = nn.L1Loss()

        mselossv = nn.MSELoss()
        l1lossv = nn.L1Loss()

        self.x_mult = torch.max(torch.abs(X))
        self.xmean = torch.mean(X, dim=0)
        self.xstd = self.x_mult  # torch.std(X, dim=0)

        nX = (X - self.xmean) / self.xstd
        nXn = (Xn - self.xmean) / self.xstd

        if self.has_input:
            self.u_mult = torch.max(torch.abs(U))
            nU = U / self.u_mult
        else:
            nU = None

        # upload validation data to device if needed
        if self.validation_data is not None:
            # transpose necessary to change construction
            _Xv, _Xnv, _Uv = self.validation_data.next_step_matrices
            Xv = torch.tensor(_Xv.T, dtype=torch.float32).to(self.device)
            Xnv = torch.tensor(_Xnv.T, dtype=torch.float32).to(self.device)
            nXv = (Xv - self.xmean) / self.xstd
            nXnv = (Xnv - self.xmean) / self.xstd
            if self.has_input:
                Uv = torch.tensor(_Uv.T, dtype=torch.float32).to(self.device)
                nUv = Uv / self.u_mult
            else:
                nUv = None
        else:
            nXv = None
            nUv = None

        def _get_loss(mseloss, l1loss, x, xn, y, yn, xr, Xn):
            # reconstruction and prediction loss
            ae_loss = mseloss(x, xr)
            pred_loss = mseloss(xn, Xn)

            # linearity loss
            with torch.no_grad():
                lin_loss = mseloss(yn, self.encoder(Xn))

            # largest loss
            inf_loss = torch.max(torch.abs(x - xr)) + torch.max(torch.abs(Xn - xn))

            # metric loss
            metric_loss = l1loss(torch.norm(yn - y, dim=1), torch.norm(xn - x, dim=1))

            # frobenius norm of operator
            weight_loss = 0
            for l in self.encoder:
                if isinstance(l, nn.Linear):
                    weight_loss += torch.norm(l.weight.data)
            for l in self.decoder:
                if isinstance(l, nn.Linear):
                    weight_loss += torch.norm(l.weight.data)

            total_loss = (
                (ae_loss + pred_loss)
                + self.pred_loss_weight * lin_loss
                + 1e-2 * inf_loss
                + 1e-4 * weight_loss
                # + self.metric_loss_weight * metric_loss
            )
            return {
                "total_loss": total_loss,
                "recon_loss": ae_loss,
                "pred_loss": pred_loss,
                "lin_loss": lin_loss,
                "inf_loss": inf_loss,
                "weight_loss": weight_loss,
                "metric_loss": metric_loss,
            }

        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.lr)
        propagate_optimizer = torch.optim.Adam(self.propagate.parameters(), lr=self.lr)

        loss_hist = {}

        for _ in tqdm.tqdm(range(self.max_iter)):
            # get risk (loss)
            x, xn, y, yn, xr = self.forward(nX, nU)
            risk = _get_loss(mseloss, l1loss, x, xn, y, yn, xr, nXn)
            for k, v in risk.items():
                if k in loss_hist:
                    loss_hist[k].append(v.cpu().detach().numpy())
                else:
                    loss_hist[k] = [v.cpu().detach().numpy()]

            # optimize
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            propagate_optimizer.zero_grad()

            risk["total_loss"].backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
            propagate_optimizer.step()

            # do validation if applicable
            if self.validation_data is not None:
                with torch.no_grad():
                    x, xn, y, yn, xr = self.forward(nXv, nUv)
                    loss = _get_loss(mselossv, l1lossv, x, xn, y, yn, xr, nXnv)

                    # add to the loss history with validation prefix
                    for k, v in loss.items():
                        k = f"validation_{k}"
                        if k in loss_hist:
                            loss_hist[k].append(v.cpu().detach().numpy())
                        else:
                            loss_hist[k] = [v.cpu().detach().numpy()]

        self.loss_hist = loss_hist
