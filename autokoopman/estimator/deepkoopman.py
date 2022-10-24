import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import tqdm
import autokoopman.core.system as ksys
import autokoopman.core.estimator as kest
import autokoopman.core.trajectory as ktraj


class DeepKoopman(kest.TrajectoryEstimator):
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

    :param state_dim: system state space dimension
    :param input_dim: system input space dimension
    :param hidden_dim: observables space dimension
    :param pred_loss_weight: weighting factor for prediction loss (of total loss)
    :param metric_loss_weight: weighting factor for metric loss (of total loss)
    :param max_loss_weight: weighting factor against max loss (0 if outliers)
    :param weight_decay_weight: weight decay
    :param hidden_enc_dim: dimension of hidden layers in the encoder/decoder
    :param max_iter: maximum number of iterations
    :param lr: learning rate
    :param validation_data: additional uniform time trajectories to validate at each iteration
    :param rollout_steps: number of rollout steps
    :param encoder_module: PyTorch Module Encoder (pass in externally)
    :param decoder_module: PyTorch Module Decoder (pass in externally)
    :param torch_device: device to run PyTorch (if None, it will attempt to use cuda:0)
    :param display_progress: show progress bar
    :param verbose: set verbosity

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
        pred_loss_weight: float = 0.5,
        metric_loss_weight: float = 1e-5,
        max_loss_weight: float = 1e-5,
        weight_decay_weight: float = 1e-7,
        hidden_enc_dim: int = 32,
        max_iter: int = 500,
        lr: float = 1e-3,
        validation_data: Optional[ktraj.UniformTimeTrajectoriesData] = None,
        num_hidden_layers: int = 1,
        rollout_steps: int = 4,
        encoder_module: Optional[nn.Module] = None,
        decoder_module: Optional[nn.Module] = None,
        torch_device: Optional[str] = None,
        display_progress: bool = True,
        verbose: bool = True,
    ):
        hidden_enc_dim = int(hidden_enc_dim)
        num_hidden_layers = int(num_hidden_layers)
        hidden_dim = int(hidden_dim)

        self.names = None
        self.max_iter = max_iter
        self.lr = lr
        self.validation_data = validation_data
        self.verbose = verbose
        self.disp_progress = display_progress

        self.has_input = input_dim > 0

        if torch_device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if self.verbose:
                print(f"DeepKoopman is using torch device '{self.device}'")
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
        self.max_loss_weight = max_loss_weight
        self.weight_decay_weight = weight_decay_weight
        self.rollout_steps = rollout_steps

    def forward(self, x, u, step_size):
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
        if step_size > 1:
            for _ in range(step_size - 1):
                if u is not None:
                    yn = self.propagate(torch.cat((yn, u), axis=-1))
                else:
                    yn = self.propagate(yn)

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
                _, xn, _, _, _ = self.forward(tx, tu, 1)
                xn = xn * self.xstd + self.xmean
                return xn.cpu().detach().numpy()

        return ksys.StepDiscreteSystem(step_func, self.names)

    def fit(self, X: ktraj.TrajectoriesData) -> None:
        """fits the discrete system model

        :param X: trajectories
        """
        self.train(X)

    def train(self, trajs):
        """
        Train the AutoEncoder

        Internally updates the loss_hist field to record training information.

        TODO: implement early stopping

        :param X: snapshot of states
        :param Xn: snapshot of next states
        :param U: snapshot of inputs corresponding to X
        """
        assert isinstance(
            trajs, ktraj.UniformTimeTrajectoriesData
        ), "trajs must be uniform time"
        self.sampling_period = trajs.sampling_period
        self.names = trajs.state_names

        mseloss = nn.MSELoss()
        l1loss = nn.L1Loss()

        mselossv = nn.MSELoss()
        l1lossv = nn.L1Loss()

        # upload single step matrices to the compute device
        X, Xn, U = trajs.next_step_matrices
        X = torch.tensor(X.T, dtype=torch.float32).to(self.device)
        Xn = torch.tensor(Xn.T, dtype=torch.float32).to(self.device)
        U = (
            torch.tensor(U.T, dtype=torch.float32).to(self.device)
            if self.has_input
            else None
        )

        # get the scaling factors
        self.x_mult = torch.max(torch.abs(X))
        self.xmean = torch.mean(X, dim=0)
        self.xstd = self.x_mult  # torch.std(X, dim=0)

        # normalize it
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
                self.pred_loss_weight * (ae_loss + pred_loss)
                + lin_loss
                + self.max_loss_weight * inf_loss
                + self.weight_decay_weight * weight_loss
                + self.metric_loss_weight * metric_loss
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

        for _ in (
            tqdm.tqdm(range(self.max_iter))
            if (self.verbose and self.disp_progress)
            else range(self.max_iter)
        ):
            risk = None
            # roll out over several steps
            for step_size in range(1, self.rollout_steps + 1):
                # upload to GPU and normalize the data
                # NOTE: this is expensive :(
                X, Xn, U = trajs.n_step_matrices(step_size)
                X = torch.tensor(X.T, dtype=torch.float32).to(self.device)
                Xn = torch.tensor(Xn.T, dtype=torch.float32).to(self.device)
                U = (
                    torch.tensor(U.T, dtype=torch.float32).to(self.device)
                    if self.has_input
                    else None
                )
                nX = (X - self.xmean) / self.xstd
                nXn = (Xn - self.xmean) / self.xstd
                if self.has_input:
                    nU = U / self.u_mult
                else:
                    nU = None

                # get risk (loss)
                # sum over all rollouts
                x, xn, y, yn, xr = self.forward(nX, nU, step_size)
                if risk is None:
                    risk = _get_loss(mseloss, l1loss, x, xn, y, yn, xr, nXn)
                else:
                    _risk = _get_loss(mseloss, l1loss, x, xn, y, yn, xr, nXn)
                    for k, v in _risk.items():
                        risk[k] += v

            # store total loss
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
                    x, xn, y, yn, xr = self.forward(nXv, nUv, 1)
                    loss = _get_loss(mselossv, l1lossv, x, xn, y, yn, xr, nXnv)

                    # add to the loss history with validation prefix
                    for k, v in loss.items():
                        k = f"validation_{k}"
                        if k in loss_hist:
                            loss_hist[k].append(v.cpu().detach().numpy())
                        else:
                            loss_hist[k] = [v.cpu().detach().numpy()]

        self.loss_hist = loss_hist
