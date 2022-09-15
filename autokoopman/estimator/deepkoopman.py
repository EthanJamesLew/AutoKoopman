import torch
import torch.nn as nn
import tqdm


class DeepKoopman:
    """
    Deep Learning Koopman with Inputs

    References
        Li, Y., He, H., Wu, J., Katabi, D., & Torralba, A. (2019).
        Learning compositional koopman operators for model-based control.
        arXiv preprint arXiv:1910.08264.
    """

    def __init__(
        self,
        state_dim,
        input_dim,
        hidden_dim,
        pred_loss_weight=1.0,
        metric_loss_weight=0.2,
        hidden_enc_dim=32,
        encoder_module=None,
        decoder_module=None,
        torch_device=None,
    ):
        if torch_device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch_device

        self.hidden_dim = hidden_dim

        if encoder_module is None:
            self.encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_enc_dim),
                nn.PReLU(),
                nn.Linear(hidden_enc_dim, hidden_enc_dim),
                nn.PReLU(),
                nn.Linear(hidden_enc_dim, hidden_dim),
                nn.Tanh(),
            )
        else:
            self.encoder = encoder_module

        if decoder_module is None:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_enc_dim),
                nn.PReLU(),
                nn.Linear(hidden_enc_dim, hidden_enc_dim),
                nn.PReLU(),
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

    def forward(self, xt, ut):
        # get current step observable
        y = self.encoder(xt)

        # get next step observable
        yn = self.propagate(torch.cat((y, ut), axis=-1))

        # get next step state
        xn = self.decoder(yn)

        # get x reconstruction
        xr = self.decoder(y)

        return xt, xn, y, yn, xr

    @property
    def system_matrices(self):
        weight = self.propagate.weight.data.numpy()
        A = weight[:, : self.hidden_dim]
        B = weight[:, self.hidden_dim :]
        return A, B

    def train(self, X, Xn, U, max_iter=500, lr=1e-3, validation_data=None):
        mseloss = nn.MSELoss()
        l1loss = nn.L1Loss()

        def _get_loss(mseloss, l1loss, x, xn, y, yn, xr, Xn):
            ae_loss = mseloss(x, xr)
            pred_loss = mseloss(xn, Xn)
            metric_loss = l1loss(torch.norm(yn - y, dim=1), torch.norm(xn - x, dim=1))

            total_loss = (
                ae_loss
                + self.pred_loss_weight * pred_loss
                + self.metric_loss_weight * metric_loss
            )
            return {
                "total_loss": total_loss,
                "recon_loss": ae_loss,
                "pred_loss": pred_loss,
                "metric_loss": metric_loss,
            }

        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        propagate_optimizer = torch.optim.Adam(self.propagate.parameters(), lr=lr)

        loss_hist = {}

        for it in tqdm.tqdm(range(max_iter)):
            # get risk (loss)
            x, xn, y, yn, xr = self.forward(X, U)
            risk = _get_loss(mseloss, l1loss, x, xn, y, yn, xr, Xn)
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
            if validation_data is not None:
                with torch.no_grad():
                    Xv, Xnv, Uv = validation_data
                    x, xn, y, yn, xr = self.forward(Xv, Uv)
                    loss = _get_loss(mseloss, l1loss, x, xn, y, yn, xr, Xnv)

                    # add to the loss history with validation prefix
                    for k, v in loss.items():
                        k = f"validation_{k}"
                        if k in loss_hist:
                            loss_hist[k].append(v.cpu().detach().numpy())
                        else:
                            loss_hist[k] = [v.cpu().detach().numpy()]

        self.loss_hist = loss_hist
