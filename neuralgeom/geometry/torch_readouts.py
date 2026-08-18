"""PyTorch readouts with EXACT autograd Jacobians (run in the `pullback` conda env).

These are ReadoutMap-compatible (`.forward(x) -> y`, `.jacobian(x) -> (m, n)`), so they drop
straight into `PullbackMetric(readout, output_metric)`. The Jacobian is the exact reverse-mode
autograd Jacobian (`torch.func.jacrev`), and standardization is folded into the differentiated
function so J is taken w.r.t. the RAW input x.

Not importable without PyTorch; the rest of the package does not depend on this module.

Primary class:
    TorchHeteroscedasticReadout -- an MLP z -> (mu, log sigma) trained by JOINT Gaussian
    negative log-likelihood. Use with GaussianMuLogSigmaFisher() for the Tier-2 rank-2
    Fisher-Rao pullback. This is the principled version of the 2-stage mean-variance estimate
    used in the sklearn demos.

Self-test:  python -m neuralgeom.geometry.torch_readouts   (checks jacrev vs finite differences)
"""

from __future__ import annotations
import numpy as np


def torch_available() -> bool:
    try:
        import torch  # noqa
        return True
    except Exception:
        return False


class TorchHeteroscedasticReadout:
    """MLP x -> (mu, s=log sigma), trained by joint Gaussian NLL; exact autograd Jacobian.

    Parameters mirror a small regularized net; keep it small on limited data.
    """

    def __init__(self, hidden=(64, 32), weight_decay=1e-3, lr=1e-2, epochs=4000,
                 seed=0, val_frac=0.15, patience=250):
        self.cfg = dict(hidden=tuple(hidden), weight_decay=weight_decay, lr=lr,
                        epochs=epochs, seed=seed, val_frac=val_frac, patience=patience)
        self.net = None

    # --- training -----------------------------------------------------------
    def fit(self, X, y):
        import torch
        import torch.nn as nn
        c = self.cfg
        torch.manual_seed(c["seed"])
        g = torch.Generator().manual_seed(c["seed"])
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mean_ = X.mean(0); self.std_ = X.std(0) + 1e-8
        Xs = torch.tensor((X - self.mean_) / self.std_)
        yt = torch.tensor(y)

        layers, d = [], Xs.shape[1]
        for h in c["hidden"]:
            layers += [nn.Linear(d, h), nn.Tanh()]; d = h
        layers += [nn.Linear(d, 2)]
        self.net = nn.Sequential(*layers).double()

        n = len(y); idx = torch.randperm(n, generator=g)
        nval = max(int(c["val_frac"] * n), 1)
        vi, ti = idx[:nval], idx[nval:]
        opt = torch.optim.Adam(self.net.parameters(), lr=c["lr"],
                               weight_decay=c["weight_decay"])

        def nll(out, yy):
            mu, s = out[:, 0], out[:, 1]
            return torch.mean(s + 0.5 * (yy - mu) ** 2 * torch.exp(-2.0 * s))

        best, best_state, bad = np.inf, None, 0
        for _ in range(c["epochs"]):
            self.net.train(); opt.zero_grad()
            nll(self.net(Xs[ti]), yt[ti]).backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                v = nll(self.net(Xs[vi]), yt[vi]).item()
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad > c["patience"]:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    # --- ReadoutMap interface ----------------------------------------------
    def _map(self, x):
        import torch
        mean = torch.tensor(self.mean_); std = torch.tensor(self.std_)
        return self.net((x - mean) / std)              # raw x -> (mu, s)

    def forward(self, x):
        import torch
        xt = torch.tensor(np.atleast_1d(np.asarray(x, float)))
        with torch.no_grad():
            return self._map(xt).numpy()

    def jacobian(self, x, h=None):
        import torch
        from torch.func import jacrev
        xt = torch.tensor(np.atleast_1d(np.asarray(x, float)))
        return jacrev(self._map)(xt).detach().numpy()   # (2, n) exact

    def __call__(self, x):
        return self.forward(x)


def _selftest():
    import torch
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 5))
    y = 0.6 + 0.3 * X[:, 0] - 0.2 * X[:, 1] + 0.05 * rng.standard_normal(300)
    ro = TorchHeteroscedasticReadout(hidden=(16,), epochs=800).fit(X, y)
    x0 = X[0]
    Ja = ro.jacobian(x0)                                # (2,5) exact
    # finite-difference check
    Jf = np.zeros((2, 5)); hh = 1e-5
    for i in range(5):
        xp = x0.copy(); xp[i] += hh; xm = x0.copy(); xm[i] -= hh
        Jf[:, i] = (ro.forward(xp) - ro.forward(xm)) / (2 * hh)
    err = np.abs(Ja - Jf).max()
    print("torch", torch.__version__, "| jacrev vs finite-diff max err = %.2e" % err,
          "->", "OK" if err < 1e-5 else "MISMATCH")


if __name__ == "__main__":
    _selftest()
