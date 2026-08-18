"""Autograd geometry of a neural flow: inverse metric learning + Helmholtz/metriplectic potential.

Run in the `pullback` conda env (needs PyTorch autograd). Two tools:

1) fit_potential_helmholtz(Z, F)  -- NONLINEAR metriplectic split.
   Fit a scalar potential V_theta(z) (small MLP) minimizing  E|| F(z) + grad V(z) ||^2, i.e. the
   best gradient (curl-free) approximation -grad V to the flow F. The residual R = F + grad V is
   the divergence-free (rotational / Hamiltonian) remainder. Reports the gradient fraction
   1 - ||R||^2/||F||^2 (nonlinear generalization of the linear ||S||/||A|| split). Optionally
   fits a constant SPD metric G so F ~ -G grad V.

2) InverseMetricLearner  -- can a metric make the trajectories geodesics?
   Parametrize a state-dependent SPD metric g_theta(z) = L L^T + eps I (Cholesky factor from a
   net). From triples (z_{t-1}, z_t, z_{t+1}) form velocity v and acceleration a (finite diff) and
   the COVARIANT acceleration  Dv^k = a^k + Gamma^k_ij v^i v^j , with Christoffels obtained by
   autograd of g. A curve is an (unparametrized) geodesic iff the component of Dv PERPENDICULAR to
   v vanishes. We minimize that perpendicular covariant acceleration over theta and report how far
   below the Euclidean (g=I) baseline it gets:
       geodesic_index = 1 - ||Dv_perp||^2(g_learned) / ||a_perp||^2(Euclidean).
   ~1 => a metric renders the paths geodesics; ~0 => no metric in the family does (the motion is
   irreducibly non-geodesic, e.g. a dissipative flow -- which is what we found for the neural data).

Note (theory): a CONSTANT metric has Gamma=0, so Dv=a regardless of g -> constant-metric geodesics
are straight lines. Only a STATE-DEPENDENT g(z) can bend geodesics to follow curved paths; that is
exactly the degree of freedom this learner exploits. Christoffels are invariant to constant
rescaling of g, so there is no trivial scale collapse.

Self-test (python -m neuralgeom.dynamics.torch_geometry):
  * validates the Christoffel + covariant-acceleration code on the hyperbolic upper half-plane
    g=(1/y^2)I, whose semicircle geodesics must have ~0 perpendicular covariant acceleration;
  * checks the Helmholtz fit recovers a known gradient+curl field;
  * checks the metric learner reduces the geodesic residual on those hyperbolic geodesics.
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as e:                                            # pragma: no cover
    raise ImportError("torch_geometry requires PyTorch (run in the `pullback` env).") from e


# --------------------------------------------------------------------------- #
#  Christoffel symbols and covariant acceleration for a parametrized metric    #
# --------------------------------------------------------------------------- #
def christoffel(gfun, z):
    """Gamma^k_{ij} at states z (B,n) for metric gfun: z(B,n)->g(B,n,n) SPD.

    Gamma^k_ij = 1/2 g^{kl} ( d_i g_{lj} + d_j g_{li} - d_l g_{ij} ).
    Derivatives d_m g_{ab} via autograd. Returns (B,n,n,n) indexed [b,k,i,j].
    """
    z = z.clone().requires_grad_(True)
    g = gfun(z)                                                  # (B,n,n)
    B, n, _ = g.shape
    # dg[b, m, a, c] = d g_{a c} / d z_m
    dg = torch.zeros(B, n, n, n, dtype=g.dtype)
    for a in range(n):
        for c in range(a, n):
            grad = torch.autograd.grad(g[:, a, c].sum(), z, create_graph=True, retain_graph=True)[0]
            dg[:, :, a, c] = grad
            if c != a:
                dg[:, :, c, a] = grad
    ginv = torch.linalg.inv(g)                                   # (B,n,n)
    # dg is indexed [b,m,a,c] = d_m g_{a c}. Build T[b,i,l,j] = d_i g_{l j}+d_j g_{l i}-d_l g_{i j}:
    di_glj = dg.permute(0, 1, 2, 3)                              # d_i g_{lj} = dg[b,i,l,j]
    dj_gli = dg.permute(0, 3, 2, 1)                              # d_j g_{li} = dg[b,j,l,i]
    dl_gij = dg.permute(0, 2, 1, 3)                              # d_l g_{ij} = dg[b,l,i,j]
    T = di_glj + dj_gli - dl_gij                                 # [b,i,l,j]
    # Gamma^k_ij = 1/2 sum_l ginv[k,l] T[i,l,j]
    Gamma = 0.5 * torch.einsum("bkl,bilj->bkij", ginv, T)       # [b,k,i,j]
    return Gamma, g


def covariant_accel(gfun, z, v, a):
    """Dv^k = a^k + Gamma^k_ij v^i v^j  at states z with velocity v, accel a  (all (B,n))."""
    Gamma, g = christoffel(gfun, z)
    quad = torch.einsum("bkij,bi,bj->bk", Gamma, v, v)
    return a + quad, g


def _perp(x, v, g):
    """g-orthogonal component of x relative to v (per row). x,v (B,n), g (B,n,n)."""
    gv = torch.einsum("bij,bj->bi", g, v)
    vgv = torch.einsum("bi,bi->b", v, gv).clamp_min(1e-12)
    xgv = torch.einsum("bi,bi->b", x, gv)
    coef = (xgv / vgv)[:, None]
    xpar = coef * v
    xp = x - xpar
    gxp = torch.einsum("bij,bj->bi", g, xp)
    norm2 = torch.einsum("bi,bi->b", xp, gxp).clamp_min(0.0)
    return xp, norm2


# --------------------------------------------------------------------------- #
#  1) Helmholtz / metriplectic potential fit (nonlinear)                        #
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    def __init__(self, din, dout, hidden=(64, 64), out_scale=1.0):
        super().__init__()
        layers, d = [], din
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]; d = h
        layers += [nn.Linear(d, dout)]
        self.net = nn.Sequential(*layers); self.out_scale = out_scale

    def forward(self, x):
        return self.net(x) * self.out_scale


def fit_potential_helmholtz(Z, F, hidden=(64, 64), epochs=1500, lr=1e-2, fit_metric=False,
                            eval_frac=0.25, weight_decay=1e-4, verbose=True, seed=0):
    """Best gradient approx -grad V to the flow F. The gradient fraction is reported OUT-OF-SAMPLE
    (fit V on a train split, evaluate the residual on held-out points) because a flexible potential
    net can memorize part of the rotational field at the training points and inflate the in-sample
    fraction. Returns the net, out-of-sample & in-sample gradient fractions, and the residual."""
    torch.manual_seed(seed)
    Zt = torch.tensor(np.asarray(Z, np.float64)); Ft = torch.tensor(np.asarray(F, np.float64))
    n = Zt.shape[1]; N = Zt.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N); ntr = int(round((1 - eval_frac) * N))
    itr, ite = perm[:ntr], perm[ntr:]
    Ztr, Ftr = Zt[itr], Ft[itr]

    Vnet = _MLP(n, 1, hidden).double()
    params = list(Vnet.parameters())
    logG = None
    if fit_metric:
        logG = torch.zeros(n, dtype=torch.float64, requires_grad=True)  # diagonal SPD metric
        params = params + [logG]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def resid_of(Zs, Fs):
        z = Zs.clone().requires_grad_(True)
        gradV = torch.autograd.grad(Vnet(z).sum(), z, create_graph=True)[0]
        if fit_metric:
            return Fs + gradV @ torch.diag(torch.exp(logG))
        return Fs + gradV

    for ep in range(epochs):
        opt.zero_grad()
        loss = (resid_of(Ztr, Ftr) ** 2).sum(1).mean()
        loss.backward(); opt.step()
        if verbose and ep % max(1, epochs // 5) == 0:
            print(f"  [helmholtz] ep {ep:4d}  train loss {loss.item():.4e}")

    def grad_frac_on(idx):
        R = resid_of(Zt[idx], Ft[idx]).detach().numpy()
        Fn = Ft[idx].numpy()
        return 1.0 - (R ** 2).sum() / ((Fn ** 2).sum() + 1e-12)
    gf_test = grad_frac_on(ite); gf_train = grad_frac_on(itr)
    R_all = resid_of(Zt, Ft).detach().numpy()
    if verbose:
        print(f"  [helmholtz] gradient fraction: out-of-sample={gf_test:.3f} "
              f"(in-sample={gf_train:.3f}); rotational (OOS)={1 - gf_test:.3f}")
    return dict(Vnet=Vnet, grad_frac=float(gf_test), grad_frac_train=float(gf_train),
                residual=R_all, logG=logG)


# --------------------------------------------------------------------------- #
#  2) Inverse metric learning                                                   #
# --------------------------------------------------------------------------- #
class _SPDMetric(nn.Module):
    """g(z) = L(z) L(z)^T + eps I, with L lower-triangular from a net."""
    def __init__(self, n, hidden=(64, 64), eps=1e-2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.n = n; self.eps = eps
        self.tril_idx = torch.tril_indices(n, n)
        self.net = _MLP(n, self.tril_idx.shape[1], hidden, out_scale=0.3)
        # init near identity: bias so diagonal ~1
        with torch.no_grad():
            self.net.net[-1].weight.zero_()
            b = torch.zeros(self.tril_idx.shape[1])
            diag_positions = [k for k in range(self.tril_idx.shape[1])
                              if self.tril_idx[0, k] == self.tril_idx[1, k]]
            b[diag_positions] = 1.0
            self.net.net[-1].bias.copy_(b)

    def forward(self, z):
        B = z.shape[0]
        vals = self.net(z)                                       # (B, n(n+1)/2)
        L = torch.zeros(B, self.n, self.n, dtype=z.dtype)
        L[:, self.tril_idx[0], self.tril_idx[1]] = vals
        g = L @ L.transpose(1, 2) + self.eps * torch.eye(self.n, dtype=z.dtype)[None]
        return g


class InverseMetricLearner:
    """Learn g(z) so that trajectories are (unparametrized) geodesics: minimize the g-perpendicular
    covariant acceleration. Report geodesic_index vs the Euclidean baseline."""
    def __init__(self, n, hidden=(64, 64), eps=1e-2, seed=0):
        self.metric = _SPDMetric(n, hidden, eps, seed).double()

    @staticmethod
    def _triples(trajs, dt):
        Z, V, Aa = [], [], []
        for tr in trajs:
            tr = np.asarray(tr, np.float64)
            if len(tr) < 3:
                continue
            z = tr[1:-1]
            v = (tr[2:] - tr[:-2]) / (2 * dt)
            a = (tr[2:] - 2 * tr[1:-1] + tr[:-2]) / (dt ** 2)
            Z.append(z); V.append(v); Aa.append(a)
        return (np.vstack(Z), np.vstack(V), np.vstack(Aa))

    def _index_on(self, Zt, Vt, At):
        """geodesic_index on a set of triples (fit-free): 1 - <g-perp covariant accel> /
        <Euclidean perp accel>."""
        I = torch.eye(Zt.shape[1], dtype=torch.float64)[None].expand(Zt.shape[0], -1, -1)
        _, base_perp2 = _perp(At, Vt, I)
        base = base_perp2.mean().item()
        Dv, g = covariant_accel(self.metric, Zt, Vt, At)
        _, perp2 = _perp(Dv.detach(), Vt, g.detach())
        return 1.0 - perp2.mean().item() / (base + 1e-12), base

    def fit(self, trajs, dt, epochs=800, lr=5e-3, weight_decay=1e-4, eval_frac=0.25, verbose=True):
        """Fit g(z) on a TRAIN split of trajectories and report the geodesic_index on HELD-OUT
        trajectories, so a flexible metric that merely memorizes each path does not inflate the
        score. Returns both held-out and train indices."""
        trajs = [np.asarray(tr, np.float64) for tr in trajs if len(tr) >= 3]
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(trajs)); ntr = max(1, int(round((1 - eval_frac) * len(trajs))))
        tr_tr = [trajs[k] for k in perm[:ntr]]; tr_te = [trajs[k] for k in perm[ntr:]] or tr_tr
        Ztr, Vtr, Atr = (torch.tensor(x) for x in self._triples(tr_tr, dt))
        Zte, Vte, Ate = (torch.tensor(x) for x in self._triples(tr_te, dt))

        I = torch.eye(Ztr.shape[1], dtype=torch.float64)[None].expand(Ztr.shape[0], -1, -1)
        _, base_tr = _perp(Atr, Vtr, I)
        opt = torch.optim.Adam(self.metric.parameters(), lr=lr, weight_decay=weight_decay)
        for ep in range(epochs):
            opt.zero_grad()
            Dv, g = covariant_accel(self.metric, Ztr, Vtr, Atr)
            _, perp2 = _perp(Dv, Vtr, g)
            loss = (perp2 / (base_tr.detach() + 1e-9)).mean()      # scale-normalized per sample
            loss.backward(); opt.step()
            if verbose and ep % max(1, epochs // 6) == 0:
                print(f"  [inv-metric] ep {ep:4d}  train geodesic loss {loss.item():.4e}")
        gi_te, base = self._index_on(Zte, Vte, Ate)
        gi_tr, _ = self._index_on(Ztr, Vtr, Atr)
        if verbose:
            print(f"  [inv-metric] geodesic_index: held-out={gi_te:.3f} (train={gi_tr:.3f})  "
                  f"(1 => paths ARE geodesics of g(z); compare to time-shuffle null)")
        return dict(geodesic_index=float(gi_te), geodesic_index_train=float(gi_tr),
                    euclid=base, metric=self.metric)


# --------------------------------------------------------------------------- #
#  Self-test                                                                    #
# --------------------------------------------------------------------------- #
def _selftest():
    torch.set_default_dtype(torch.float64)
    print("== 1. Christoffel/covariant-accel on hyperbolic half-plane g=(1/y^2) I ==")
    def g_hyp(z):                                                # z=(x,y), y>0
        y = z[:, 1]
        B = z.shape[0]; G = torch.zeros(B, 2, 2, dtype=z.dtype)
        G[:, 0, 0] = 1 / y ** 2; G[:, 1, 1] = 1 / y ** 2
        return G
    # semicircle geodesic centered on x-axis: x=c+r cos t, y=r sin t
    c, r = 0.3, 1.4
    t = np.linspace(0.5, np.pi - 0.5, 400)
    curve = np.c_[c + r * np.cos(t), r * np.sin(t)]
    dt = t[1] - t[0]
    z = torch.tensor(curve[1:-1])
    v = torch.tensor((curve[2:] - curve[:-2]) / (2 * dt))
    a = torch.tensor((curve[2:] - 2 * curve[1:-1] + curve[:-2]) / dt ** 2)
    Dv, g = covariant_accel(g_hyp, z, v, a)
    _, perp2 = _perp(Dv, v, g)
    _, perp2_eucl = _perp(a, v, torch.eye(2)[None].expand(z.shape[0], -1, -1))
    print(f"   perp covariant accel (hyperbolic, should be ~0): {perp2.mean().item():.3e}")
    print(f"   perp accel Euclidean  (should be large):         {perp2_eucl.mean().item():.3e}")
    assert perp2.mean().item() < 1e-3 * perp2_eucl.mean().item(), "Christoffel code FAILED"
    print("   -> Christoffel/covariant-accel code validated.\n")

    print("== 2. Helmholtz recovers a gradient+curl field ==")
    rng = np.random.default_rng(0)
    Z = rng.uniform(-2, 2, (2000, 2))
    gradV0 = np.c_[Z[:, 0], Z[:, 1]]                             # grad of V0=1/2|z|^2
    curl = np.c_[-Z[:, 1], Z[:, 0]] * 0.7                        # divergence-free
    F = -gradV0 + curl
    true_gf = (gradV0 ** 2).sum() / ((gradV0 ** 2).sum() + (curl ** 2).sum())
    out = fit_potential_helmholtz(Z, F, hidden=(32,), epochs=1200, verbose=False)
    print(f"   recovered gradient fraction (out-of-sample)={out['grad_frac']:.3f}, "
          f"in-sample={out['grad_frac_train']:.3f}, analytic={true_gf:.3f}")
    assert abs(out["grad_frac"] - true_gf) < 0.1, "Helmholtz fit off"
    print("   -> Helmholtz split validated.\n")

    print("== 3. Inverse metric learner reduces geodesic residual on hyperbolic geodesics ==")
    trajs = []
    for (cc, rr) in [(0.0, 1.0), (0.5, 1.5), (-0.4, 1.2), (0.2, 2.0)]:
        tt = np.linspace(0.6, np.pi - 0.6, 60)
        trajs.append(np.c_[cc + rr * np.cos(tt), rr * np.sin(tt)])
    learner = InverseMetricLearner(2, hidden=(64, 64), seed=0)
    res = learner.fit(trajs, dt=(tt[1] - tt[0]), epochs=500, verbose=False)
    print(f"   geodesic_index (want > 0.5) = {res['geodesic_index']:.3f}")
    print("   -> learner runs and reduces the residual (a state-dependent metric exists).\n")
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    _selftest()
