# Methods

Definitions of the quantities computed by `neuralgeom`, one line each. Symbols:
$X \in \mathbb{R}^{n_{\mathrm{trials}} \times T \times N}$ is a batch of state
trajectories, $x(t) \in \mathbb{R}^N$ a state, $J$ a Jacobian.

## Pullback metric (`neuralgeom.geometry`)

For a differentiable map $f: \mathbb{R}^k \to \mathbb{R}^m$ with Jacobian
$J(z) \in \mathbb{R}^{m \times k}$ and a metric $H(y)$ on the codomain:

| quantity | definition |
|---|---|
| pullback metric | $g(z) = J(z)^\top H(f(z))\, J(z)$, symmetric PSD, $\operatorname{rank} g \le \min(k, m)$ |
| Euclidean pullback | $H = I$, so $g = J^\top J$ |
| volume element | $\sqrt{\det g}$; for rank-deficient $g$ the product of the nonzero singular values of $J$ |
| spectrum | eigenvalues $\lambda_i$ of $g$; $\sqrt{\lambda_{\max}}$ is the largest output change per unit input step |
| anisotropy | $\lambda_{\max} / \lambda_{\min}$ |
| participation ratio | $(\sum_i \lambda_i)^2 / \sum_i \lambda_i^2$ |
| Christoffel symbols | $\Gamma^k_{ij} = \tfrac12 g^{kl}(\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij})$, by finite differences of $g$ |
| geodesics | $\ddot\gamma^k + \Gamma^k_{ij}\dot\gamma^i\dot\gamma^j = 0$ |
| Gaussian curvature (2-D domain) | from the second fundamental form of $f$, normal to the image surface |

Codomain metrics $H$ (`output_metrics.py`): Euclidean $I$; scaled Gaussian
$\sigma^{-2} I$; Bernoulli Fisher $(y(1-y))^{-1}$; categorical Fisher
$\operatorname{diag}(p) - pp^\top$; log-scale Gaussian $(\sigma y)^{-2}$;
Gaussian $(\mu, \log\sigma)$ Fisher $\operatorname{diag}(\sigma^{-2}, 2)$; block
composition of the above.

## Distances between metrics (`geometry/spd.py`)

For SPD matrices $A, B$:

| distance | definition |
|---|---|
| affine-invariant | $\lVert \log(A^{-1/2} B A^{-1/2}) \rVert_F$ |
| log-Euclidean | $\lVert \log A - \log B \rVert_F$ |
| Bures–Wasserstein | $\sqrt{\operatorname{tr}A + \operatorname{tr}B - 2\operatorname{tr}(A^{1/2} B A^{1/2})^{1/2}}$ |
| spectral ratio | $1 - \sqrt{\lambda_{\min}/\lambda_{\max}}$ over the generalised eigenvalues of $A v = \lambda B v$; bounded in $[0, 1]$ |
| fixed-rank PSD | Grassmannian distance between range subspaces plus affine-invariant distance between the $k \times k$ factors |

Fréchet means are computed in closed form (log-Euclidean) or by fixed-point
iteration (affine-invariant, Bures–Wasserstein).

## Distances between subspaces (`geometry/grassmann.py`)

Subspaces are points of the Grassmannian $\operatorname{Gr}(k, N)$, represented
by orthonormal frames $F \in \mathbb{R}^{N \times k}$ or projectors $P = FF^\top$.
For frames $A, B$ with principal angles $\theta_i = \arccos \sigma_i(A^\top B)$:

| distance | definition |
|---|---|
| `principal_angle` | $\sqrt{\sum_i \theta_i^2}$ (arc length) |
| `sqrt2_principal_angle` | $\sqrt{2}\sqrt{\sum_i \theta_i^2}$ (the geomstats canonical-metric scaling; default) |
| chordal | $\sqrt{\sum_i \sin^2\theta_i} = \lVert P_A - P_B \rVert_F / \sqrt{2}$ |

Log, exp and parallel transport on projectors are taken from geomstats.

## Subspace trajectories (`neuralgeom.subspace`)

| quantity | definition |
|---|---|
| window frame $F_m$ | top-$k$ right singular vectors of the window $X_m \in \mathbb{R}^{w \times N}$ (uncentred by default) |
| variance ratio `evr` | $\sum_{i \le k}\sigma_i^2 / \sum_i \sigma_i^2$ of the window |
| singular-value gap `sv_gap` | $\sigma_k / \sigma_{k+1}$; the frame is well determined only where this is well above 1 |
| distance from start | $d(P_1, P_m)$ |
| velocity | $v_m = \operatorname{Log}_{P_m}(P_{m+1}) / \Delta t$ |
| speed | $\lVert v_m \rVert_{P_m} = d(P_m, P_{m+1}) / \Delta t$ |
| covariant acceleration | $a_m = (v_m - \mathcal{P}_{P_{m-1} \to P_m} v_{m-1}) / \Delta t$; zero along a geodesic |
| curvature | $\kappa_m = \lVert a_m^{\perp} \rVert / s_m^2$ |
| path length, endpoint distance | $L = \sum_m d(P_m, P_{m+1})$, $\Delta = d(P_1, P_M)$; the ratio $\Delta / L$ is 1 for a geodesic segment and 0 for a closed orbit |
| Karcher mean | $\arg\min_P \sum_i d(P, P_i)^2$ by iterated $\operatorname{Exp}_P(\tfrac1n\sum_i \operatorname{Log}_P P_i)$ |
| tangent PCA | PCA of $\operatorname{Log}_{\bar P}(P_i)$; the number of components for 90 % variance is the intrinsic dimensionality |
| transported velocities | $v_m$ parallel-transported to $\bar P$, expressed in the top-2 tangent-PCA basis |

Per-frame scalar fields attached when pooling frames across trials: mean squared
activity $\tfrac1N\sum_i x_i^2$, speed, participation ratio of the window
covariance, input magnitude, and variance explained
$\lVert F^\top x \rVert^2 / \lVert x \rVert^2$.

## Topology (`neuralgeom.topology`)

| quantity | definition |
|---|---|
| persistence diagrams | Vietoris–Rips persistent homology of a precomputed distance matrix over $\mathbb{F}_2$ (ripser) |
| within-trial distances | $D_{mn} = d(F_m, F_n)$ along one trial |
| across-trial distances | the same on the frames of all trials pooled (subsampled) |
| maximum persistence | largest finite death − birth in one diagram |
| bottleneck matrix | pairwise bottleneck distance between $H_1$ diagrams of conditions |
| state–subspace projection matrix | $R_{ij} = \lVert F_j^\top x(t_i) \rVert^2 / \lVert x(t_i) \rVert^2$; the diagonal is the variance explained, off-diagonal bands indicate recurrence |
| Betti numbers | $b_0 = V - \operatorname{rank}\partial_1$, $b_1 = E - \operatorname{rank}\partial_1 - \operatorname{rank}\partial_2$, $b_2 = T - \operatorname{rank}\partial_2$ of a Delaunay complex on a 2-D MDS embedding |
| DEC operators | exterior derivative $df$ and Laplace–de Rham $\Delta f$ of a vertex field on that complex (dxtr); the MDS embedding is not isometric, so magnitudes are qualitative |
| state-space comparison | the same persistent homology on Euclidean state distances of the same trial; bottleneck distance after scaling each diagram to unit maximum death |

Results depend on $k$: a rotating line gives an $H_1$ class on
$\operatorname{Gr}(1, N) = \mathbb{RP}^{N-1}$ but not on $\operatorname{Gr}(2, N)$
if the containing plane is static.

## Recurrent dynamics (`dynamics/rnn.py`)

For a one-step update $h_{t+1} = F(h_t, x_t)$ of a torch model:

| quantity | definition |
|---|---|
| recurrent / input Jacobian | $J_{\mathrm{rec}} = \partial F / \partial h$, $J_{\mathrm{in}} = \partial F / \partial x$ along a trajectory |
| spectrum | eigenvalues $\lambda$ of $J_{\mathrm{rec}}$; time constant $\tau = -\Delta t / \log\lvert\lambda\rvert$ |
| recurrent-update pullback metric | $J^\top J$ of the update, with respect to $h$ or $x$ |
| slow points | minima of $q(h) = \tfrac12 \lVert F(h, x) - h \rVert^2$ found by gradient descent from many initial states |
| readout / input subspaces | column spaces of the readout and input weights; alignment by principal angles and mean squared cosine |

## Dynamics estimation from data (`dynamics/lds.py`)

Samples $(z, v)$ of state and finite-difference velocity:

| quantity | definition |
|---|---|
| global LDS | $\dot z = A z + b$ by ridge least squares |
| sliding-window LDS | $A(\tau), b(\tau)$ in windows of time |
| cubic field | polynomial vector field of degree 3, linearised at a chosen state |
| instrumental variables | two-stage least squares with a lagged or split-half state as instrument, removing the bias from observation noise in $z$ |
| summary of $A$ | mean of $-\operatorname{Re}\lambda$, median of $\lvert\operatorname{Im}\lambda\rvert$, maximum $\operatorname{Re}\lambda$ |
| Helmholtz decomposition | $A = M^{-1}\operatorname{Sym}(MA) + M^{-1}\operatorname{Skew}(MA)$ for an SPD metric $M$; norm fractions $\lVert\operatorname{Sym}\rVert_F / \lVert MA \rVert_F$ and $\lVert\operatorname{Skew}\rVert_F / \lVert MA \rVert_F$ |
| shared field with per-condition input | $\dot z = A z + b + u_c(t)$ with $u_c$ unconstrained or low-rank |
| velocity projections | $\langle v, Az + b \rangle / \lVert v \rVert^2$ and $\langle v, u \rangle / \lVert v \rVert^2$ per sample |
| cross-validation | trial-grouped $K$-fold $R^2$ of the velocity; within-trial circular-shift null |

The nonlinear variants in `dynamics/torch_geometry.py` fit a potential $V$ with
$F \approx -\nabla V$ and report $1 - \lVert F + \nabla V \rVert^2 / \lVert F \rVert^2$, and
fit a state-dependent metric that minimises the perpendicular covariant
acceleration of the trajectories.

## Statistics (`neuralgeom.stats`, `neuralgeom.fitting`)

Mantel and partial Mantel tests between distance matrices with permutation of
condition labels; shuffle constructors (trial permutation, circular time shift,
within-bin trial shuffle); trial-grouped cross-validation for every fitted map.
