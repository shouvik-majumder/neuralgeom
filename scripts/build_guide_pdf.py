"""Builds geometry_guide.pdf — a plain-language companion to the demo figures.

Covers all five demos (classification, Grassmannian, SPD, regression, mixed
populations) and includes a definitions section for every quantity computed.

Run:  python build_guide_pdf.py     (needs reportlab + pillow)
"""

# --- make the neuralgeom package importable without installing it ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuralgeom.paths import PDF_DIR  # noqa: E402

from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
OUT = PDF_DIR / "geometry_guide.pdf"

styles = getSampleStyleSheet()
ACCENT = HexColor("#3b4cc0")
GREY = HexColor("#555555")

title = ParagraphStyle("title", parent=styles["Title"], fontSize=20,
                       textColor=ACCENT, spaceAfter=4)
subtitle = ParagraphStyle("subtitle", parent=styles["Normal"], fontSize=11,
                          textColor=GREY, alignment=TA_CENTER, spaceAfter=18)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=14,
                    textColor=ACCENT, spaceBefore=16, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11.5,
                    spaceBefore=10, spaceAfter=4)
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.8,
                      leading=13.6, spaceAfter=7)
caption = ParagraphStyle("caption", parent=styles["Normal"], fontSize=8.6,
                         leading=11.5, textColor=GREY, spaceBefore=2,
                         spaceAfter=12, leftIndent=14, rightIndent=14)
box = ParagraphStyle("box", parent=body, backColor=HexColor("#eef1fb"),
                     borderPadding=6, borderColor=ACCENT, borderWidth=0.7,
                     spaceBefore=6, spaceAfter=10)
cell = ParagraphStyle("cell", parent=body, fontSize=9.0, leading=12.2,
                      spaceAfter=0)
term = ParagraphStyle("term", parent=cell, textColor=ACCENT)


def fig(name, width=6.3 * inch):
    path = FIGS / name
    w, h = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)


def P(text, style=body):
    return Paragraph(text, style)


def deftable(rows):
    t = Table([[P(f"<b>{a}</b>", term), P(b, cell)] for a, b in rows],
              colWidths=[1.55 * inch, 4.85 * inch])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#aab2d8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [HexColor("#ffffff"), HexColor("#f4f6fc")]),
    ]))
    return t


story = []

# --------------------------------------------------------------------------- #
story += [
    P("Reading the Geometry of a Neural Network", title),
    P("A plain-language guide to the pullback-metric, Grassmannian, and "
      "SPD demos — with definitions", subtitle),
]

story += [
    P("1. The setup: ordinary models, viewed unusually", h1),
    P("The running examples are deliberately mundane. <b>Demos 1–3</b> use a "
      "two-class classification problem: the classic <i>two moons</i> "
      "dataset (two interlocking crescents in the plane, one per class) and "
      "a small fully connected network — 2 &#8594; 32 &#8594; 32 &#8594; 2, "
      "Tanh activations — trained to ~100% accuracy. <b>Demo 4</b> switches "
      "to regression (fitting the surface sin 2x &#183; cos 2y), and "
      "<b>Demo 5</b> makes the classification problem realistic by letting "
      "the two populations overlap."),
    P("The unusual part is what we do after training. Instead of asking "
      "<i>“what does the network predict?”</i>, we treat the trained "
      "network as a <b>map between spaces</b>: every input point <i>x</i> "
      "is sent to a point <i>f(x)</i> in the network's output space (or, "
      "using the layer-selection tools, to any hidden layer's activation "
      "space). A map between spaces <i>distorts</i> the space it acts on — "
      "it stretches some neighborhoods, squashes others, and rotates them — "
      "and that distortion pattern is precisely <i>what the network "
      "learned</i>. The three modules measure the distortion, each "
      "answering a different question:"),
]

cheat = Table([
    [P("<b>Module</b>", cell), P("<b>Object it computes</b>", cell),
     P("<b>Question it answers</b>", cell)],
    [P("pullback_metric", cell),
     P("metric tensor g<sub>x</sub> = J<sub>x</sub><super>T</super>"
       "J<sub>x</sub>, volume element, spectrum", cell),
     P("<i>How much</i> does the network stretch or squash space around "
       "each point?", cell)],
    [P("grassmannian", cell),
     P("tangent subspaces of J<sub>x</sub> (column / row spaces), "
       "principal angles, subspace means", cell),
     P("<i>In which directions</i> is it sensitive — and how do the "
       "directions vary across points and layers?", cell)],
    [P("spd_geometry", cell),
     P("distances and averages between whole metric tensors "
       "g<sub>x</sub>", cell),
     P("Are two points (or layers, or models) <i>geometrically "
       "equivalent</i>, taking stretch and direction together?", cell)],
], colWidths=[1.2 * inch, 2.6 * inch, 2.5 * inch])
cheat.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#dfe5f7")),
    ("GRID", (0, 0), (-1, -1), 0.5, GREY),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
]))
story += [cheat, Spacer(1, 8),
          P("Because the inputs are 2-dimensional, every quantity can be "
            "drawn directly on the data plane — that is why the demos use "
            "toy problems. The code runs identically on high-dimensional "
            "inputs; only the pictures stop being drawable."),
          PageBreak()]

# --------------------------------------------------------------------------- #
story += [
    P("2. Definitions: everything the demos compute", h1),
    P("Read this section as a reference; each term links conceptually to "
      "the figures that follow. Throughout, the trained network is a smooth "
      "map f from input space R<super>n</super> to an output or hidden "
      "space R<super>m</super>, and x is one input point."),

    P("2.1 Differential objects (demo 1, pullback_metric)", h2),
    deftable([
        ("Jacobian J<sub>x</sub>",
         "The m&#215;n matrix of first derivatives &#8706;f/&#8706;x: the "
         "best linear approximation of the network at x. It tells you what "
         "happens to an infinitesimal arrow attached to x when pushed "
         "through the network. Computed per input point (so it is a "
         "<i>field</i> over the data)."),
        ("Pullback metric g<sub>x</sub>",
         "g<sub>x</sub> = J<sub>x</sub><super>T</super>J<sub>x</sub>, an "
         "n&#215;n symmetric matrix. It is a local <i>ruler</i>: the "
         "squared length the network assigns to an input direction v is "
         "v<super>T</super>g<sub>x</sub>v — i.e. the squared length of "
         "J<sub>x</sub>v measured in the output space. “Pullback” because "
         "the ordinary Euclidean ruler of the output space is pulled back "
         "through f onto the input space."),
        ("Singular values s<sub>i</sub> / stretch factors",
         "The singular values of J<sub>x</sub> (equivalently "
         "&#8730;eigenvalues of g<sub>x</sub>), sorted s<sub>1</sub> &#8805; "
         "s<sub>2</sub> &#8805; ... Each is the factor by which a unit "
         "input arrow in one principal direction is stretched. "
         "s<sub>1</sub> = the largest local stretch."),
        ("Volume element &#8730;det g",
         "The product of all stretch factors: the factor by which the map "
         "locally magnifies area/volume. Its log (used in all plots, "
         "numerically safe at any dimension) is &#931;<sub>i</sub> log "
         "s<sub>i</sub>. Large near decision boundaries, tiny where the "
         "network is insensitive."),
        ("Pseudo-determinant / pseudo-volume",
         "When g is singular (det g = 0, see <i>rank</i> below), the "
         "product over only the <i>nonzero</i> stretch factors: the volume "
         "change restricted to the directions that survive. For a scalar "
         "output (regression) this is exactly the gradient magnitude "
         "|&#8711;f|."),
        ("Rank / degeneracy",
         "The number of independent directions the map actually responds "
         "to (numerical rank of J<sub>x</sub>). rank &lt; n means the "
         "network is locally <i>blind</i> to some input directions; then g "
         "is called degenerate or rank-deficient, det g = 0, and it sits "
         "on the boundary of the SPD manifold (a PSD matrix)."),
        ("Anisotropy",
         "The ratio s<sub>1</sub>/s<sub>n</sub> (or its log): how "
         "direction-dependent the stretching is. 1 = uniform magnification; "
         "10<super>6</super> = the map keeps one direction and collapses "
         "the rest — the typical trained-classifier regime."),
        ("Tissot glyph / indicatrix",
         "A visualization, borrowed from cartography: the image of an "
         "infinitesimal circle under the map, an ellipse with semi-axes "
         "s<sub>1</sub>, s<sub>2</sub> along the principal directions. One "
         "glyph per grid point shows the whole distortion field at a "
         "glance."),
    ]),

    P("2.2 Subspace objects (demo 2, grassmannian)", h2),
    deftable([
        ("Column space / row space",
         "The column space of J<sub>x</sub> (a k-dim plane inside the "
         "output space) is where the local patch of input data lands — the "
         "tangent plane of the embedded data manifold. The row space (a "
         "k-dim plane inside the <i>input</i> space) contains the "
         "directions the network responds to; its orthogonal complement, "
         "the kernel, are directions it ignores."),
        ("Grassmannian Gr(k, D)",
         "The manifold whose “points” are all k-dimensional subspaces of "
         "R<super>D</super>. Treating each point's row/column space as a "
         "point on Gr(k, D) is what lets us do statistics on directions: "
         "distances between subspaces, averages of subspaces, etc. A "
         "subspace is represented in code by an orthonormal basis matrix Q "
         "(D&#215;k); any rotation of the basis represents the same "
         "subspace, and all quantities below are invariant to that choice."),
        ("Principal angles &#952;<sub>i</sub>",
         "The natural sequence of angles between two k-dim subspaces, "
         "0 &#8804; &#952;<sub>1</sub> &#8804; ... &#8804; &#952;"
         "<sub>k</sub> &#8804; 90&#176;: first the smallest angle "
         "achievable between a line in one and a line in the other, then "
         "the smallest achievable in the remaining directions, and so on. "
         "All angles 0 &#8660; same subspace; all 90&#176; &#8660; "
         "orthogonal. Computed from the singular values of Q<sub>1</sub>"
         "<super>T</super>Q<sub>2</sub> (their arc-cosines)."),
        ("Grassmann distances",
         "All are functions of the principal angles. <b>geodesic</b>: "
         "&#8730;(&#931;&#952;<sub>i</sub><super>2</super>), the length of "
         "the shortest rotation path between the subspaces (the canonical "
         "choice). <b>projection</b>: &#8730;(&#931;sin<super>2</super>"
         "&#952;<sub>i</sub>), bounded, robust. <b>chordal</b>: "
         "2&#8730;(&#931;sin<super>2</super>(&#952;<sub>i</sub>/2)). "
         "<b>max angle</b>: &#952;<sub>k</sub>, worst-case. "
         "<b>Binet–Cauchy</b>: &#8730;(1 &#8722; &#928;cos<super>2"
         "</super>&#952;<sub>i</sub>). <b>Martin</b>: diverges as any "
         "angle &#8594; 90&#176;."),
        ("Exp / log map",
         "The manifold versions of “move along a straight line” and “find "
         "the straight line between two points”. log<sub>Q0</sub>(Q1) "
         "returns the direction-and-speed (a tangent vector) of the "
         "shortest rotation taking subspace Q0 to Q1; exp plays it back. "
         "They are what iterative algorithms (e.g. the Karcher mean) walk "
         "with. The log map is undefined at the <b>cut locus</b> — for "
         "Grassmannians, when some principal angle is exactly 90&#176; "
         "(the shortest path is no longer unique)."),
        ("Fréchet mean (Karcher mean)",
         "The generalization of the ordinary average to a curved space. "
         "The arithmetic mean of points minimizes the sum of squared "
         "straight-line distances; the Fréchet mean minimizes the sum of "
         "squared <i>manifold</i> distances: mean = argmin<sub>M</sub> "
         "&#931;<sub>i</sub> d(M, Q<sub>i</sub>)<super>2</super>. On the "
         "Grassmannian it is “the average subspace” of a field of "
         "subspaces. “Karcher mean” = the local minimizer found by "
         "iterating: average the log-map vectors, step with exp, repeat "
         "until the step vanishes. The demos also use the closed-form "
         "<b>projection mean</b> (top-k eigenvectors of the averaged "
         "projector QQ<super>T</super>) as initialization and as the "
         "exact mean for the projection distance."),
        ("Dispersion",
         "The spread of a subspace field around its Fréchet mean: the "
         "distribution (histogram) of each point's principal angle to the "
         "mean. Plays the role of a standard deviation for directions."),
    ]),
    PageBreak(),

    P("2.3 Metric-tensor comparison (demo 3, spd_geometry)", h2),
    deftable([
        ("SPD manifold",
         "The set of symmetric positive definite n&#215;n matrices — "
         "matrices that are legitimate metrics/covariances (all eigenvalues "
         "&gt; 0). It is not a flat vector space: straight-line operations "
         "(entrywise averaging) leave physically meaningful families and "
         "cause artifacts like swelling (below). Its boundary is the "
         "<b>PSD</b> matrices (some eigenvalues exactly 0) — where "
         "degenerate pullback metrics live."),
        ("Affine-invariant distance",
         "d(A,B) = &#8214;logm(A<super>-1/2</super>B A<super>-1/2</super>)"
         "&#8214;<sub>F</sub>, the canonical Riemannian distance on SPD. "
         "“Affine-invariant” = unchanged if you linearly re-parametrize "
         "the input space (any invertible G: A &#8594; GAG<super>T"
         "</super>). Sees <i>relative</i> differences (ratios of "
         "eigenvalues); undefined for singular matrices. logm/expm are "
         "the matrix logarithm/exponential (log/exp applied to "
         "eigenvalues)."),
        ("Log-Euclidean distance",
         "d(A,B) = &#8214;logm A &#8722; logm B&#8214;<sub>F</sub>: map "
         "each matrix to its logarithm (where SPD becomes a flat space) "
         "and use the ordinary distance there. Cheap, and a close "
         "approximation to affine-invariant when matrices roughly "
         "commute."),
        ("Bures–Wasserstein distance",
         "d(A,B)<super>2</super> = tr A + tr B &#8722; 2 tr(A<super>1/2"
         "</super>B A<super>1/2</super>)<super>1/2</super>. Equals the "
         "optimal-transport (Wasserstein-2) distance between Gaussian "
         "distributions N(0,A) and N(0,B). The only one of the three that "
         "stays well-defined for singular (PSD) matrices — the tool of "
         "choice for degenerate metrics. Sees <i>absolute</i> scale."),
        ("Fréchet mean on SPD",
         "Same concept as on the Grassmannian, with the chosen SPD "
         "distance: the matrix minimizing &#931; d(M, G<sub>i</sub>)"
         "<super>2</super>. Log-Euclidean: closed form expm(mean of logm "
         "G<sub>i</sub>). Affine-invariant: Karcher iteration. Bures: "
         "fixed-point iteration. All three avoid the swelling of the "
         "naive mean."),
        ("Swelling effect",
         "The artifact of entrywise (arithmetic) averaging of SPD "
         "matrices: the mean of two anisotropic matrices with modest "
         "determinants can have an enormously larger determinant — the "
         "average invents volume that neither input has. The reason "
         "arithmetic averaging of metrics/covariances is wrong."),
        ("Regularization ridge &#949;",
         "Adding &#949;I to a singular metric to force it into SPD so "
         "affine/log-Euclidean tools apply. The demos show the resulting "
         "distances grow like log(1/&#949;) — the answer is then an "
         "artifact of &#949;, which is why PSD-native tools are provided."),
        ("Fixed-rank PSD distance",
         "For rank-k PSD matrices A &#8776; U R U<super>T</super> "
         "(U = orthonormal basis of the range, R = k&#215;k SPD factor): "
         "d<super>2</super> = &#8214;&#952;&#8214;<super>2</super> + "
         "&#945;<super>2</super> d<sub>affine</sub>(R'<sub>A</sub>, "
         "R'<sub>B</sub>)<super>2</super> — a Grassmann term (principal "
         "angles between the ranges: do the two metrics act on the same "
         "directions?) plus an SPD term (do they act with the same "
         "strengths?). Interpretable decomposition; but a <i>structure</i> "
         "metric (Bonnabel–Sepulchre), not a geodesic distance — triangle "
         "inequality not guaranteed."),
    ]),

    P("2.4 Uncertainty-related quantities (demo 5)", h2),
    deftable([
        ("Predictive entropy",
         "H(x) = &#8722;&#931;<sub>c</sub> p<sub>c</sub>(x) log p<sub>c"
         "</sub>(x), computed from the softmax probabilities: 0 when the "
         "model is certain, log 2 &#8776; 0.69 nats at 50/50. The standard "
         "probabilistic uncertainty measure the geometry is compared "
         "against."),
        ("Logit map vs probability map",
         "Two different codomains to pull back through. The <b>logit "
         "map</b> x &#8594; f(x) (raw network outputs) gives the geometry "
         "of the learned features. The <b>probability map</b> x &#8594; "
         "softmax(f(x))<sub>1</sub> gives the geometry of the learned "
         "belief; since |&#8711;p| = p(1&#8722;p)|&#8711;(logit margin)|, "
         "its stretch is largest exactly where the model is uncertain — "
         "pulling back through softmax turns the toolbox into an "
         "uncertainty detector."),
        ("Aleatoric vs epistemic",
         "Aleatoric uncertainty = irreducible randomness in the data "
         "(truly overlapping populations). Epistemic = the model's own "
         "ignorance (too little data, wrong fit). The geometry of a "
         "single trained model cannot separate the two — a limitation "
         "demonstrated in demo 5."),
    ]),
    PageBreak(),
]

# --------------------------------------------------------------------------- #
story += [
    P("3. Demo 1 — pullback_metric: how much does space stretch?", h1),
    fig("pb_step0_data_and_model.png", 5.9 * inch),
    P("<b>Step 0.</b> Raw material: the two-moons data (left) and the "
      "trained decision function (right; background color = predicted "
      "class probability, black curve = decision boundary).", caption),
    fig("pb_step2_fields.png"),
    P("<b>Step 2.</b> Left: the log-volume element log &#8730;det g at "
      "every point of the plane. Bright ridges = regions the network "
      "<i>magnifies</i>; they hug the decision boundary. Dark valleys = "
      "regions it <i>collapses</i> — inside each class blob the network is "
      "almost blind to input changes. Right: anisotropy, the ratio of the "
      "largest to smallest stretch (up to ~10<super>11</super>): the "
      "network never stretches uniformly, it stretches in essentially "
      "<i>one</i> direction.", caption),
    P("Interpretation: a well-trained classifier concentrates its "
      "discriminative power in a thin shell around the class boundary. "
      "Crossing the boundary moves the output a lot (large volume "
      "element); moving <i>within</i> a class moves it almost nowhere. "
      "The geometry makes the network's implicit invariances visible and "
      "quantitative."),
    fig("pb_step3_tissot.png", 4.6 * inch),
    P("<b>Step 3.</b> Tissot glyphs (see &#167;2.1): at each grid point, "
      "the image of a tiny circle after passing through the network. Each "
      "glyph's orientation is the surviving (most-stretched) direction; "
      "color is the stretch magnitude. The glyphs are needle-thin because "
      "the collapse ratio is ~10<super>6</super>:1 — the network keeps one "
      "direction per point and discards the other.", caption),
    PageBreak(),
    fig("pb_step4_summary.png"),
    P("<b>Step 4.</b> Summaries along a straight transect that crosses the "
      "boundary twice. Left: the two singular values (stretch factors) of "
      "J along the line. Middle: the volume element (red) peaks exactly "
      "where the class probability (dashed) flips — the boundary is a "
      "volume-expansion ridge. Right: distribution of log-volume at the "
      "actual data points, per class.", caption),
    P("What this demo shows it <b>cannot</b> do: for maps that reduce "
      "dimension the true volume element is zero (only a pseudo-volume "
      "exists); the plain volume element overflows beyond ~1000 input "
      "dimensions (use the log version, which is exact at any dimension); "
      "and for ReLU networks the Jacobian is piecewise constant, so the "
      "smooth fields shatter into flat polytopes:"),
    fig("pb_step5_relu_vs_tanh.png"),
    P("The same training problem with Tanh (left) vs ReLU (right). ReLU "
      "networks are not smooth: their geometry is constant on linear "
      "regions and undefined at the kinks between them — smooth-manifold "
      "quantities should be interpreted with care.", caption),
    PageBreak(),
]

# --------------------------------------------------------------------------- #
story += [
    P("4. Demo 2 — grassmannian: which directions does the network "
      "listen to?", h1),
    P("Demo 1 measured <i>how much</i> stretching happens. This demo "
      "isolates the <i>directions</i>, using the subspace machinery of "
      "&#167;2.2: row spaces (input directions the network responds to) "
      "and column spaces (where the local data patch lands in the output "
      "or hidden space), compared via principal angles."),
    fig("gr_step1_tangent_planes.png"),
    P("<b>Step 1.</b> To make “column space” concrete, a separate toy map "
      "embeds the input plane as a curved surface in 3D. The red/blue "
      "arrow pairs span the column space of J at sample points — they are "
      "exactly the tangent planes of the surface. For the real classifier "
      "the same object lives in 32- or 2-dimensional space; it just can't "
      "be drawn.", caption),
    fig("gr_step2_rowspace_fields.png"),
    P("<b>Step 2.</b> The row-space field of the classifier (k = 1: the "
      "single most-sensitive input direction), computed at three depths — "
      "after the first hidden layer, after the second, and at the output. "
      "Bars show the direction (a 1-dim subspace has no arrowhead); color "
      "its strength. Near the boundary all layers agree: the sensitive "
      "direction is perpendicular to the boundary. Away from it, early "
      "and late layers disagree — depth progressively re-orients "
      "sensitivity toward the class-relevant direction. Crucially, row "
      "spaces of every layer live in the <i>same input plane</i>, which "
      "is what makes this comparison legal.", caption),
    PageBreak(),
    fig("gr_step3_distance_matrices.png"),
    P("<b>Step 3.</b> Walk a closed loop around the left moon (leftmost "
      "panel) and compare the sensitive direction at every pair of loop "
      "points: each heatmap is an 80&#215;80 matrix of subspace distances "
      "under a different Grassmann metric (&#167;2.2). Bright bands mark "
      "the loop segments where the direction turns by ~90&#176; — the two "
      "boundary crossings. All four metrics tell the same qualitative "
      "story and differ only in scale: choose a metric for its "
      "invariance/robustness properties, not hoping for a different "
      "story.", caption),
    fig("gr_step4_crosslayer_and_mean.png"),
    P("<b>Step 4.</b> Left: per-point disagreement between the hidden "
      "layer's and the output's sensitive direction — bright yellow "
      "filaments are where depth <i>changes what the network attends "
      "to</i>. Middle: the Fréchet (Karcher) mean (&#167;2.2) of all "
      "data-point subspaces — the “average direction” the classifier uses "
      "(arrow) — with each point colored by its angle to that mean. "
      "Right: the dispersion histogram, a one-number-per-point summary of "
      "how coherent the direction field is.", caption),
    P("What this demo shows it <b>cannot</b> do: subspaces are only "
      "comparable inside the <i>same</i> ambient space (column spaces of "
      "a 32-dim hidden layer vs a 2-dim output cannot be compared — the "
      "code raises an error, and the demo triggers it); subspaces of "
      "different dimension k are likewise incomparable; the Riemannian "
      "log map is undefined at the cut locus (&#167;2.2); the Martin "
      "distance is infinite there and is clamped numerically; and every "
      "distance <i>depends on the choice of k</i> — with k = 2 in a "
      "2-dim input, every subspace is the whole plane and all distances "
      "are zero."),
    fig("gr_step5_cut_locus.png", 5.6 * inch),
    P("<b>Step 5.</b> Two of those limits, quantified: the log map is "
      "exact right up to — but not at — 90&#176; (left); the Martin "
      "distance diverges toward orthogonality and saturates at the "
      "implementation cap (right).", caption),
    PageBreak(),
]

# --------------------------------------------------------------------------- #
story += [
    P("5. Demo 3 — spd_geometry: comparing whole metric tensors", h1),
    P("Demos 1 and 2 each looked at one ingredient (size, direction). "
      "This demo treats each metric tensor g<sub>x</sub> as a single "
      "object — a point on the SPD manifold (&#167;2.3) — and asks how "
      "far apart two of them are, under the three standard geometries: "
      "affine-invariant, log-Euclidean, and Bures–Wasserstein."),
    fig("spd_step1_metric_ellipses.png"),
    P("<b>Step 1.</b> The raw objects: metric ellipses at nine points "
      "along a transect (left) and their eigenvalue spectra (right). "
      "Condition numbers of 10<super>3</super>–10<super>5</super> mean "
      "every one of these matrices sits close to the singular boundary "
      "of the SPD manifold — which is exactly where the choice of "
      "geometry starts to matter.", caption),
    fig("spd_step2_distance_comparison.png"),
    P("<b>Step 2.</b> Pairwise distances between the metric tensors of 60 "
      "transect points, under the three geometries, plus a scatter of "
      "affine vs Bures values. The heatmaps look similar, but the "
      "scatter is thick and curved: the two geometries are <i>not</i> "
      "monotone transforms of each other. Affine-invariant distance "
      "responds to relative conditioning; Bures to absolute scale.", caption),
    PageBreak(),
    fig("spd_step3_interpolation_swelling.png", 5.7 * inch),
    P("<b>Step 3 — the key picture.</b> Interpolating between two metric "
      "tensors (t = 0 &#8594; 1) four ways. Naive entrywise averaging "
      "(top row, red — drawn 3.4&#215; smaller to fit!) balloons the "
      "determinant to ~40,000 when both endpoints have determinant "
      "&#8804; 133: the swelling effect of &#167;2.3. The three "
      "Riemannian interpolations pass between the endpoints without "
      "inventing volume. Moral: never average metric tensors (or "
      "covariance matrices) arithmetically.", caption),
    fig("spd_step4_frechet_means.png"),
    P("<b>Step 4.</b> The same lesson at the dataset level: the average "
      "metric tensor over the 60 most boundary-adjacent data points, "
      "computed as a naive mean and as Fréchet means (&#167;2.3) in the "
      "three geometries. The naive mean (red) is fat and misoriented; "
      "affine and log-Euclidean means exactly preserve the geometric "
      "mean of the determinants (bar chart, dashed line).", caption),
    P("What this demo shows it <b>cannot</b> do: affine-invariant and "
      "log-Euclidean distances are <i>undefined</i> on singular "
      "(rank-deficient) metrics — the demo builds two bottleneck "
      "networks whose metrics are rank-1 everywhere, and shows that the "
      "“distance” you get after adding a small ridge &#949; grows like "
      "log(1/&#949;): the answer is an artifact of the ridge, not a "
      "property of the models. Bures–Wasserstein and the fixed-rank "
      "PSD metric are the honest tools there; the fixed-rank metric "
      "additionally decomposes into “range misalignment” + “spectral "
      "mismatch” (&#167;2.3), but it is a structure metric (triangle "
      "inequality not guaranteed), and no Fréchet mean is implemented "
      "on the fixed-rank manifold."),
    fig("spd_step5_degenerate_eps.png", 4.2 * inch),
    P("<b>Step 5.</b> Distance between two rank-deficient metrics as a "
      "function of the regularization &#949;: affine-invariant (blue) "
      "never converges — it diverges as &#949; &#8594; 0 — while "
      "Bures–Wasserstein (orange) is &#949;-independent.", caption),
    PageBreak(),
]

# --------------------------------------------------------------------------- #
story += [
    P("6. Demo 4 — regression: the geometry of a scalar output", h1),
    P("A regression network f: R<super>2</super> &#8594; R is "
      "geometrically extreme: with a 1-dimensional output, the pullback "
      "metric g = &#8711;f &#8711;f<super>T</super> has rank 1 at "
      "<i>every</i> point. The degenerate-metric machinery is not an edge "
      "case here — it is the whole story. The pseudo-volume is exactly "
      "the gradient magnitude |&#8711;f|; the row space is the gradient "
      "direction; the kernel of g is the level-set direction the network "
      "is blind to. And because the regression target (sin 2x &#183; "
      "cos 2y) is a known analytic function, the learned geometry can be "
      "<b>validated against exact ground truth</b> — something the "
      "classification demos could not do."),
    fig("rg_step0_fit.png"),
    P("<b>Step 0.</b> The target surface (left), the learned fit (middle; "
      "600 noisy training samples in the dotted region), and the residual "
      "(right). The fit is good — MSE near the noise floor.", caption),
    fig("rg_step1_pseudo_volume.png", 5.6 * inch),
    P("<b>Step 1.</b> The learned pseudo-volume field |&#8711;f| (left) "
      "against the analytic |&#8711;h| (right): the geometry of the fit "
      "reproduces the true gradient-magnitude landscape, including the "
      "grid of critical points where it vanishes. (volume_element itself "
      "warns here — rank(g) = 1 &lt; 2 — and returns this pseudo-volume; "
      "the honest determinant is identically zero.)", caption),
    PageBreak(),
    fig("rg_step2_gradient_field.png", 4.7 * inch),
    P("<b>Step 2.</b> The row-space field (k = 1) drawn over the level "
      "sets of the learned function. The bars are everywhere "
      "perpendicular to the level curves: the network can only “see” "
      "motion <i>across</i> level sets; motion along them is its kernel. "
      "Column-space analysis, by contrast, is vacuous for scalar outputs "
      "— every column space is the same point on Gr(1,1), and the demo "
      "verifies the distance is identically 0.", caption),
    fig("rg_step3_validation.png"),
    P("<b>Step 3 — the key picture.</b> Angle between the learned and the "
      "true gradient direction, over a region larger than the training "
      "support (cyan box). Inside the box the median error is 2.6&#176;; "
      "outside it jumps to 41&#176; and the histogram (right) becomes "
      "nearly uniform — random directions. The geometry is exactly as "
      "good as the fit, and no better: it cannot flag its own "
      "extrapolation failure.", caption),
    fig("rg_step4_fixed_rank.png"),
    P("<b>Step 4.</b> Comparing these rank-1 metrics honestly with the "
      "fixed-rank PSD distance (&#167;2.3) along a diagonal transect. "
      "Left: consecutive distances decompose into the Grassmann part "
      "(the gradient direction <i>turning</i>) and the SPD part (the "
      "gradient magnitude <i>changing</i>) — two different kinds of "
      "change the total would conflate. Right: the full pairwise "
      "distance matrix. Affine-invariant distance on the same data "
      "remains an &#949; artifact (printed by the script: d &#8776; 73 "
      "at &#949; = 10<super>-10</super> vs 13 at 10<super>-4</super>).",
      caption),
    P("Limitations specific to regression: g is blind along level sets "
      "<i>by construction</i>; the geometry describes the fitted surface, "
      "not the noise in the targets (no error bars are implied); and "
      "off-support the fields are confidently wrong (Step 3).", box),
    PageBreak(),
]

# --------------------------------------------------------------------------- #
story += [
    P("7. Demo 5 — mixed populations: overlap, uncertainty, memorization",
      h1),
    P("Real class populations overlap. This demo trains the same "
      "architecture on two-moons data at three noise levels — mild, "
      "moderate, heavy overlap — and asks what the geometry does when "
      "the classes genuinely mix. The central conceptual point: the "
      "pullback geometry depends on <b>which map you pull back through</b> "
      "(&#167;2.4) — the logit map gives feature geometry; the softmax "
      "probability map gives <i>belief</i> geometry, and only the latter "
      "tracks uncertainty."),
    fig("mx_step0_datasets.png"),
    P("<b>Step 0.</b> The three datasets with their fitted decision "
      "functions (all trained identically, with weight decay). With heavy "
      "overlap, training accuracy honestly drops to ~0.83 — close to the "
      "best achievable.", caption),
    fig("mx_step1_logit_volume.png"),
    P("<b>Step 1.</b> Logit-map log-volume fields. As overlap grows, the "
      "expansion ridge broadens and <i>weakens</i>: peak stretch falls "
      "from 21.5 to 6.4. Overlapping data punishes steep logit cliffs "
      "(they would cost cross-entropy on the wrong-side points), and the "
      "geometry records exactly that.", caption),
    PageBreak(),
    fig("mx_step2_uncertainty_pullback.png"),
    P("<b>Step 2 — the key picture.</b> Left: predictive entropy of the "
      "moderate-overlap model. Middle: the purely geometric field "
      "|&#8711;p| from pulling back through the probability map — the "
      "same uncertain set, found without computing an entropy. Right: "
      "per-grid-point scatter; the probability-map stretch tracks entropy "
      "at correlation +0.97 (blue), while the logit-map stretch does not "
      "(+0.48, red). Choose the codomain to match the question.", caption),
    fig("mx_step3_overfit_geometry.png", 5.5 * inch),
    P("<b>Step 3.</b> Same heavy-overlap data, two fits. The regularized "
      "model (left column) has a smooth decision function and a single "
      "coherent volume ridge. The overfit model (right column; no weight "
      "decay, 5&#215; longer training, 100% train accuracy where ~85% is "
      "the honest ceiling) carves islands around individual training "
      "points — and its log-volume field shatters into filaments snaking "
      "between memorized examples. Memorization is directly visible in "
      "the geometry.", caption),
    fig("mx_step3b_direction_coherence.png", 4.3 * inch),
    P("The same diagnosis, quantified with Grassmann statistics "
      "(&#167;2.2): the distribution of each data point's sensitive "
      "direction relative to the field's Fréchet mean. The overfit "
      "model's direction field is visibly more dispersed.", caption),
    PageBreak(),
    fig("mx_step4_bandwidth.png", 4.3 * inch),
    P("<b>Step 4.</b> Summary across overlap levels: the fraction of the "
      "input plane flagged as uncertain — probabilistically (entropy "
      "above threshold) and geometrically (|&#8711;p| above 25% of max) — "
      "both grow with class overlap, in step.", caption),
    P("What this demo shows it <b>cannot</b> do: the geometry describes "
      "the fitted model only — it cannot separate aleatoric overlap "
      "(“the populations truly mix here”) from epistemic wrongness (“the "
      "model happens to be flat/steep here”), and it carries no "
      "calibration guarantee: the overfit model is near-certain inside "
      "genuinely mixed regions, and its geometric uncertainty band is "
      "confidently wrong in exactly the same places. Geometry can "
      "<i>expose</i> memorization (Step 3) but not prevent it, and all "
      "statements are local and derivative-based — combine with density "
      "estimates for population-level claims."),
]

# --------------------------------------------------------------------------- #
story += [
    P("8. The takeaway", h1),
    P("For the classification problem, the geometry gives a complete "
      "local description of <i>how</i> the classifier classifies: it "
      "flattens each class onto a point-like representation (volume "
      "collapse, Demo 1), keeps exactly one sensitive direction per "
      "point — the local “which side of the boundary” axis (needle "
      "anisotropy, Demo 1; direction fields, Demo 2) — rotates that axis "
      "coherently along the boundary and across depth (Grassmann "
      "distances, Demo 2), and concentrates all metric mass in a thin "
      "shell around the decision boundary (SPD distances, Demo 3)."),
    P("For regression, the same toolbox becomes gradient-field analysis "
      "with exact validation (Demo 4); for mixed populations it becomes "
      "an uncertainty detector — provided you pull back through the "
      "probability map — and an overfitting diagnostic (Demo 5)."),
    P("None of this used the labels, the loss, or the architecture "
      "internals — only first derivatives of the trained map. The same "
      "pipeline applies verbatim to regression networks, embeddings, "
      "hidden layers of large models, or two different models compared "
      "point-by-point.", box),
    P("Reading order for the code", h2),
    P("<font face='Courier' size='9'>pullback_metric.py</font> — Jacobians "
      "and metrics (start here) &#8226; "
      "<font face='Courier' size='9'>grassmannian.py</font> — subspaces "
      "and principal angles &#8226; "
      "<font face='Courier' size='9'>spd_geometry.py</font> — SPD/PSD "
      "distances and means &#8226; demos: "
      "<font face='Courier' size='9'>demo_pullback_metric.py, "
      "demo_grassmannian.py, demo_spd_geometry.py, demo_regression.py, "
      "demo_mixed_populations.py</font> &#8226; each module has a "
      "matching <font face='Courier' size='9'>test_*.py</font> with "
      "analytic ground truths, and each demo prints its own limitation "
      "section at the end."),
]

doc = SimpleDocTemplate(str(OUT), pagesize=letter,
                        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                        topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                        title="Reading the Geometry of a Neural Network",
                        author="pullback-metric demos")
doc.build(story)
print("wrote", OUT)
