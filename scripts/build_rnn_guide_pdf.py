"""Builds rnn_geometry_guide.pdf — findings from the RNN cognitive-task demos.

Run:  python build_rnn_guide_pdf.py   (needs reportlab+pillow)
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
OUT = PDF_DIR / "rnn_geometry_guide.pdf"

styles = getSampleStyleSheet()
ACCENT = HexColor("#0f766e")          # teal, to distinguish from the other guide
GREY = HexColor("#555555")

title = ParagraphStyle("title", parent=styles["Title"], fontSize=20,
                       textColor=ACCENT, spaceAfter=4)
subtitle = ParagraphStyle("subtitle", parent=styles["Normal"], fontSize=11,
                          textColor=GREY, alignment=TA_CENTER, spaceAfter=16)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=14,
                    textColor=ACCENT, spaceBefore=15, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11.5,
                    spaceBefore=10, spaceAfter=4)
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.8,
                      leading=13.6, spaceAfter=7)
caption = ParagraphStyle("caption", parent=styles["Normal"], fontSize=8.6,
                         leading=11.5, textColor=GREY, spaceBefore=2,
                         spaceAfter=12, leftIndent=14, rightIndent=14)
box = ParagraphStyle("box", parent=body, backColor=HexColor("#e6f4f2"),
                     borderPadding=6, borderColor=ACCENT, borderWidth=0.7,
                     spaceBefore=6, spaceAfter=10)
cell = ParagraphStyle("cell", parent=body, fontSize=9.0, leading=12.2,
                      spaceAfter=0)
term = ParagraphStyle("term", parent=cell, textColor=ACCENT)
mono = ParagraphStyle("mono", parent=body, fontName="Courier", fontSize=8.4,
                      leading=11.4, backColor=HexColor("#f2f5f5"),
                      borderPadding=5, spaceBefore=4, spaceAfter=8)


def fig(name, width=6.4 * inch):
    p = FIGS / name
    w, h = PILImage.open(p).size
    return Image(str(p), width=width, height=width * h / w)


def P(t, s=body):
    return Paragraph(t, s)


def dtable(rows, w0=1.5 * inch):
    t = Table([[P(f"<b>{a}</b>", term), P(b, cell)] for a, b in rows],
              colWidths=[w0, 6.4 * inch - w0])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#9fc7c2")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [HexColor("#ffffff"), HexColor("#f3f8f7")]),
    ]))
    return t


def gtable(header, rows, widths):
    data = [[P(f"<b>{c}</b>", cell) for c in header]] + \
           [[P(str(c), cell) for c in r] for r in rows]
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#cfe6e3")),
        ("GRID", (0, 0), (-1, -1), 0.5, GREY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


story = []

# --------------------------------------------------------------------------- #
story += [
    P("The Geometry of Recurrent Computation", title),
    P("Vanilla RNNs on perceptual decision, evidence integration, "
      "context-dependent decision, and working memory", subtitle),

    P("1. What this is", h1),
    P("A companion to <i>Reading the Geometry of a Neural Network</i>, which "
      "analysed feedforward networks. Here the networks are <b>recurrent</b> "
      "and the tasks require <b>time</b>: the answer cannot be computed from "
      "any single instant of the input, only from its history. That change "
      "moves the object of study. For a feedforward net we differentiated "
      "the input&#8594;output map. For an RNN the interesting map is the "
      "<b>one-step update</b>"),
    P("<b>h</b><sub>t+1</sub> = F(<b>h</b><sub>t</sub>, <b>x</b><sub>t</sub>)"
      "&#8195;&#8195;with two derivatives worth taking at every point of "
      "every trajectory:<br/><br/>"
      "&#8195;<b>J<sub>rec</sub> = &#8706;F/&#8706;h</b> &#8195;how the "
      "network's own state evolves &#8212; <i>memory, attractors</i><br/>"
      "&#8195;<b>J<sub>inp</sub> = &#8706;F/&#8706;x</b> &#8195;how new "
      "evidence enters the state &#8212; <i>input gain, selection</i>", box),
    P("Everything below is computed with the same machinery as the "
      "feedforward guide (batched Jacobians, pullback metrics, Grassmann "
      "subspace distances) — only the map being differentiated is "
      "different. The four networks are ordinary tanh RNNs of 64–96 units, "
      "trained with Adam; nothing about the architecture is specialized for "
      "the analyses."),

    P("2. The four tasks", h1),
    P("All four follow the standard systems-neuroscience trial structure: a "
      "fixation channel that drops at the response epoch, stimulus channels, "
      "and a decision scored only during the response window. Timing varies "
      "trial-to-trial, which is what forces genuine temporal processing "
      "rather than pattern matching on a fixed window."),
    dtable([
        ("Perceptual decision",
         "Two channels receive constant drive 0.5 &#177; coherence/2 plus "
         "noise for 800&#160;ms. Report which was stronger. The random-dot "
         "motion analogue: the signal is buried in noise, so the network "
         "must average over time."),
        ("Evidence integration",
         "Two channels emit discrete <i>pulses</i> at different rates over a "
         "window whose length varies from 400 to 1200&#160;ms. Report which "
         "emitted more. Ground truth is the <i>realized</i> pulse count, not "
         "the underlying rate, so the network must literally count — a leaky "
         "snapshot cannot solve it, and the variable window forbids any "
         "fixed-latency trick."),
        ("Context decision",
         "Two independent noisy features (“motion” and "
         "“colour”) are shown <i>simultaneously</i>; a context cue "
         "says which one matters. Integrate the cued feature and ignore the "
         "other. The Mante&#8211;Sussillo task: the same stimulus must be "
         "routed to opposite decisions depending on context, which is why "
         "its geometry is the most interesting of the four."),
        ("Delay match-to-sample",
         "A sample stimulus (1 of 4), then a <i>blank</i> delay of "
         "400&#8211;900&#160;ms with no input at all, then a test stimulus; "
         "report match or non-match. Pure working memory: during the delay "
         "the network is running autonomously on its own state."),
    ]),
    PageBreak(),

    P("3. Definitions: the dynamical-systems vocabulary", h1),
    P("The feedforward guide defined Jacobians, pullback metrics, "
      "Grassmannians, principal angles and Fréchet means; those carry over "
      "unchanged. These are the additional terms used here."),
    dtable([
        ("Hidden state h<sub>t</sub>",
         "The vector of unit activations at time t (64&#8211;96 numbers). "
         "A trial is a trajectory through this state space; everything the "
         "network remembers at time t is in this vector and nowhere else."),
        ("Leaky tanh RNN",
         "h<sub>t+1</sub> = (1&#8722;&#945;)h<sub>t</sub> + &#945;&#183;"
         "tanh(W<sub>rec</sub>h<sub>t</sub> + W<sub>in</sub>x<sub>t</sub> + "
         "b), with &#945; = dt/&#964;. The “vanilla” RNN of the "
         "neuroscience literature. &#964; is the <i>single-unit</i> time "
         "constant — an individual unit forgets with that time constant if "
         "the recurrent weights are removed."),
        ("Recurrent Jacobian J<sub>rec</sub>",
         "&#8706;F/&#8706;h evaluated at one (state, input) pair: the linear "
         "approximation of the autonomous dynamics near that state. For the "
         "leaky tanh RNN it is exactly (1&#8722;&#945;)I + &#945;&#183;"
         "diag(1&#8722;tanh<super>2</super>)W<sub>rec</sub> — the test suite "
         "checks the computed Jacobian against this closed form."),
        ("Eigenvalues &#955;<sub>i</sub>",
         "Of J<sub>rec</sub>; complex in general. |&#955;| &lt; 1: "
         "perturbations along that direction shrink each step (forgetting). "
         "|&#955;| &gt; 1: they grow (instability). |&#955;| &#8776; 1: they "
         "persist indefinitely — an <b>integrator</b>. Imaginary part = "
         "rotation (oscillation)."),
        ("Effective time constant",
         "&#964;<sub>eff</sub> = &#8722;dt / log|&#955;|: how long activity "
         "along that eigen-direction survives. This is a property of the "
         "<i>circuit</i> and can be far longer than the single-unit &#964; — "
         "the central quantitative finding below."),
        ("Fixed point / slow point",
         "A state where the dynamics stop: F(h, x) = h. Found by minimizing "
         "the speed q(h) = &#189;&#8214;F(h,x) &#8722; h&#8214;<super>2"
         "</super> from many starting states (Sussillo &amp; Barak 2013). "
         "q = 0 is a true fixed point; small q is a “slow point” "
         "where dynamics crawl — behaviourally almost as good."),
        ("Stable point / saddle",
         "Classified by the eigenvalues of J<sub>rec</sub> <i>at</i> the "
         "fixed point. All |&#955;| &lt; 1: stable (an attractor — states "
         "nearby fall in). One |&#955;| &gt; 1: a <b>saddle</b> — stable in "
         "every direction but one, along which states are pushed away. "
         "Saddles sit on the boundary between two choice basins and "
         "implement the decision."),
        ("Line attractor",
         "A continuous line of marginally stable (|&#955;| &#8776; 1) fixed "
         "points. Position along the line is preserved indefinitely, so the "
         "line acts as an analogue memory register: pushing the state along "
         "it stores a running total. The canonical mechanism for evidence "
         "accumulation."),
        ("Participation ratio",
         "(&#931;&#955;<sub>i</sub>)<super>2</super>/&#931;&#955;<sub>i"
         "</sub><super>2</super> over PCA eigenvalues of the state cloud: "
         "the effective number of dimensions the activity occupies. 96 units "
         "using 2.2 dimensions means the computation is genuinely "
         "low-dimensional."),
        ("Selection index",
         "Ratio of how strongly the <i>cued</i> versus the <i>ignored</i> "
         "feature drives the readout (decision) subspace, computed from "
         "J<sub>inp</sub>. 1&#215; means no gating; &gt;1&#215; means the "
         "network amplifies the relevant input."),
    ]),
    PageBreak(),
]

# --------------------------------------------------------------------------- #
story += [
    P("4. The tasks and the trained networks", h1),
    fig("rnn_step0_tasks.png"),
    P("<b>Top:</b> the input channels of one example trial per task (time "
      "runs left to right). Note the blank delay in DMS (rightmost) and the "
      "sparse pulse trains in the integration task. <b>Bottom:</b> decision "
      "accuracy during training.", caption),
    gtable(["Task", "Units", "Trained steps", "Decision accuracy"],
           [["perceptual decision", "64", "1200", "0.87"],
            ["evidence integration", "64", "1000", "0.96"],
            ["context decision", "96", "1200", "0.94"],
            ["delay match-to-sample", "96", "1200", "0.99"]],
           [1.9 * inch, 0.8 * inch, 1.3 * inch, 1.6 * inch]),
    Spacer(1, 8),
    P("Accuracies are not ceiling for the perceptual task because low-"
      "coherence trials are genuinely ambiguous — that is the task, not a "
      "training failure. Checkpoints are cached, so the analysis can be "
      "re-run without retraining."),

    P("5. Behaviour: is the computation actually happening?", h1),
    P("Geometry of a network that has not learned the task is not worth "
      "interpreting, so the first step is behavioural verification."),
    fig("rnn_step1_behaviour.png"),
    P("<b>Left:</b> a textbook psychometric curve — choice probability varies "
      "smoothly with coherence and passes through chance at zero. "
      "<b>Middle:</b> integration accuracy rises with the realized pulse-count "
      "difference, confirming the network is using the accumulated evidence "
      "rather than a heuristic. <b>Right:</b> in the context task, accuracy on "
      "<i>conflict</i> trials (where the ignored feature points the other way) "
      "is 0.97 versus 0.94 on congruent trials — i.e. <b>no congruency "
      "cost</b>. A network that failed to gate would show conflict accuracy "
      "well below congruent; this one has genuinely suppressed the irrelevant "
      "input.", caption),
    PageBreak(),
]

# --------------------------------------------------------------------------- #
story += [
    P("6. Finding 1 — memory lives in the circuit, not the units", h1),
    fig("rnn_step2_spectra.png"),
    P("Eigenvalues of J<sub>rec</sub>, pooled over 64 trials, plotted in the "
      "complex plane with the unit circle in grey. Every task develops a "
      "cluster of eigenvalues pressed right up against |&#955;| = 1.", caption),
    gtable(["Task", "top |&#955;|", "&#964;<sub>eff</sub> (circuit)",
            "&#964; (single unit)", "ratio"],
           [["perceptual decision", "0.971", "542 ms", "100 ms", "5.4&#215;"],
            ["evidence integration", "0.969", "409 ms", "200 ms", "2.0&#215;"],
            ["context decision", "0.978", "943 ms", "150 ms", "6.3&#215;"],
            ["delay match-to-sample", "0.983", "1460 ms", "200 ms", "7.3&#215;"]],
           [1.75 * inch, 0.85 * inch, 1.35 * inch, 1.3 * inch, 0.75 * inch]),
    Spacer(1, 8),
    P("Each network's effective time constant far exceeds the time constant "
      "of its own units — by 7&#215; in the working-memory task. No unit can "
      "hold information for 1.5 seconds on its own; the <i>circuit</i> can, "
      "by arranging recurrent connections so that one direction of state "
      "space neither decays nor explodes. This is the quantitative version "
      "of “memory is a network property”, and it falls straight out "
      "of one eigen-decomposition per state.", box),
    P("Note also the ordering: the two tasks with the longest memory demands "
      "(context integration over 800&#160;ms, DMS across a blank delay) "
      "developed the longest time constants. The geometry tracks the task "
      "requirement, not the architecture — all four networks are the same "
      "kind of RNN."),
    PageBreak(),

    P("7. Finding 2 — a line attractor implements the integral", h1),
    fig("rnn_step3_fixed_points.png"),
    P("<b>Left:</b> state-space trajectories (PCA), coloured by the trial's "
      "realized evidence, with the slow points found by optimization in "
      "black. The points form a <b>line</b>, and trajectories ride along it. "
      "<b>Middle:</b> position along that line at decision onset plots against "
      "the accumulated evidence with |r| = 0.88 — the attractor coordinate "
      "<i>is</i> the running total. <b>Right:</b> the top |&#955;| of each slow "
      "point clusters at 1.", caption),
    P("The search converged from 60 trajectory-sampled starting states to 14 "
      "distinct points, with speeds q ranging from 6&#215;10<super>-29</super> "
      "(exact fixed points) to 3&#215;10<super>-5</super> (slow points). "
      "Eight of the fourteen are <b>saddles</b> with exactly one unstable "
      "direction: these sit between the two choice basins, so a state nudged "
      "off the saddle is pushed toward one decision or the other. The "
      "remaining six are marginally stable and make up the attractor line "
      "itself."),
    P("This is the mechanism, read off the geometry rather than assumed: a "
      "marginally stable line stores the integral, saddles convert position "
      "along the line into a categorical choice, and the readout projects "
      "that onto the output units.", box),
    PageBreak(),

    P("8. Finding 3 — context changes the input gain, not the stimulus", h1),
    fig("rnn_step4_context.png"),
    P("<b>Left:</b> end-of-stimulus states, coloured by context. The cue "
      "displaces the entire computation to a different region of state space. "
      "<b>Middle:</b> the drive that each feature delivers into the decision "
      "plane, computed from J<sub>inp</sub> and the readout subspace. "
      "<b>Right:</b> principal angles between the two context subspaces on "
      "Gr(3, 96).", caption),
    P("The middle panel is the key measurement. Both features are physically "
      "present on every trial with the same statistics, and the input weights "
      "W<sub>in</sub> are fixed — the network cannot change what arrives. But "
      "the <i>effective</i> drive into the decision-relevant subspace depends "
      "on the state the network is in, and the context cue puts it in a "
      "different state. Measured this way:"),
    gtable(["Context", "motion drive", "colour drive", "relevant / irrelevant"],
           [["motion cued", "0.199", "0.076", "2.6&#215;"],
            ["colour cued", "0.097", "0.204", "2.1&#215;"]],
           [1.5 * inch, 1.4 * inch, 1.4 * inch, 1.8 * inch]),
    Spacer(1, 8),
    P("The cued feature drives the decision plane <b>2.34&#215;</b> harder "
      "(geometric mean) than the ignored one. Selection is implemented as a "
      "state-dependent gain on the input Jacobian — the same stimulus is "
      "routed into or away from the decision subspace depending on where the "
      "context cue has placed the state. The two context subspaces are "
      "separated by 1.28 rad on the Grassmannian, with principal angles of "
      "23&#176;, 31&#176; and 62&#176;: partially overlapping, not "
      "orthogonal, which is exactly the “shared substrate, different "
      "routing” picture.", box),
    PageBreak(),

    P("9. Finding 4 — working memory as separated subspaces", h1),
    fig("rnn_step5_memory.png"),
    P("<b>Left:</b> states at the end of the blank delay, coloured by which "
      "stimulus is being remembered — four clean, well-separated clusters, "
      "with no input present to sustain them. <b>Middle:</b> pairwise Grassmann "
      "distances between the memory subspaces of each identity. <b>Right:</b> "
      "J<sub>rec</sub> during the delay, with memory modes at |&#955;| = "
      "0.976 (&#964;<sub>eff</sub> = 1163&#160;ms).", caption),
    P("During the delay the network runs autonomously — the input is "
      "identically zero — so the cluster structure is entirely self-"
      "sustained. The participation ratio is <b>2.24</b>: 96 units, but the "
      "memory occupies barely more than two dimensions. Memory subspaces for "
      "the four identities are mutually distinct (mean off-diagonal Grassmann "
      "distance 1.17 rad), and the distance matrix shows graded rather than "
      "uniform structure — some remembered stimuli are held in more similar "
      "subspaces than others."),

    P("10. Limitations — read before interpreting", h1),
    dtable([
        ("Local linearization",
         "J<sub>rec</sub> describes an infinitesimal neighbourhood of one "
         "state at one time. There is no single Jacobian for “the "
         "network”. Always report spectra as distributions over states, "
         "as done throughout this document."),
        ("Slow points are found, not proved",
         "The optimizer returns what it converges to from the initializations "
         "supplied (here, real trajectory states). Structure can be missed; "
         "q &gt; 0 means “slow”, not “fixed”."),
        ("Input-conditional dynamics",
         "The line attractor exists <i>under the mean stimulus</i>. A "
         "different held input can create or destroy attractors — the state "
         "portrait is always conditional on that choice."),
        ("No cross-network comparison without alignment",
         "Two networks with different widths, or even the same width with a "
         "different seed, have permuted/rotated hidden units. Grassmann and "
         "SPD comparisons need a common ambient space: align first "
         "(Procrustes/CCA) or compare basis-free quantities such as "
         "eigenvalue spectra and participation ratios."),
        ("Description, not causation",
         "A high-gain direction is not proof the network uses it. The 2.3&#215; "
         "selection index is a claim about the linearized input pathway; "
         "confirming that it <i>causes</i> the behaviour requires perturbation "
         "or ablation experiments."),
        ("Smoothness assumed",
         "ReLU RNNs are piecewise linear and gated units (GRU/LSTM) saturate; "
         "the tools still run, but “the Jacobian at h” becomes "
         "region-dependent in the way shown in the feedforward guide."),
    ]),
    PageBreak(),
]

# --------------------------------------------------------------------------- #
story += [
    P("11. The code: four swappable layers", h1),
    P("The package is deliberately partitioned so that changing the task, the "
      "architecture, the training procedure, or the analysis never requires "
      "touching the other three."),
    gtable(["Layer", "File", "Contract"],
           [["what to solve", "rnn/tasks.py",
             "sample(B) &#8594; TrialBatch(inputs, targets, loss_mask, meta)"],
            ["what solves it", "rnn/models.py",
             "step(x, h) &#8594; h&#8242;; forward(inputs) &#8594; (out, H)"],
            ["how it learns", "rnn/train.py",
             "train(model, task, steps=...) — task- and model-agnostic"],
            ["what it learned", "rnn/analysis.py",
             "Jacobians, spectra, slow points, subspaces"],
            ["optional backend", "rnn/neurogym_adapter.py",
             "any neurogym env &#8594; the same TrialBatch"]],
           [1.25 * inch, 1.75 * inch, 3.4 * inch]),
    Spacer(1, 10),
    P("The single design decision that makes the geometry work is that every "
      "model exposes <font face='Courier' size='8.5'>step(x, h)</font> as a "
      "pure function. That is what lets <font face='Courier' size='8.5'>"
      "torch.func</font> take batched Jacobians of the dynamics, so the same "
      "code that pulled a metric back through a feedforward map applies "
      "unchanged to the recurrent update.", box),
    P("Adding a task", h2),
    P("Subclass <font face='Courier' size='8.5'>tasks.Task</font>, set "
      "<font face='Courier' size='8.5'>self.spec</font>, implement "
      "<font face='Courier' size='8.5'>sample(batch_size)</font>, register in "
      "<font face='Courier' size='8.5'>tasks.TASKS</font>. Adding a model: "
      "subclass <font face='Courier' size='8.5'>models._BaseRNN</font> and "
      "implement <font face='Courier' size='8.5'>step(x, h)</font> — the "
      "analysis tools need nothing else."),
    P("Using neurogym instead", h2),
    P("The adapter was verified end-to-end: a VanillaRNN trained through it "
      "on <font face='Courier' size='8.5'>PerceptualDecisionMaking-v0</font> "
      "reached 87% with no changes to the model, trainer, or analysis code."),
    P("from rnn.neurogym_adapter import NeuroGymTask<br/>"
      "task&#160;&#160;= NeuroGymTask(\"ContextDecisionMaking-v0\", dt=20)<br/>"
      "model = make_model(\"gru\", task.spec, hidden_size=128)<br/>"
      "train(model, task, steps=3000)", mono),
    P("Reproducing this document", h2),
    P("python scripts/rnn/train_rnn_models.py&#160;&#160;&#160;&#160;# cached; "
      "--resume adds steps<br/>"
      "python scripts/rnn/demo_rnn_geometry.py&#160;&#160;&#160;# analysis + "
      "figures<br/>"
      "python scripts/rnn/build_rnn_guide_pdf.py&#160;# this PDF<br/>"
      "python test_rnn.py&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;"
      "&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;# 45 "
      "checks incl. analytic Jacobians", mono),

    P("12. Summary", h1),
    P("Four ordinary tanh RNNs, trained on four temporally demanding tasks, "
      "analysed only through derivatives of their update map:"),
    dtable([
        ("Memory",
         "Effective time constants of 409&#8211;1460&#160;ms against "
         "single-unit &#964; of 100&#8211;200&#160;ms — up to 7&#215; longer. "
         "Memory is built by the circuit."),
        ("Integration",
         "A line of 14 slow points; position along it encodes accumulated "
         "evidence at |r| = 0.88; 8 saddles with one unstable direction "
         "convert that position into a choice."),
        ("Context",
         "Fixed input weights, but state-dependent input gain: the cued "
         "feature drives the decision plane 2.34&#215; harder. Context "
         "subspaces 1.28 rad apart on Gr(3, 96)."),
        ("Working memory",
         "Four self-sustained state clusters through a blank delay, "
         "occupying only 2.24 effective dimensions, with distinct memory "
         "subspaces per identity."),
    ], w0=1.15 * inch),
    Spacer(1, 6),
    P("None of these findings required labels, loss values, or knowledge of "
      "the architecture — only J<sub>rec</sub> and J<sub>inp</sub> evaluated "
      "along trajectories the networks actually visit."),
]

doc = SimpleDocTemplate(str(OUT), pagesize=letter,
                        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                        topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                        title="The Geometry of Recurrent Computation",
                        author="pullback-metric / rnn demos")
doc.build(story)
print("wrote", OUT)
