"""Benchmark section of the appendix, from the re-evaluated checkpoints (<model>_eval.csv).
python scripts/make_appendix_tables.py   -> docs/paper/appendix_benchmark_tables.tex
Then paste it over the ArithmeticMNIST ... params part of docs/paper/appendix_results.tex."""
import os, pandas as pd, numpy as np
from scipy.stats import wilcoxon
B = os.environ.get("BENCH", "outputs/results/benchmark")
CIRC = r"^{\circ}"
NAMES = {"cbm": "CBM", "cem": "CEM (ours)", "cem_linear": "CEM-Linear", "cem_linear_noln": "CEM-Linear (no LN)",
         "cem_linear_raw": "CEM-Linear (raw units)",
         "binned5": "Binned, $k{=}5$", "binned10": "Binned, $k{=}10$", "binned15": "Binned, $k{=}15$"}
M = list(NAMES)
LIDC_C = [("subtlety", "Subtlety", "1--5", 2), ("sphericity", "Sphericity", "1--5", 2), ("margin", "Margin", "1--5", 2),
          ("lobulation", "Lobulation", "1--5", 2), ("spiculation", "Spiculation", "1--5", 2),
          ("texture", "Texture", "1--5", 2), ("diameter", "Diameter", "mm", 2),
          ("volume", "Volume", "mm$^3$", 0), ("surface_area", "Surface area", "mm$^2$", 0)]


def load(ds):
    D = {}
    for m in M:
        p = f"{B}/{ds}/{m}_eval.csv"
        if os.path.exists(p):
            d = pd.read_csv(p).sort_values("seed").drop_duplicates("seed").reset_index(drop=True)
            if ds == "mnist":
                d["digit_mae"] = (d.concept_mae_raw_digit_a + d.concept_mae_raw_digit_b) / 2
            D[m] = d
    return D


def fmt(v, d):
    return f"{v:,.{d}f}".replace(",", "{,}")


def ns(D, m, k):
    """True if NOT significantly different from CEM (paired Wilcoxon)."""
    a, b = D["cem"], D[m]
    if m == "cem" or k not in a or k not in b or b[k].isna().all():
        return False
    s = a.seed[a.seed.isin(b.seed)]
    if len(s) < 10:
        return False
    x, y = a.set_index("seed").loc[s, k].values, b.set_index("seed").loc[s, k].values
    if np.allclose(x, y):
        return True
    return wilcoxon(x, y).pvalue >= 0.05


def best_of(D, k, hib):
    vals = {m: d[k].mean() for m, d in D.items() if k in d and not d[k].isna().all() and len(d) >= 30}
    return (max if hib else min)(vals, key=vals.get) if vals else None


def table(D, cols, caption, label, small=False):
    """cols: list of (key, header, decimals, higher_is_better)"""
    best = {k: best_of(D, k, hib) for k, _, _, hib in cols if hib is not None}
    lines = [r"\begin{table}[h]", r"\centering" + (r"\footnotesize" if small else r"\small"),
             rf"\caption{{{caption}}}", rf"\label{{{label}}}"]
    if small: lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines += [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
              r"\textbf{Model} & " + " & ".join(rf"\textbf{{{h}}}" for _, h, _, _ in cols) + r" \\", r"\midrule"]
    for m, d in D.items():
        cells = []
        for k, _, dec, _ in cols:
            if k not in d or d[k].isna().all():
                cells.append("---"); continue
            s = f"{fmt(d[k].mean(), dec)} \\pm {fmt(d[k].std(ddof=0), dec)}"
            if best.get(k) == m: s = rf"\mathbf{{{s}}}"
            cells.append(f"${s}{CIRC if ns(D, m, k) else ''}$")
        name = NAMES[m] + (r"\,\textsuperscript{\ddag}" if m == "cbm" else "")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def nseeds(D):
    n = {len(d) for d in D.values()}
    return f"{min(n)}" if len(n) == 1 else f"{min(n)}--{max(n)}"


FOOT = (r"Mean $\pm$ standard deviation over seeds. Bold: best mean. "
        r"$^{\circ}$: \emph{not} significantly different from CEM (ours) "
        r"(two-sided Wilcoxon signed-rank test paired by seed, $p \ge 0.05$); every unmarked difference is significant. "
        r"\ddag: original CBM with a linear concept read-out (see Section~\ref{app:caveats}).")
SWAP = (r"Swap test: every test input $x_A$ is given \emph{all} the concepts $c_B$ of another, randomly paired test "
        r"sample (fixed pairing, identical for all models), and the output is scored against $y_B$ "
        r"(``follows the concepts'') and against $y_A$ (``follows the input''). ")

out = []
# ---------------- MNIST
D = load("mnist")
out.append(r"\subsection{ArithmeticMNIST}" "\n\\label{app:mnist}\n")
out.append(table(D, [("task_mae", "Task MAE", 2, False), ("interv_mae", "Interv.\\ MAE", 2, False),
                     ("digit_mae", "Digit MAE", 3, False),
                     ("concept_acc", "Bin acc.", 3, True)],
    rf"ArithmeticMNIST, {nseeds(D)} seeds, 30 epochs. Task: the result of $a \circ b$ with $\circ \in \{{+,\times\}}$ "
    r"(regression, raw units). ``Interv.'' replaces both predicted digits with the ground truth. "
    r"All errors are mean absolute errors in raw units: task MAE in units of the result, digit MAE "
    r"the error of the predicted digits in "
    r"digit units ($1$--$9$), averaged over $a$ and $b$. "
    r"Bin accuracy only exists for binned models; it is not digit accuracy (with $k{=}5$, digits 1--2, 3--4, 6--7 "
    r"and 8--9 share a bin). Binned concept errors are measured at the bin centre, so even perfect bins have a "
    r"quantisation floor. Under MSE (result files), Binned $k{=}5$ is not significantly different from CEM (ours) "
    rf"on task error (${D['binned5'].task_mse.mean():.1f}$ vs.\ ${D['cem'].task_mse.mean():.1f}$); "
    rf"under MAE it is better (${D['binned5'].task_mae.mean():.2f}$ vs.\ ${D['cem'].task_mae.mean():.2f}$). " + FOOT, "tab:app-mnist"))
out.append(table(D, [("zero_task_mae", "Zero: no interv.", 2, False),
                     ("zero_interv0_mae", "Zero: fix zero concept", 2, False),
                     ("zero_interv_mae", "Zero: fix all", 2, False),
                     ("nonzero_interv_mae", "Non-zero: fix all", 2, False)],
    rf"ArithmeticMNIST zero test. The zero group is the {int(D['cem'].n_zero.iloc[0])} of "
    rf"{int(D['cem'].n_test.iloc[0])} test samples where at least one digit is a 5, which is exactly $0$ "
    r"after normalisation. ``Fix zero concept'' sets only the zero-valued concept(s) to the truth and keeps "
    r"the other prediction. Task MAE, lower is better. " + FOOT, "tab:app-mnist-zero"))
# ---------------- Pendulum
D = load("pendulum")
out.append(r"\subsection{Pendulum}" "\n\\label{app:pendulum}\n")
out.append(table(D, [("task_mae", "Task MAE", 3, False), ("interv_mae", "Interv.\\ MAE", 3, False),
                     ("concept_mae_raw_theta", r"$\theta$ MAE", 2, False),
                     ("concept_mae_raw_phi", r"$\varphi$ MAE", 2, False), ("concept_acc", "Bin acc.", 3, True)],
    rf"Pendulum, {nseeds(D)} seeds, 20 epochs. Concepts: pendulum angle $\theta$ and light angle $\varphi$; "
    r"task: shadow position and length (2 outputs, raw units), a deterministic function of the concepts. "
    r"Mean absolute errors in raw units: task MAE averaged over the two outputs, $\theta$ and $\varphi$ MAE in "
    r"the units of the dataset "
    r"($\theta \in [-200, 200]$, $\varphi \in [60, 140]$). " + FOOT, "tab:app-pendulum", small=True))
out.append(table(D, [("zero_task_mae", "Zero: no interv.", 3, False),
                     ("zero_interv0_mae", "Zero: fix zero concept", 3, False),
                     ("zero_interv_mae", "Zero: fix all", 3, False),
                     ("nonzero_interv_mae", "Non-zero: fix all", 3, False)],
    rf"Pendulum zero test. Zero group: the {int(D['cem'].n_zero.iloc[0])} of {int(D['cem'].n_test.iloc[0])} "
    r"test samples with a concept within $0.025$ of $0$ (normalised units). Columns as in "
    r"Table~\ref{tab:app-mnist-zero}. " + FOOT, "tab:app-pendulum-zero"))
out.append(table(D, [("interv_mae", "Interv.\\ ($c_A$) vs.\\ $y_A$", 3, False),
                     ("swap_mae", "Swap ($c_B$) vs.\\ $y_B$", 3, False),
                     ("swapA_mae", "Swap ($c_B$) vs.\\ $y_A$", 3, None)],
    r"Pendulum swap test, task MAE (raw units). " + SWAP + r"Since the task is a function of $(\theta, \varphi)$ only, a model "
    r"that follows its concepts has a low error in the second column and a high one in the third. The third column "
    r"has no `better' direction and is not bolded. " + FOOT, "tab:app-pendulum-swap"))
# ---------------- LIDC
D = load("lidc")
nz = D["cem"].n_zero; nt = D["cem"].n_test
out.append(r"\subsection{LIDC-IDRI}" "\n\\label{app:lidc}\n")
out.append(table(D, [("task_bacc", "Task bal.\\ acc.", 3, True),
                     ("interv_bacc", "Interv.\\ bal.\\ acc.", 3, True),
                     ("concept_mae_rel", "Concept MAE (rel.)", 4, False)],
    rf"LIDC-IDRI malignancy (binary), 9 concepts (six radiologist ratings on a 1--5 scale and three size "
    rf"measurements: diameter, volume, surface area), {nseeds(D)} seeds, 250 epochs; train/test split by patient, "
    r"redrawn per seed (same split for every model within a seed). Higher is better except concept errors "
    r"(plain accuracy shows the same ordering and is in the result files). "
    r"Concept MAE (rel.) is the raw-unit absolute error divided by the concept's "
    r"range, averaged over concepts (per-concept raw errors in Table~\ref{tab:app-lidc-raw}). " + FOOT,
    "tab:app-lidc", small=True))
out.append(table(D, [("zero_task_bacc", "Zero: no interv.", 3, True),
                     ("zero_interv0_bacc", "Zero: fix zero concept", 3, True),
                     ("zero_interv_bacc", "Zero: fix all", 3, True),
                     ("nonzero_interv_bacc", "Non-zero: fix all", 3, True)],
    rf"LIDC-IDRI zero test, balanced accuracy. Zero group: test nodules with a concept within $0.025$ of $0$; "
    rf"{nz.mean():.0f} of {nt.mean():.0f} test nodules on average (range {nz.min()}--{nz.max()}), so these "
    r"estimates are noisy. Columns as in Table~\ref{tab:app-mnist-zero}. " + FOOT, "tab:app-lidc-zero"))
out.append(table(D, [("interv_bacc", "Interv.\\ ($c_A$) vs.\\ $y_A$", 3, True),
                     ("swap_bacc", "Swap ($c_B$) vs.\\ $y_B$", 3, True),
                     ("swapA_bacc", "Swap ($c_B$) vs.\\ $y_A$", 3, None)],
    r"LIDC-IDRI swap test, balanced accuracy. " + SWAP + r"Malignancy is not a function of the concepts, so "
    r"$y_B$ is only the label most consistent with $c_B$; CBM, whose head sees only the concepts, gives the "
    r"reference. The third column is not bolded. " + FOOT, "tab:app-lidc-swap"))
# per-concept raw MAE, transposed: rows = concepts, columns = models
ms = [m for m in M if m in D]
hdr = {"cbm": "CBM\\textsuperscript{\\ddag}", "cem": "Ours", "cem_linear": "Lin.", "cem_linear_noln": "Lin.\\ no LN",
       "cem_linear_raw": "Lin.\\ raw", "binned5": "$k{=}5$", "binned10": "$k{=}10$", "binned15": "$k{=}15$"}
lines = [r"\begin{table}[h]", r"\centering\small",
         rf"\caption{{LIDC-IDRI concept error in raw units: mean absolute error of each predicted concept, "
         rf"mean over {nseeds(D)} seeds (standard deviations in the result files). Ours: CEM (ours); Lin.: CEM-Linear; "
         r"$k$: binned. Bold: lowest error. $^{\circ}$: not significantly different from CEM (ours) "
         r"(Wilcoxon, $p \ge 0.05$). \ddag: original CBM (Section~\ref{app:caveats}).}",
         r"\label{tab:app-lidc-raw}", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{ll" + "c" * len(ms) + "}", r"\toprule",
         r"\textbf{Concept} & \textbf{Range} & " + " & ".join(rf"\textbf{{{hdr[m]}}}" for m in ms) + r" \\", r"\midrule"]
for key, name, unit, dec in LIDC_C:
    k = f"concept_mae_raw_{key}"
    b = best_of(D, k, False)
    lo, hi = {"1--5": ("1", "5"), "mm": ("2.6", "50"), "mm$^3$": ("8.5", "31{,}112"), "mm$^2$": ("14", "9{,}027")}[unit]
    rng = f"{lo}--{hi}" + ("" if unit == "1--5" else f" {unit}")
    cells = []
    for m in ms:
        s = fmt(D[m][k].mean(), dec)
        if b == m: s = rf"\mathbf{{{s}}}"
        cells.append(f"${s}{CIRC if ns(D, m, k) else ''}$")
    lines.append(f"{name} & {rng} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
out.append("\n".join(lines))
# ---------------- params
rows = []
for ds in ("mnist", "pendulum", "lidc"):
    D = load(ds); rows.append({m: int(d.params.iloc[0]) for m, d in D.items()})
lines = [r"\begin{table}[h]", r"\centering\small",
         r"\caption{Trainable parameters per model (encoder + bottleneck + head). All models share the encoder, "
         r"concept-network width ($64$), depth ($2$) and embedding size ($16$ per concept); only the bottleneck differs.}",
         r"\label{tab:app-params}", r"\begin{tabular}{lrrr}", r"\toprule",
         r"\textbf{Model} & \textbf{ArithmeticMNIST} & \textbf{Pendulum} & \textbf{LIDC-IDRI} \\", r"\midrule"]
for m in M:
    lines.append(NAMES[m] + (r"\,\textsuperscript{\ddag}" if m == "cbm" else "") + " & "
                 + " & ".join(f"{r[m]:,}".replace(",", "{,}") if m in r else "---" for r in rows) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
out.append("\n".join(lines))
open(os.environ.get("OUT", "docs/paper/appendix_benchmark_tables.tex"), "w").write("\n".join(out))
print("wrote", os.environ.get("OUT", "docs/paper/appendix_benchmark_tables.tex"))
