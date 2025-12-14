import os
import re
import glob
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Welch PSD (bandpower)
try:
    from scipy.signal import welch
    HAVE_SCIPY_SIGNAL = True
except Exception:
    HAVE_SCIPY_SIGNAL = False

# Paired t-test
try:
    from scipy.stats import ttest_rel
    HAVE_SCIPY_STATS = True
except Exception:
    HAVE_SCIPY_STATS = False


# =========================
# CONFIG
# =========================
DATA_GLOB = "data/*"
SIGNAL_KEY = "RawData/Samples"

OUT_DIR = "results"
SAVE_PDF_PLOTS = True
SHOW_PLOTS = True

# Dacă nu găsim sampling rate din fișier, folosim fallback:
FS_FALLBACK = 256  # schimbă aici dacă știi sigur altă valoare

# Benzi EEG
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
}

# Detectare nume fișier (robust la "glasses2025..." și typo "galsses")
GLASSES_PAT = re.compile(r"(?:^|[\s_\-])(?:glasses|galsses)(?![a-z])", re.IGNORECASE)
NOGLASSES_PAT = re.compile(
    r"(?:^|[\s_\-])no[\s_\-]+(?:glasses|galsses)(?![a-z])"
    r"|(?:^|[\s_\-])noglasses(?![a-z])"
    r"|(?:^|[\s_\-])no_glasses(?![a-z])"
    r"|(?:^|[\s_\-])without(?![a-z])"
    r"|(?:^|[\s_\-])fara(?![a-z])"
    r"|(?:^|[\s_\-])fară(?![a-z])",
    re.IGNORECASE
)

# Pair id = tot ce e înainte de (no )?glasses/galsses (inclusiv dacă după urmează cifre)
PAIR_ID_PAT = re.compile(
    r"^(.*?)(?:[\s_\-]+no[\s_\-]+)?(?:glasses|galsses)(?=\d|$|[\s_\-\.])",
    re.IGNORECASE
)

# Curăță timestampuri în pair_id (dacă rămân)
TS_PAT = re.compile(r"\d{4}\.\d{2}\.\d{2}[_\s]\d{2}\.\d{2}(?:\.\d+)?", re.IGNORECASE)

# Regex simplu ca să încercăm să scoatem sampling rate din descrieri
FS_PAT = re.compile(r"(?:sampling\s*rate|sample\s*rate|fs)\s*[:=]\s*(\d+)", re.IGNORECASE)


# =========================
# Logger (terminal -> fișier + ecran)
# =========================
class Tee:
    def __init__(self, filepath: str, mode: str = "w", encoding: str = "utf-8"):
        self.filepath = filepath
        self.f = open(filepath, mode, encoding=encoding)
        self._stdout = None

    def __enter__(self):
        import sys
        self._stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc, tb):
        import sys
        sys.stdout = self._stdout
        self.f.close()

    def write(self, s):
        self._stdout.write(s)
        self.f.write(s)

    def flush(self):
        self._stdout.flush()
        self.f.flush()


# =========================
# File helpers
# =========================
def list_hdf5_files():
    paths = sorted(glob.glob(DATA_GLOB))
    return [p for p in paths if p.lower().endswith((".h5", ".hdf5"))]

def classify_file(fname: str) -> str:
    low = fname.lower()
    if NOGLASSES_PAT.search(low):
        return "noglasses"
    if GLASSES_PAT.search(low):
        return "glasses"
    return "unknown"

def pair_id_from_filename(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    m = PAIR_ID_PAT.match(base)
    pid = m.group(1).strip() if m else base.strip()
    pid = TS_PAT.sub("", pid).strip()
    pid = re.sub(r"\s{2,}", " ", pid).strip()
    pid = pid.strip("_- .")
    return pid or base

def build_pairs(paths: list[str]):
    bucket = {}
    debug = []
    for p in paths:
        fname = os.path.basename(p)
        cls = classify_file(fname)
        pid = pair_id_from_filename(p)
        debug.append((fname, cls, pid))
        if cls == "glasses":
            bucket.setdefault(pid, {})["glasses_path"] = p
        elif cls == "noglasses":
            bucket.setdefault(pid, {})["noglasses_path"] = p

    rows = []
    for pid, d in bucket.items():
        if "glasses_path" in d and "noglasses_path" in d:
            rows.append({"pair_id": pid, **d})

    df = pd.DataFrame(rows, columns=["pair_id", "glasses_path", "noglasses_path"])
    if not df.empty:
        df = df.sort_values("pair_id").reset_index(drop=True)
    return df, debug

def safe_decode(x):
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, np.ndarray) and x.dtype.kind == "S":
        try:
            return x.astype("U")
        except Exception:
            return str(x)
    return str(x)

def infer_sampling_rate_from_hdf5(path: str) -> int | None:
    """
    Încearcă să găsească sampling rate în:
      - RawData/SessionDescription (string)
      - RawData/AcquisitionTaskDescription (string mare)
      - atribute (rare)
    Dacă nu găsește, întoarce None.
    """
    candidates = [
        "RawData/SessionDescription",
        "RawData/AcquisitionTaskDescription",
        "RawData/DAQDeviceDescription",
    ]
    try:
        with h5py.File(path, "r") as f:
            # din dataseturi text
            for k in candidates:
                if k in f:
                    raw = f[k][()]
                    text = safe_decode(raw)
                    m = FS_PAT.search(text)
                    if m:
                        try:
                            return int(m.group(1))
                        except Exception:
                            pass

            # din atribute (fallback)
            for objk in ["RawData", "Version", "/"]:
                try:
                    obj = f[objk] if objk != "/" else f
                    for ak, av in obj.attrs.items():
                        s = f"{ak}:{safe_decode(av)}"
                        m = FS_PAT.search(s)
                        if m:
                            return int(m.group(1))
                except Exception:
                    continue
    except Exception:
        return None
    return None

def read_samples(path: str, signal_key: str = SIGNAL_KEY) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if signal_key not in f:
            keys = []
            f.visititems(lambda n, o: keys.append(n) if isinstance(o, h5py.Dataset) else None)
            cand = [k for k in keys if "samples" in k.lower()]
            if not cand:
                raise KeyError(f"Nu găsesc '{signal_key}' și nici alternative cu 'Samples' în {path}")
            signal_key = cand[0]
        x = f[signal_key][()]

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Samples nu e 2D în {os.path.basename(path)}: shape={x.shape}")

    # vrem (time, channels)
    if x.shape[0] <= x.shape[1]:
        x = x.T
    return x


# =========================
# Feature extraction: Bandpower
# =========================
def bandpower_1d(sig_1d: np.ndarray, fs: int, fmin: float, fmax: float) -> float:
    f, Pxx = welch(sig_1d, fs=fs, nperseg=min(len(sig_1d), fs * 2))
    idx = (f >= fmin) & (f <= fmax)
    if not np.any(idx):
        return float("nan")
    return float(np.trapz(Pxx[idx], f[idx]))

def extract_bandpower_features(x: np.ndarray, fs: int, bands: dict) -> np.ndarray:
    """
    x: (time, channels)
    Return: vector 1D lungime = channels * n_bands
    Ordine: ch0:[delta,theta,alpha,beta], ch1:[...], ...
    """
    if not HAVE_SCIPY_SIGNAL:
        raise RuntimeError("Ai nevoie de scipy pentru Welch bandpower: pip install scipy")

    feats = []
    for ch in range(x.shape[1]):
        sig = x[:, ch]
        for _, (fmin, fmax) in bands.items():
            feats.append(bandpower_1d(sig, fs, fmin, fmax))
    return np.asarray(feats, dtype=np.float32)

def reshape_features_to_matrix(vec: np.ndarray, n_channels: int, n_bands: int) -> np.ndarray:
    return vec.reshape(n_channels, n_bands)


# =========================
# Summaries & plots
# =========================
def plot_mean_difference(meanD, out_png, out_pdf=None, title="Mean paired difference"):
    plt.figure()
    plt.plot(meanD)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    if out_pdf:
        plt.savefig(out_pdf)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

def plot_channel_band_heatmap(mean_abs_D_ch_band, out_png, out_pdf=None, title="Mean |difference| by channel & band"):
    # mean_abs_D_ch_band: (channels, bands)
    plt.figure()
    plt.imshow(mean_abs_D_ch_band, aspect="auto")
    plt.title(title)
    plt.xlabel("Band (0=delta,1=theta,2=alpha,3=beta)")
    plt.ylabel("Channel")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    if out_pdf:
        plt.savefig(out_pdf)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

def make_channel_effects(D_ch_band: np.ndarray, band_names: list[str]) -> pd.DataFrame:
    # D_ch_band: (n_pairs, channels, bands)
    mean_abs = np.mean(np.abs(D_ch_band), axis=(0, 2))  # (channels,)
    return pd.DataFrame({"channel": np.arange(D_ch_band.shape[1]), "mean_abs_effect": mean_abs}).sort_values(
        "mean_abs_effect", ascending=False
    )

def make_band_effects(D_ch_band: np.ndarray, band_names: list[str]) -> pd.DataFrame:
    mean_abs = np.mean(np.abs(D_ch_band), axis=(0, 1))  # (bands,)
    return pd.DataFrame({"band": band_names, "mean_abs_effect": mean_abs}).sort_values(
        "mean_abs_effect", ascending=False
    )

def feature_index_to_channel_band(idx: int, n_bands: int, band_names: list[str]):
    ch = idx // n_bands
    b = idx % n_bands
    return int(ch), band_names[int(b)]


# =========================
# Report (gata de lucrare)
# =========================
def write_report_md(path, context: dict):
    """
    context: dict cu rezultate (perechi, fs, acc, top_effects etc.)
    """
    band_list = ", ".join([f"{k}({v[0]}–{v[1]}Hz)" for k, v in context["bands"].items()])
    pairs_table = context["pairs_df"][["pair_id"]].copy()
    pairs_table["diff_norm"] = context["diff_norm"]
    pairs_table = pairs_table.sort_values("diff_norm", ascending=False)

    top_pairs_md = "\n".join(
        [f"- **{row.pair_id}**: diff_norm={row.diff_norm:.3e}" for row in pairs_table.head(5).itertuples(index=False)]
    )

    top_feats_md = ""
    if context.get("top_ttest"):
        top_feats_md = "\n".join(
            [
                f"- feat[{i}] → ch={ch}, band={band}, t={t:+.3f}, p={p:.3e}"
                for (i, ch, band, t, p) in context["top_ttest"]
            ]
        )
    else:
        top_feats_md = "_(t-test indisponibil: instalează scipy sau prea puține perechi)_"

    md = f"""# Raport analiză EEG — ochelari vs fără ochelari

## Date
- Număr fișiere HDF5: **{context["n_files"]}**
- Număr perechi (glasses vs no_glasses): **{context["n_pairs"]}**
- Sampling rate (fs): **{context["fs"]} Hz** (inferat dacă posibil; altfel fallback)
- Semnal: **{SIGNAL_KEY}** (time × channels)

Perechi analizate:
{os.linesep.join(["- " + x for x in context["pairs_df"]["pair_id"].tolist()])}

## Metodologie
1. Pentru fiecare pereche (aceeași sarcină), s-a calculat diferența *paired*:
   **D = features(glasses) − features(no_glasses)**.
2. Features: puterea spectrală (Welch) pe benzile EEG: **{band_list}**,
   pentru fiecare canal (32 canale) ⇒ **{context["n_features"]} features**.
3. Statistică: test t pereche pe fiecare feature (unde e disponibil).
4. Clasificare: Logistic Regression + StandardScaler, evaluată cu **GroupKFold pe pair_id**
   pentru a evita scurgerea informației între condiții.

## Rezultate principale
### Diferențe globale
- Norma diferenței per pereche (diff_norm) sugerează că unele sarcini sunt mai afectate.
Top 5 perechi după diff_norm:
{top_pairs_md}

### Efecte pe canale / benzi
- Cel mai afectat canal (după media |D|): **ch={context["top_channel"]}** (mean_abs_effect={context["top_channel_effect"]:.3e})
- Cea mai afectată bandă (după media |D|): **{context["top_band"]}** (mean_abs_effect={context["top_band_effect"]:.3e})

### Test t pereche (top features)
{top_feats_md}

### Clasificare (glasses vs no_glasses)
- Acuratețe GroupKFold (by pair_id): **mean={context["acc_mean"]:.3f}**, std={context["acc_std"]:.3f}, splits={context["acc_splits"]}

## Fișiere generate
- `terminal_output.txt` — log complet
- `pairs_and_diffnorm.csv` — rezultate per pereche
- `channel_effects.csv` — efect pe canal
- `band_effects.csv` — efect pe bandă
- `mean_paired_difference.png/pdf` — plot diferență medie
- `heatmap_channel_band.png/pdf` — heatmap canal × bandă
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)


# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    log_path = os.path.join(OUT_DIR, "terminal_output.txt")
    pairs_csv = os.path.join(OUT_DIR, "pairs_and_diffnorm.csv")
    channel_csv = os.path.join(OUT_DIR, "channel_effects.csv")
    band_csv = os.path.join(OUT_DIR, "band_effects.csv")
    report_md = os.path.join(OUT_DIR, "report.md")

    plot_mean_png = os.path.join(OUT_DIR, "mean_paired_difference.png")
    plot_mean_pdf = os.path.join(OUT_DIR, "mean_paired_difference.pdf") if SAVE_PDF_PLOTS else None

    plot_heat_png = os.path.join(OUT_DIR, "heatmap_channel_band.png")
    plot_heat_pdf = os.path.join(OUT_DIR, "heatmap_channel_band.pdf") if SAVE_PDF_PLOTS else None

    with Tee(log_path):
        print("CWD:", os.getcwd())
        print("OUT_DIR:", os.path.abspath(OUT_DIR))

        if not HAVE_SCIPY_SIGNAL:
            print("ERROR: scipy lipsește pentru bandpower. Instalează: pip install scipy")
            return

        paths = list_hdf5_files()
        print(f"Found {len(paths)} HDF5 files.")
        if not paths:
            print("Nu am găsit fișiere .h5/.hdf5. Verifică folderul data/.")
            return

        pairs_df, debug = build_pairs(paths)
        if pairs_df.empty:
            print("\nNu am găsit perechi glasses/no_glasses.\n")
            print("Clasificare fișiere (primele 80):")
            for fname, cls, pid in debug[:80]:
                print(f" - {fname}  =>  {cls:9}  pair_id='{pid}'")
            return

        print(f"\nFound {len(pairs_df)} pairs:\n")
        print(pairs_df.to_string(index=False))

        # Sampling rate: încearcă să-l ia dintr-un fișier, altfel fallback
        fs = infer_sampling_rate_from_hdf5(pairs_df.iloc[0]["glasses_path"])
        if fs is None:
            fs = FS_FALLBACK
            print(f"\nSampling rate not found in file; using FS_FALLBACK={fs} Hz")
        else:
            print(f"\nSampling rate inferred from file: fs={fs} Hz")

        band_names = list(BANDS.keys())
        n_bands = len(band_names)

        feats_g, feats_n = [], []
        for _, r in pairs_df.iterrows():
            xg = read_samples(r["glasses_path"])
            xn = read_samples(r["noglasses_path"])

            fg = extract_bandpower_features(xg, fs, BANDS)
            fn = extract_bandpower_features(xn, fs, BANDS)

            if fg.shape != fn.shape:
                raise ValueError(f"Feature mismatch la pair '{r['pair_id']}': {fg.shape} vs {fn.shape}")

            feats_g.append(fg)
            feats_n.append(fn)

        Xg = np.vstack(feats_g)
        Xn = np.vstack(feats_n)
        D = Xg - Xn  # (pairs, features)

        n_pairs = D.shape[0]
        n_features = D.shape[1]
        n_channels = 32
        if n_features % n_bands != 0:
            # dacă nu sunt 32 canale (de ex. alt nr), derivă din features
            n_channels = n_features // n_bands

        # Reshape per pair: (pairs, channels, bands)
        D_ch_band = D.reshape(n_pairs, n_channels, n_bands)

        # Plot 1: mean difference (pe vectorul de features)
        meanD = D.mean(axis=0)
        plot_mean_difference(
            meanD,
            out_png=plot_mean_png,
            out_pdf=plot_mean_pdf,
            title="Mean paired difference (bandpower) (glasses - no_glasses)"
        )
        print(f"\nSaved plot: {os.path.abspath(plot_mean_png)}")
        if plot_mean_pdf:
            print(f"Saved plot: {os.path.abspath(plot_mean_pdf)}")

        # Plot 2: heatmap canal × bandă (media |D|)
        mean_abs_D_ch_band = np.mean(np.abs(D_ch_band), axis=0)  # (channels, bands)
        plot_channel_band_heatmap(
            mean_abs_D_ch_band,
            out_png=plot_heat_png,
            out_pdf=plot_heat_pdf,
            title="Mean |paired difference| (bandpower) by channel & band"
        )
        print(f"\nSaved plot: {os.path.abspath(plot_heat_png)}")
        if plot_heat_pdf:
            print(f"Saved plot: {os.path.abspath(plot_heat_pdf)}")

        # CSV: diff_norm per pereche
        diff_norm = np.linalg.norm(D, axis=1)
        out_pairs = pairs_df.copy()
        out_pairs["diff_norm"] = diff_norm
        out_pairs = out_pairs.sort_values("diff_norm", ascending=False).reset_index(drop=True)
        out_pairs.to_csv(pairs_csv, index=False, encoding="utf-8")
        print(f"\nSaved CSV: {os.path.abspath(pairs_csv)}")
        print("\nTop perechi după diff_norm:")
        print(out_pairs[["pair_id", "diff_norm"]].head(30).to_string(index=False))

        # CSV: efecte pe canal / bandă
        channel_df = make_channel_effects(D_ch_band, band_names)
        band_df = make_band_effects(D_ch_band, band_names)

        channel_df.to_csv(channel_csv, index=False, encoding="utf-8")
        band_df.to_csv(band_csv, index=False, encoding="utf-8")
        print(f"\nSaved CSV: {os.path.abspath(channel_csv)}")
        print(f"Saved CSV: {os.path.abspath(band_csv)}")
        print("\nTop 10 canale afectate:")
        print(channel_df.head(10).to_string(index=False))
        print("\nBenzi afectate (desc):")
        print(band_df.to_string(index=False))

        # Paired t-test (feature-wise)
        top_ttest = None
        if HAVE_SCIPY_STATS and n_pairs >= 5:
            t, p = ttest_rel(Xg, Xn, axis=0, nan_policy="omit")
            idx = np.argsort(p)[:10]
            top_ttest = []
            print("\nTop 10 features by paired t-test p-value:")
            for i in idx:
                ch, band = feature_index_to_channel_band(int(i), n_bands, band_names)
                top_ttest.append((int(i), ch, band, float(t[i]), float(p[i])))
                print(f"  feat[{int(i):4d}]  -> ch={ch:2d} band={band:5s}  t={t[i]:+.3f}  p={p[i]:.3e}")
        else:
            print("\n(Info) SciPy stats nu e instalat sau ai prea puține perechi pentru t-test. (pip install scipy)")

        # Clasificare
        X = np.vstack([Xg, Xn])
        y = np.hstack([np.ones(len(pairs_df), dtype=int), np.zeros(len(pairs_df), dtype=int)])
        groups = np.hstack([pairs_df["pair_id"].to_numpy(), pairs_df["pair_id"].to_numpy()])

        n_splits = min(5, len(np.unique(groups)))
        cv = GroupKFold(n_splits=n_splits)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000))
        ])
        scores = cross_val_score(pipe, X, y, cv=cv, groups=groups, scoring="accuracy")

        acc_mean = float(scores.mean())
        acc_std = float(scores.std())
        print(f"\nGroupKFold accuracy (by pair_id): mean={acc_mean:.3f} std={acc_std:.3f} splits={len(scores)}")

        # Raport .md
        top_channel = int(channel_df.iloc[0]["channel"])
        top_channel_effect = float(channel_df.iloc[0]["mean_abs_effect"])
        top_band = str(band_df.iloc[0]["band"])
        top_band_effect = float(band_df.iloc[0]["mean_abs_effect"])

        context = {
            "n_files": len(paths),
            "n_pairs": len(pairs_df),
            "fs": fs,
            "bands": BANDS,
            "pairs_df": pairs_df,
            "diff_norm": diff_norm,
            "n_features": n_features,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "acc_splits": len(scores),
            "top_channel": top_channel,
            "top_channel_effect": top_channel_effect,
            "top_band": top_band,
            "top_band_effect": top_band_effect,
            "top_ttest": top_ttest,
        }
        write_report_md(report_md, context)
        print(f"\nSaved report: {os.path.abspath(report_md)}")

        print(f"\nSaved terminal output: {os.path.abspath(log_path)}")


if __name__ == "__main__":
    main()
