# Raport analiză EEG — ochelari vs fără ochelari

## Date
- Număr fișiere HDF5: **13**
- Număr perechi (glasses vs no_glasses): **5**
- Sampling rate (fs): **256 Hz** (inferat dacă posibil; altfel fallback)
- Semnal: **RawData/Samples** (time × channels)

Perechi analizate:
- Vlasan Robert color test
- Vlasan Robert eye test
- Vlasan Robert grade
- Vlasan Robert grafic 4
- Vlasan Robert grapgh

## Metodologie
1. Pentru fiecare pereche (aceeași sarcină), s-a calculat diferența *paired*:
   **D = features(glasses) − features(no_glasses)**.
2. Features: puterea spectrală (Welch) pe benzile EEG: **delta(1–4Hz), theta(4–8Hz), alpha(8–13Hz), beta(13–30Hz)**,
   pentru fiecare canal (32 canale) ⇒ **128 features**.
3. Statistică: test t pereche pe fiecare feature (unde e disponibil).
4. Clasificare: Logistic Regression + StandardScaler, evaluată cu **GroupKFold pe pair_id**
   pentru a evita scurgerea informației între condiții.

## Rezultate principale
### Diferențe globale
- Norma diferenței per pereche (diff_norm) sugerează că unele sarcini sunt mai afectate.
Top 5 perechi după diff_norm:
- **Vlasan Robert eye test**: diff_norm=2.970e+04
- **Vlasan Robert grapgh**: diff_norm=2.314e+04
- **Vlasan Robert color test**: diff_norm=2.142e+04
- **Vlasan Robert grade**: diff_norm=1.625e+04
- **Vlasan Robert grafic 4**: diff_norm=4.947e+03

### Efecte pe canale / benzi
- Cel mai afectat canal (după media |D|): **ch=4** (mean_abs_effect=2.080e+03)
- Cea mai afectată bandă (după media |D|): **delta** (mean_abs_effect=2.376e+03)

### Test t pereche (top features)
- feat[114] → ch=28, band=alpha, t=-3.637, p=2.202e-02
- feat[2] → ch=0, band=alpha, t=-2.127, p=1.006e-01
- feat[81] → ch=20, band=theta, t=-2.096, p=1.041e-01
- feat[103] → ch=25, band=beta, t=-2.047, p=1.100e-01
- feat[29] → ch=7, band=theta, t=-1.959, p=1.217e-01
- feat[94] → ch=23, band=alpha, t=-1.936, p=1.250e-01
- feat[1] → ch=0, band=theta, t=-1.719, p=1.608e-01
- feat[98] → ch=24, band=alpha, t=-1.717, p=1.611e-01
- feat[80] → ch=20, band=delta, t=-1.666, p=1.711e-01
- feat[17] → ch=4, band=theta, t=-1.557, p=1.945e-01

### Clasificare (glasses vs no_glasses)
- Acuratețe GroupKFold (by pair_id): **mean=0.600**, std=0.200, splits=5

## Fișiere generate
- `terminal_output.txt` — log complet
- `pairs_and_diffnorm.csv` — rezultate per pereche
- `channel_effects.csv` — efect pe canal
- `band_effects.csv` — efect pe bandă
- `mean_paired_difference.png/pdf` — plot diferență medie
- `heatmap_channel_band.png/pdf` — heatmap canal × bandă
