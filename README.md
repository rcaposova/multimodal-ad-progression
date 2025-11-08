# multimodal-ad-progression

Jupyter notebooks and results for multimodal AD progression modeling (ADNI): CNN–BiLSTM vs. homoscedastic weighting, saliency/snapshot fidelity, and CVAE–MDN tau imputation with paired cross-fold statistics.

---

## Supplementary Materials

Code is available in the supplementary materials:
- **SM1.ipynb**
  *Evaluates homoscedastic uncertainty weighting and saliency/snapshot interpretability using the multimodal CNN–BiLSTM framework.*
- **SM2.ipynb**
  *Implements CVAE–MDN generative tau imputation and downstream evaluation in CNN–BiLSTM.*
- **SM3.zip** – Logs and results  
  *Contains model outputs, per-fold metrics, paired t-tests, and reproducibility artifacts produced by SM1–SM2.*

---

## Dataset

All data were obtained from the **Alzheimer’s Disease Neuroimaging Initiative (ADNI)**  
<https://adni.loni.usc.edu>

Access requires ADNI credentials.  

---

## Environment

All analyses were performed in **Jupyter notebooks** using **Python 3.10**.

Main dependencies (installable via pip or conda):
- tensorflow / keras  
- numpy, pandas, scipy, scikit-learn  
- matplotlib, seaborn  
- umap-learn, opentsne  
- xarray  

To reproduce results, simply open the notebooks (`SM1.ipynb`, `SM2.ipynb`) in Jupyter or VS Code and run all cells sequentially.
