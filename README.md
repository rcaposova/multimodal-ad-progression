# multimodal-ad-progression

Jupyter notebooks and audit code for multimodal Alzheimer’s disease progression modeling in ADNI, including split-safe preprocessing, longitudinal tau imputation, downstream multimodal prediction, interpretability analyses, and predictor-audit checks.

---

## Supplementary Materials

- **SM1.ipynb**  
  *Contains the main analysis workflow, including split-safe preprocessing, longitudinal tau imputation, downstream multimodal modeling, attribution/snapshot evaluation, and post hoc regression scatter plots.*

- **SM2.ipynb**  
  *Contains supplementary occlusion heatmaps.*

- **predictor_audit.py**  
  *Provides a split-specific predictor audit to screen for direct forbidden target columns and suspicious target-adjacent predictor names, and exports summary tables and review files for leakage checks.* 
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
