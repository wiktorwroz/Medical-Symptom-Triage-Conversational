# Medical Symptom Triage (Final V3)

Independent NLP project based on `sweatSmile/medical-symptom-triage-conversational`.

![App Screenshot](screenshot.png)

## Executive Summary
- Primary prediction target is `urgency` (stable, decision-oriented triage signal).
- `specialty` is presented as ranked recommendations, not a hard final diagnosis class.
- CPU-friendly final architecture:
  - `specialty`: MiniLM embeddings + LogReg/SVM
  - `urgency`: TF-IDF + LogReg

## Why Specialty Is Recommendation-Only
- This dataset is conversational and partially synthetic, with overlap across specialties.
- Multiple specialties can be clinically plausible for similar symptom descriptions.
- `urgency` is more reliable for direct actioning, while `specialty` is safer as top suggestions.

## Final V3 Pipeline
1. Load `train` and `validation` splits from HF parquet/datasets.
2. Extract patient text from `messages`, then clean and normalize.
3. Merge rare specialty classes (`<50`) into `Other` based on train distribution.
4. Build features:
   - `specialty`: MiniLM contextual embeddings
   - `urgency`: TF-IDF (`ngram_range=(1,2)`, `max_features=10000`)
5. Handle imbalance with `RandomOverSampler` (fallback if unavailable).
6. Train:
   - Specialty: LogReg + SVM
   - Urgency: LogReg
7. Evaluate on validation:
   - Accuracy
   - F1 weighted
   - MCC
   - Cohen Kappa
   - optional joint triage effectiveness score
8. Deploy Streamlit:
   - primary output: `urgency`
   - secondary output: top specialty suggestions

## Reproducibility
- Notebook: `Medical_Symptom_Triage_Conversational.ipynb`
- Run final section only:
  - `## V3 Final CPU Pipeline – Specialty + Urgency`
  - then `V3-1` -> `V3-7`
  - optional: `V3-8` export

## Streamlit
- App file: `medical_triage_streamlit.py`
- Run command:
```bash
python3 -m streamlit run "medical_triage_streamlit.py" --server.address 127.0.0.1 --server.port 8503
```

## Files
- `Medical_Symptom_Triage_Conversational.ipynb`
- `medical_triage_streamlit.py`
- `Medical_Symptom_Triage_README.md`
- `screenshot.png` (add after capturing app view)
