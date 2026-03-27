Why urgency classification works better than specialty prediction

While the model achieves strong performance for urgency classification, predicting the exact medical specialty proved significantly more challenging. This is primarily due to limitations of the dataset rather than the modeling approach.

Patient-reported symptom descriptions are often short, ambiguous, and lack sufficient clinical context, making it difficult to distinguish between closely related specialties (e.g., cardiology vs pulmonology). In contrast, urgency levels are more directly reflected in language intensity and severity cues (e.g., “severe pain”, “cannot breathe”), which are easier for models to capture.

Additionally, clinical NLP suffers from limited availability of high-quality, annotated datasets, especially for fine-grained tasks like specialty classification. Small and imbalanced datasets further reduce model performance, particularly for multi-class problems with overlapping symptom patterns.

Reducing the number of specialty classes (e.g., merging rare categories into an “Other” group) can partially improve performance, but does not fully resolve the issue due to the inherent ambiguity of the input data.

Recommended next steps include:
	•	using larger and more diverse clinical datasets,
	•	applying more powerful transformer models (e.g., full BERT or domain-specific variants),
	•	incorporating additional structured features (e.g., symptom duration, patient metadata),
	•	or reframing the task as specialty suggestion / ranking rather than strict classification.

As a result, while urgency can be reliably predicted, specialty classification in this setting is treated as a low-confidence suggestion rather than a definitive recommendation.
Dlaczego klasyfikacja urgency działa lepiej niż przewidywanie specjalisty

Model osiąga dobre wyniki dla klasyfikacji pilności (urgency), natomiast przewidywanie konkretnej specjalizacji medycznej okazało się znacznie trudniejsze. Wynika to głównie z ograniczeń datasetu, a nie samego podejścia modelowego.

Opisy objawów tworzone przez pacjentów są często krótkie, niejednoznaczne i pozbawione wystarczającego kontekstu klinicznego, co utrudnia rozróżnienie między zbliżonymi specjalizacjami (np. kardiologia vs pulmonologia). Natomiast poziom pilności jest częściej bezpośrednio widoczny w języku – poprzez intensywność i ciężkość objawów (np. „silny ból”, „nie mogę oddychać”), co modele łatwiej wychwytują.

Dodatkowo w medycznym NLP występuje problem ograniczonej liczby wysokiej jakości, oznaczonych danych, szczególnie dla zadań wymagających dokładnego rozróżniania wielu klas, takich jak specjalizacja lekarza. Małe i niezbalansowane zbiory danych dodatkowo pogarszają wyniki modeli, zwłaszcza przy problemach wieloklasowych z nakładającymi się objawami.

Zmniejszenie liczby klas (np. poprzez połączenie rzadkich specjalizacji w kategorię „Other”) może częściowo poprawić wyniki, ale nie rozwiązuje problemu w pełni ze względu na niejednoznaczność danych wejściowych.

Rekomendowane dalsze kroki:
	•	wykorzystanie większych i bardziej zróżnicowanych datasetów klinicznych,
	•	zastosowanie bardziej zaawansowanych modeli transformerowych (np. pełny BERT lub modele domenowe),
	•	dodanie dodatkowych cech strukturalnych (np. czas trwania objawów, dane pacjenta),
	•	lub zmiana podejścia na sugerowanie/ranking specjalizacji zamiast sztywnej klasyfikacji.

W rezultacie poziom pilności może być przewidywany wiarygodnie, natomiast specjalizacja powinna być traktowana jako sugestia o niskiej pewności, a nie jednoznaczna rekomendacja.

## Additional Experimental Conclusion (Specialty Modeling)

We evaluated multiple specialty models (LDA, centroid-based, and embedding + classifier variants), but none improved MCC in a meaningful way. This indicates that, in this dataset setting, specialty prediction is not well-suited to strict single-label classification.

This behavior is driven mainly by dataset structure:
- high overlap between classes (similar symptoms across specialties),
- non-unique mapping between text and label (one description can fit multiple specialties),
- weak class separability in embedding space,
- template-like and generic language that reduces discriminative signal,
- semantic continuity that behaves more like similarity search than discrete class boundaries.

As a result, this task is better framed as ranking/retrieval (top-K specialty suggestions) rather than strict top-1 classification.

### Project Outcome
- Single-label specialty models do not achieve strong MCC.
- Embedding-based methods capture semantic similarity better than hard class boundaries (often stronger top-3 behavior than top-1).
- The dataset is more suitable for:
  - top-K recommendation,
  - semantic retrieval,
  - or multi-label formulation.
