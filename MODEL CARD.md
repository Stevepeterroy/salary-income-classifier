# Historical income-band classification

Educational Adult-schema classifier using age, hours, education and occupation. It does not predict a salary amount. Historical social and sampling biases persist even without explicit sensitive features; never use for hiring, lending or pay decisions.

Earlier local holdout: 9,769 rows, accuracy 77.8%, >50K recall 41.3%. These figures were reproduced during the polish pass; no improved performance is claimed. Retraining now records a majority-class baseline alongside per-class metrics. Training/testing use a stratified random split, not external validation.

Run `python -m unittest discover -v` for CSV boundary checks. Train only trusted local data; load only locally created joblib artifacts. Batch prediction is capped at 10,000 rows with numeric and missing-value validation. Unknown categories are handled by the encoder but can reduce reliability.
