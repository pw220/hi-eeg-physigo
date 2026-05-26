# Repository Rules for Future Codex Work

This project studies EEG-based driver fatigue source-free adaptation. The current stable stage is an EEGNet source-only LOSO baseline on SEED-VIG raw EEG.

Rules:

- Do not implement TRACE, SFDA, adaptation, pseudo-labeling, entropy minimization, Riemannian reference, or new research methods unless explicitly requested.
- Never use target labels during training, validation, normalization, clipping, class weighting, early stopping, or model selection.
- Target labels are allowed only for final evaluation and diagnostic reporting.
- All splits must be subject-wise.
- Class weights must be computed from source-training labels only.
- Normalization and clipping statistics must be computed from source-training data only.
- Save per-sample predictions for every evaluated target subject.
- Run only lightweight smoke tests locally unless explicitly requested.
- Do not commit datasets, checkpoints, outputs, logs, processed arrays, or large binary files.
- Keep code modular and avoid large refactors that break the working baseline.
