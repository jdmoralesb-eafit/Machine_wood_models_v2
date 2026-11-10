"""Estimadores de V4 (RF, SVM, GB con hiperparametro `peso`). Viven en un modulo propio para que
los modelos guardados con joblib se puedan cargar desde cualquier script (p. ej. use_trained_models.py)."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

SEED = 42


class _PesoMixin:
    """Anade el hiperparametro `peso` (None | 'balanced' | 'dos_clases') via sample_weight."""
    def _sw(self, y):
        if self.peso is None:
            return None
        if self.peso == "balanced":
            return compute_sample_weight("balanced", y)
        # dos_clases: T1.1 y T1.2 se equilibran entre si; el resto (T1.3) queda con peso 1
        y = np.asarray(y)
        sw = np.ones(len(y))
        main = [c for c in (0, 1) if (y == c).any()]
        for c in main:
            sw[y == c] = len(y[np.isin(y, main)]) / (len(main) * (y == c).sum())
        return sw

    def fit(self, X, y):
        return super().fit(X, y, sample_weight=self._sw(y))


class RF(_PesoMixin, RandomForestClassifier):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1, max_features="sqrt",
                 peso=None, random_state=SEED, n_jobs=1):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                         max_features=max_features, random_state=random_state, n_jobs=n_jobs)
        self.peso = peso


class SVM(_PesoMixin, SVC):
    def __init__(self, C=1.0, gamma="scale", peso=None, random_state=SEED):
        super().__init__(C=C, gamma=gamma, random_state=random_state)
        self.peso = peso


class GB(_PesoMixin, GradientBoostingClassifier):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0,
                 peso=None, random_state=SEED):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                         subsample=subsample, random_state=random_state)
        self.peso = peso


