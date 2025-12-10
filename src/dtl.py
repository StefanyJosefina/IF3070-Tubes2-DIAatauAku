from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


@dataclass
class _Node:
    feature: Optional[int] = None
    threshold: Optional[Any] = None  
    is_categorical: bool = False  
    left: Optional[Any] = None
    right: Optional[Any] = None
    value: Optional[np.ndarray] = None 


class DTL(BaseEstimator, ClassifierMixin):
    """CART (Classification And Regression Trees) - Decision Tree Learning dari scratch.

    Implementasi pohon keputusan yang mendukung fitur numerik dan kategorik.
    Kompatibel dengan scikit-learn untuk hyperparameter tuning.
    
    Catatan Algoritma:
    - Binary splits: setiap node memisah data menjadi dua cabang (kiri/kanan)
    - Fitur numerik: split menggunakan threshold (nilai <= threshold)
    - Fitur kategorik: split menggunakan kesamaan nilai (nilai == kategori)
    - Kriteria impurity: Gini atau Entropy untuk memilih split terbaik
    - Auto-deteksi tipe fitur dari data
    - Kompatibel dengan pandas DataFrame dan numpy array

    Parameters
    ----------
    max_depth : int or None, default=None
        Kedalaman maksimum pohon.
        - None: tidak ada batasan kedalaman (hingga stopping condition lainnya)
        - int > 0: batasi kedalaman pohon (cegah overfitting)
        
        Best Practice:
        - Mulai dengan None, kemudian tune: [5, 10, 15, 20, 30]
        - Lebih kecil = model lebih sederhana, kurang overfitting
        - Untuk data besar: gunakan 5-15
        
    min_samples_split : int, default=2
        Jumlah sampel minimum yang diperlukan untuk membagi sebuah node.
        - int > 1: jika node memiliki sampel < nilai ini, tidak boleh split
        
        Best Practice:
        - Default (2) cocok untuk kebanyakan kasus
        - Gunakan lebih besar untuk data noisy: [5, 10, 20]
        - Untuk data besar (>100k): gunakan 5-20
        - Untuk data kecil (<1k): gunakan 2-5
        
    min_samples_leaf : int, default=1
        Jumlah sampel minimum di leaf node (hasil akhir split).
        - int >= 1: setiap leaf harus punya minimal sampel ini
        
        Best Practice:
        - Default (1) memungkinkan leaf sangat kecil
        - Untuk cegah overfitting: gunakan [2, 4, 8]
        - min_samples_leaf >= min_samples_split / 2 biasanya baik
        - Untuk data imbalanced: gunakan lebih besar (4-8)
        
    criterion : {'gini', 'entropy'}, default='gini'
        Metode mengukur kualitas split.
        - 'gini': Gini impurity (default, lebih cepat)
          * Formula: 1 - sum(p_i^2) untuk setiap kelas i
          * Interpretasi: probabilitas misklasifikasi jika label diassign random
          * Lebih cocok untuk dataset balanced
          
        - 'entropy': Information Gain / Entropy
          * Formula: -sum(p_i * log2(p_i)) untuk setiap kelas i
          * Interpretasi: "kekacauan" atau uncertainty
          * Lebih cocok untuk dataset imbalanced
          * Komputasi sedikit lebih lambat
        
        Best Practice:
        - Mulai dengan 'gini' (standar, lebih cepat)
        - Jika hasil kurang baik: coba 'entropy'
        - Untuk fraud detection (imbalanced): coba 'entropy'
        
    splitter : {'best', 'random'}, default='best'
        Strategi memilih split di setiap node.
        - 'best': cari split terbaik di semua fitur & threshold (exhaustive search)
          * Hasil lebih optimal, tapi lebih lambat
          
        - 'random': pilih split secara random di features yang random
          * Lebih cepat, berguna untuk Extra Trees / Bagging
          * Hasil kurang optimal tapi lebih diverse
        
        Best Practice:
        - Gunakan 'best' untuk model standalone (default)
        - Gunakan 'random' jika melatih banyak pohon (Random Forest, Extra Trees)
        
    max_features : {None, 'sqrt', 'log2'}, default=None
        Jumlah fitur yang dipertimbangkan saat mencari split terbaik.
        - None: gunakan semua fitur (default)
          * Hasil lebih optimal, tapi perlu banyak komputasi
          
        - 'sqrt': sqrt(n_features) fitur random
          * Contoh: jika 100 fitur -> pertimbangkan sqrt(100)=10 fitur
          * Berguna untuk Random Forest
          
        - 'log2': log2(n_features) fitur random
          * Contoh: jika 100 fitur -> pertimbangkan log2(100)≈6.6 fitur
          * Berguna untuk mengurangi dependensi antar fitur
        
        Best Practice:
        - Gunakan None untuk data sedikit fitur (<20)
        - Gunakan 'sqrt' atau 'log2' untuk banyak fitur (>50)
        - Untuk Extra Trees: gunakan 'sqrt' atau 'log2'
        - Untuk Dataset lebar: tuning antara None, 'sqrt', 'log2'
        
    random_state : int or None, default=None
        Seed untuk reproducibility (ketika splitter='random' atau max_features random).
        - None: hasil random tiap run (non-deterministic)
        - int: hasil sama jika dijalankan berkali-kali (reproducible)
        
        Best Practice:
        - Set random_state=42 untuk reproducible results saat development
        - Biarkan None di production untuk diversity (terutama dengan Random Forest)
        
    Examples
    --------
    >>> from dtl import DTL
    >>> import numpy as np
    >>> 
    >>> # Data dummy
    >>> X = np.array([[2.5, 'A'], [5.5, 'B'], [3.0, 'A']], dtype=object)
    >>> y = np.array([0, 1, 0])
    >>> 
    >>> # Model default
    >>> dtl = DTL()
    >>> dtl.fit(X, y)
    >>> dtl.predict([[3.0, 'A']])  # array([0])
    >>> 
    >>> # Model dengan tuning
    >>> dtl_tuned = DTL(
    ...     max_depth=5,
    ...     min_samples_split=5,
    ...     min_samples_leaf=2,
    ...     criterion='entropy',
    ...     random_state=42
    ... )
    >>> dtl_tuned.fit(X, y)
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        splitter: str = "best",
        max_features: Optional[str] = None,
        random_state: Optional[int] = None,
    ):
        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.criterion = criterion
        self.splitter = splitter
        self.max_features = max_features
        self.random_state = random_state
        
        # Internal: menyimpan tipe fitur (numeric atau categorical)
        self._feature_types = None 

    def fit(self, X, y):
        """Latih pohon keputusan dari data (X, y).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data input dengan fitur numerik dan/atau kategorik.
            - Bisa berupa pandas DataFrame atau numpy array
            - Tipe fitur (numerik/kategorik) otomatis terdeteksi
            
        y : array-like of shape (n_samples,)
            Target label untuk klasifikasi (binary atau multi-class).
            
        Returns
        -------
        self : DTL
            Mengembalikan self untuk method chaining (sklearn compatible)
        
        Catatan
        -------
        - Fitur numerik: diperlakukan sebagai continuous (split dengan threshold)
        - Fitur kategorik: diperlakukan sebagai discrete (split dengan equality)
        - Tipe deteksi otomatis berdasarkan dtype data
        - Missing values: akan menyebabkan error (silakan handle sebelum fit())
        """
        # Konversi X ke numpy array jika pandas DataFrame
        if hasattr(X, 'to_numpy'):
            X_array = X.to_numpy()
            self._feature_types = {}
            for col_idx, col in enumerate(X.columns):
                col_data = X.iloc[:, col_idx]
                is_cat = not np.issubdtype(col_data.dtype, np.number)
                self._feature_types[col_idx] = 'categorical' if is_cat else 'numeric'
        else:
            X_array = np.asarray(X)
            # Auto-deteksi tipe fitur dari data numpy
            self._feature_types = {}
            for col_idx in range(X_array.shape[1]):
                col_data = X_array[:, col_idx]
                try:
                    col_data.astype(np.float64)
                    self._feature_types[col_idx] = 'numeric'
                except (ValueError, TypeError):
                    self._feature_types[col_idx] = 'categorical'
        
        y = np.asarray(y)
        
        if X_array.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        y_encoded = np.searchsorted(self.classes_, y)
        
        self._tree = self._build_tree(X_array, y_encoded, depth=0)
        return self

    def _impurity(self, y: np.ndarray) -> float:
        """Compute impurity (gini or entropy) for labels y."""
        if y.size == 0:
            return 0.0
        
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        probs = counts / counts.sum()
        
        if self.criterion == "gini":
            return 1.0 - np.sum(probs ** 2)
        else:  
            ps = probs[probs > 0]
            return -np.sum(ps * np.log2(ps))

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        
        if n_samples < 2 or len(np.unique(y)) == 1:
            return None, None, 0.0, False
        
        base_impurity = self._impurity(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        best_is_categorical = False
        
        if self.max_features is None:
            features_to_try = range(n_features)
        elif self.max_features == "sqrt":
            n_try = max(1, int(np.sqrt(n_features)))
            features_to_try = np.random.choice(n_features, size=n_try, replace=False)
        elif self.max_features == "log2":
            n_try = max(1, int(np.log2(n_features)))
            features_to_try = np.random.choice(n_features, size=n_try, replace=False)
        else:
            features_to_try = range(n_features)
        
        for feature in features_to_try:
            values = X[:, feature]
            is_categorical = self._feature_types.get(feature, 'numeric') == 'categorical'
            
            if is_categorical:
                unique_vals = np.unique(values)
                if unique_vals.size < 2:
                    continue
                
                for category in unique_vals:
                    left_mask = values == category
                    right_mask = values != category
                    
                    y_left = y[left_mask]
                    y_right = y[right_mask]
                    
                    if y_left.size == 0 or y_right.size == 0:
                        continue
                    
                    n_left = y_left.size
                    n_right = y_right.size
                    impurity_left = self._impurity(y_left)
                    impurity_right = self._impurity(y_right)
                    
                    weighted_impurity = (n_left / n_samples) * impurity_left + (n_right / n_samples) * impurity_right
                    gain = base_impurity - weighted_impurity
                    
                    if gain > best_gain:
                        best_gain = float(gain)
                        best_feature = feature
                        best_threshold = category
                        best_is_categorical = True
            else:
                try:
                    numeric_vals = values.astype(np.float64)
                except (ValueError, TypeError):
                    continue
                
                unique_vals = np.unique(numeric_vals)
                
                if unique_vals.size < 2:
                    continue
                
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
                
                for threshold in thresholds:
                    left_mask = numeric_vals <= threshold
                    right_mask = ~left_mask
                    
                    y_left = y[left_mask]
                    y_right = y[right_mask]
                    
                    if y_left.size == 0 or y_right.size == 0:
                        continue
                    
                    n_left = y_left.size
                    n_right = y_right.size
                    impurity_left = self._impurity(y_left)
                    impurity_right = self._impurity(y_right)
                    
                    weighted_impurity = (n_left / n_samples) * impurity_left + (n_right / n_samples) * impurity_right
                    gain = base_impurity - weighted_impurity
                    
                    if gain > best_gain:
                        best_gain = float(gain)
                        best_feature = feature
                        best_threshold = threshold
                        best_is_categorical = False
        
        return best_feature, best_threshold, best_gain, best_is_categorical

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        node = _Node()
        n_samples = y.size
        unique_labels = np.unique(y)
        
        stopping = (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or len(unique_labels) == 1
        )
        
        if stopping:
            counts = np.bincount(y, minlength=self.n_classes_).astype(float)
            node.value = counts / counts.sum()
            return node
        
        if self.splitter == "random":
            n_features = X.shape[1]
            feat = np.random.randint(0, n_features)
            values = X[:, feat]
            is_cat = self._feature_types.get(feat, 'numeric') == 'categorical'
            unique_vals = np.unique(values)
            
            if unique_vals.size < 2:
                counts = np.bincount(y, minlength=self.n_classes_).astype(float)
                node.value = counts / counts.sum()
                return node
            
            if is_cat:
                category = unique_vals[np.random.randint(0, len(unique_vals))]
                best_feature, best_threshold, best_is_categorical = feat, category, True
            else:
                unique_vals = unique_vals.astype(np.float64)
                thr = (unique_vals[np.random.randint(0, len(unique_vals) - 1)] + 
                       unique_vals[np.random.randint(1, len(unique_vals))]) / 2.0
                best_feature, best_threshold, best_is_categorical = feat, thr, False
        else:  
            best_feature, best_threshold, gain, best_is_categorical = self._best_split(X, y)
        
        if best_feature is None:
            counts = np.bincount(y, minlength=self.n_classes_).astype(float)
            node.value = counts / counts.sum()
            return node
        
        if best_is_categorical:
            left_mask = X[:, best_feature] == best_threshold
            right_mask = X[:, best_feature] != best_threshold
        else:
            left_mask = X[:, best_feature].astype(np.float64) <= best_threshold
            right_mask = ~left_mask
        
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            counts = np.bincount(y, minlength=self.n_classes_).astype(float)
            node.value = counts / counts.sum()
            return node
        
        node.feature = best_feature
        node.threshold = best_threshold
        node.is_categorical = best_is_categorical
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediksi label kelas untuk setiap sampel.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data input untuk prediksi.
            - Harus memiliki fitur yang sama saat fit()
            - Bisa berupa pandas DataFrame atau numpy array
            
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Prediksi label kelas untuk setiap sampel.
            - Nilai merupakan original class labels dari training data
            
        Catatan
        -------
        - Prediksi berdasarkan class dengan probabilitas tertinggi di leaf node
        - Untuk mendapatkan probabilitas, gunakan predict_proba()
        - Jika ingin adjust threshold, gunakan predict_proba() terlebih dahulu
        
        Examples
        --------
        >>> dtl = DTL(max_depth=5, criterion='entropy')
        >>> dtl.fit(X_train, y_train)
        >>> predictions = dtl.predict(X_test)  # shape: (n_samples,)
        >>> # Hitung accuracy
        >>> accuracy = (predictions == y_test).mean()
        """
        # Konversi X ke numpy array jika pandas DataFrame
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
        else:
            X = np.asarray(X)
        
        predictions = []
        
        for sample in X:
            node = self._tree
            while node.value is None:
                feature_value = sample[node.feature]
                
                if node.is_categorical:
                    # Fitur kategorik: compare equality
                    if feature_value == node.threshold:
                        node = node.left
                    else:
                        node = node.right
                else:
                    # Fitur numerik: compare dengan threshold
                    try:
                        numeric_value = float(feature_value)
                    except (ValueError, TypeError):
                        # Jika tidak bisa convert ke numeric, ambil right child
                        node = node.right
                        continue
                    
                    if numeric_value <= node.threshold:
                        node = node.left
                    else:
                        node = node.right
            
            # Ambil kelas dengan probabilitas tertinggi
            class_idx = np.argmax(node.value)
            predictions.append(self.classes_[class_idx])
        
        return np.array(predictions)

    def predict_proba(self, X):
        """Prediksi probabilitas kelas untuk setiap sampel.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data input untuk prediksi.
            - Harus memiliki fitur yang sama saat fit()
            - Bisa berupa pandas DataFrame atau numpy array
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Probabilitas untuk setiap kelas.
            - Setiap baris adalah sampel
            - Setiap kolom adalah kelas
            - Nilai berkisar [0, 1], sum per baris = 1
            
        Catatan
        -------
        - Probabilitas diambil dari distribusi kelas di leaf node
        - Berguna untuk threshold adjustment pada classification
        - Lebih informatif daripada hard prediction (predict)
        
        Examples
        --------
        >>> dtl = DTL(max_depth=5)
        >>> dtl.fit(X_train, y_train)
        >>> proba = dtl.predict_proba(X_test)  # shape: (n_samples, 2)
        >>> # Ambil probabilitas kelas 1 (fraud)
        >>> fraud_proba = proba[:, 1]
        """
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
        else:
            X = np.asarray(X)
        
        proba = []
        
        for sample in X:
            node = self._tree
            while node.value is None:
                feature_value = sample[node.feature]
                
                if node.is_categorical:
                    if feature_value == node.threshold:
                        node = node.left
                    else:
                        node = node.right
                else:
                    try:
                        numeric_value = float(feature_value)
                    except (ValueError, TypeError):
                        node = node.right
                        continue
                    
                    if numeric_value <= node.threshold:
                        node = node.left
                    else:
                        node = node.right
            
            proba.append(node.value)
        
        return np.vstack(proba)


__all__ = ["DTL"]
