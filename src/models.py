"""
Machine Learning Models Definition Module.
Uses Scikit-Learn for classical ML models and TensorFlow for a lightweight
Linear Classifier (strictly single-layer Logistic Regression, no Deep Learning).
"""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def get_sklearn_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns a dictionary of classical Scikit-Learn classifiers.
    No deep learning is used.
    """
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)
    dt = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=120, max_depth=6, min_samples_split=6, random_state=random_state)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=random_state)
    svc = SVC(probability=True, kernel='rbf', C=1.0, random_state=random_state)
    knn = KNeighborsClassifier(n_neighbors=9)

    # Stacking ensemble combining complementary classical models
    stacking = StackingClassifier(
        estimators=[
            ('lr', lr),
            ('rf', rf),
            ('gb', gb)
        ],
        final_estimator=LogisticRegression(max_iter=500, random_state=random_state),
        cv=5
    )

    return {
        "Logistic Regression": lr,
        "Decision Tree": dt,
        "Random Forest": rf,
        "Gradient Boosting": gb,
        "Support Vector Machine": svc,
        "K-Nearest Neighbors": knn,
        "Stacking Ensemble": stacking
    }


def build_tf_linear_model(input_dim: int):
    """
    Constructs a lightweight single-layer Linear Classifier using TensorFlow.
    Notice: Strictly single-layer (input -> dense 1 with sigmoid) which represents
    Generalized Linear Model (Logistic Regression) in TensorFlow tensor framework.
    Strictly NO Deep Learning (zero hidden layers).
    """
    try:
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    except ImportError:
        return None
