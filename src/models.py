"""The five classifiers required by the assignment."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from .config import SEED


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, random_state=SEED),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
    }
