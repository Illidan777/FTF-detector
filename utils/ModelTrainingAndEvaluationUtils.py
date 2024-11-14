import joblib

from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

LOGISTIC_REGRESSION = 'LogisticRegression'
GAUSSIAN_NB = 'GaussianNB'
KNN_CLASSIFIER = 'KNeighborsClassifier'
RANDOM_FOREST = 'RandomForest'
DECISION_TREE_CLASSIFIER = 'DecisionTreeClassifier'

models_dictionary = {
    LOGISTIC_REGRESSION: {
        "model": LogisticRegression(random_state=101, multi_class='auto'),
        "params": {
            "solver": ['newton-cg', 'lbfgs', 'liblinear'],
            "C": [0.1, 1, 10]
        }
    },
    GAUSSIAN_NB: {
        "model": GaussianNB(),
        "params": {
            "var_smoothing": [1e-09, 1e-08, 1e-07, 1e-06]
        }
    },
    KNN_CLASSIFIER: {
        "model": neighbors.KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ['uniform', 'distance'],
            "metric": ['euclidean', 'manhattan']
        }
    },
    RANDOM_FOREST: {
        "model": RandomForestClassifier(random_state=101),
        "params": {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    },
    DECISION_TREE_CLASSIFIER: {
        "model": DecisionTreeClassifier(),
        "params": {
            "criterion": ["gini", "entropy"],
            "max_depth": list(range(2, 25, 1)),
            "min_samples_leaf": list(range(1, 20, 1))
        }
    }
}

base_model_storage_path = '../model/saved/'


def load_or_tune_and_evaluate_models(model_name, models, X_train, X_test, Y_train, Y_test, re_train=True):
    """
    Tunes each model in the provided dictionary, selects the best model based on accuracy,
    and returns the best model with its name and accuracy.

    Args:
        models (dict): Dictionary where keys are model names and values are dictionaries
                       with keys "model" and "params" for each model.
        X_train, X_test: Training and testing features.
        Y_train, Y_test: Training and testing labels.

    Returns:
        best_model: The model with the highest accuracy after tuning.
        best_model_name (str): Name of the best model.
        best_accuracy (float): Best model's accuracy on the test set.
    """

    if not re_train:
        # Load from saved
        print(f"Loading {model_name} from saved.")
        return load_model(model_name)

    best_model = None
    best_accuracy = 0
    best_model_name = None

    for model_name, config in models.items():
        grid_search = GridSearchCV(config["model"], config["params"], cv=5, scoring='accuracy')
        grid_search.fit(X_train, Y_train)

        best_estimator = grid_search.best_estimator_
        pred = best_estimator.predict(X_test)
        accuracy = accuracy_score(Y_test, pred)

        print(f'Best parameters for {model_name}: {grid_search.best_params_}')
        print(f'Accuracy for {model_name}: {accuracy * 100:.2f}%')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = best_estimator
            best_model_name = model_name

    save_model(best_model, model_name)
    print(f'\nBest model is {best_model_name} with an accuracy of {best_accuracy * 100:.2f}%')
    return best_model


def save_model(model, model_name):
    """
    Saves the best model to a file.

    Args:
        model: The model to save.
        model_name (str): The name of the model, used for the filename.
    """
    filename = f'{base_model_storage_path}{model_name}.pkl'
    joblib.dump(model, filename)
    print(f'The best model ({model_name}) has been saved as "{filename}".')


def load_model(model_name):
    """
    Loads a model from a file.

    Args:
        model_name (str): The name of the model file to load (without extension).

    Returns:
        The loaded model.
    """
    filename = f'{base_model_storage_path}{model_name}.pkl'
    try:
        model = joblib.load(filename)
        print(f'Model "{model_name}" has been successfully loaded from "{filename}".')
        return model
    except FileNotFoundError:
        print(f'Error: The model file "{filename}" does not exist.')
    except Exception as e:
        print(f'Error loading the model: {e}')
