# Import management
from importlib import reload

import logging
import os
import time

from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

import utils.SerializationUtils as su
reload(su)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

base_model_storage_path = '../../model/saved/'


def load_or_tune_and_evaluate_models(model_name, models, X_train, X_test, Y_train, Y_test, re_train=True):
    """
    Tunes each model in the provided dictionary, selects the best model based on accuracy,
    and returns the best model with its name and accuracy.

    Args:
        model_name (str): Name of the model to save/load.
        models (dict): Dictionary where keys are model names and values are dictionaries
                       with keys "model" and "params" for each model.
        X_train, X_test: Training and testing features.
        Y_train, Y_test: Training and testing labels.
        re_train (bool): If True, models will be retrained; otherwise, the function attempts to load saved models.

    Returns:
        best_estimator: The model with the highest accuracy after tuning.
        best_estimator_name (str): Name of the best model.
        best_accuracy (float): Best model's accuracy on the test set.
    """
    try:
        if not re_train and os.path.exists(get_model_pack_subfolder_path(model_name)):
            # Load from saved models if re_train is False
            logger.info(f"Loading {model_name} from saved.")
            best_model, all_tuned_models = load_model_pack(model_name)
            return best_model, all_tuned_models

        # Variables to track the best model and its performance
        best_estimator = None
        best_accuracy = 0
        best_estimator_name = None
        estimators = []

        # Start time for tuning all models
        all_models_tune_start_time = time.time()

        # Iterate through each model and perform hyperparameter tuning
        for estimator_name, config in models.items():
            logger.info(f"------------------------------\nStarting tuning for model: {estimator_name}")
            current_model_tune_start_time = time.time()

            try:
                # Perform GridSearchCV for hyperparameter tuning
                grid_search = GridSearchCV(config["model"], config["params"], cv=5, scoring='accuracy')
                grid_search.fit(X_train, Y_train)
                logger.info(f"Model {estimator_name} has been fitted!")

                # Evaluate the model
                best_estimator = grid_search.best_estimator_
                pred = best_estimator.predict(X_test)
                accuracy = accuracy_score(Y_test, pred)
                logger.info(f"Predictions made for {estimator_name}!")

                estimators.append({
                    "model_name": estimator_name,
                    "model": best_estimator,
                    "prediction": pred,
                })

                # Log best parameters and accuracy
                logger.info(f"Best parameters for {estimator_name}: {grid_search.best_params_}")
                logger.info(f"Accuracy for {estimator_name}: {accuracy * 100:.2f}%")
                logger.info(
                    f"Model tuning time for {estimator_name}: {time.time() - current_model_tune_start_time:.2f} seconds")

                # Update best model if current model is better
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_estimator_name = estimator_name
                    best_estimator = best_estimator

            except Exception as e:
                logger.error(f"Error while tuning model {estimator_name}: {e}")

        # Save the best model and all tuned models
        save_model_pack(best_estimator, estimators, model_name)

        # Log final results
        logger.info(f"Best model is {best_estimator_name} with an accuracy of {best_accuracy * 100:.2f}%")
        logger.info(
            f"Total hyper-parameter tuning time: {time.time() - all_models_tune_start_time:.2f} seconds for all models")

        return best_estimator, estimators

    except Exception as e:
        logger.error(f"An error occurred during model tuning or loading: {e}")
        raise


def evaluate_models(models, y_test):
    """
    Evaluates multiple models on common classification metrics and displays the results.

    Args:
        models (list): A list of dictionaries, where each dictionary contains:
                       "model_name" (str): The name of the model,
                       "model" (object): The trained model,
                       "prediction" (array-like): The predictions of the model.
        y_test (array-like): The true labels of the test dataset.
    """
    print("{:<20} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Model Name", "Accuracy", "ROC AUC", "Precision", "Recall", "F1 Score"
    ))
    print("=" * 70)

    for model_info in models:
        model_name = model_info["model_name"]
        pred = model_info["prediction"]

        # Calculate metrics
        accuracy = accuracy_score(y_test, pred)
        roc_auc = roc_auc_score(y_test, pred, multi_class='ovr')
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)

        # Print results
        print("{:<20} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}".format(
            model_name, accuracy, roc_auc, precision, recall, f1
        ))


def get_model_pack_subfolder_path(model_name):
    """
    Helper function to get the path where model data is stored.

    Parameters:
        model_name (str): The name of the model.

    Returns:
        str: The path to the model's folder.
    """
    return os.path.join(base_model_storage_path, model_name)


def save_model_pack(best_model, all_tuned_models, model_name):
    """
    Saves the best model and all tuned models to their respective serialized files.

    Parameters:
        best_model: The best model to be saved.
        all_tuned_models: All tuned models to be saved.
        model_name (str): The name of the model to construct the file paths.
    """
    try:
        model_pack_subfolder_path = get_model_pack_subfolder_path(model_name)

        # Create subfolder if it doesn't exist
        os.makedirs(model_pack_subfolder_path, exist_ok=True)
        logger.info(f"Model storage folder created or already exists: {model_pack_subfolder_path}")

        # Define paths for saving the models
        best_model_path = os.path.join(model_pack_subfolder_path, f'{model_name}_best.pkl')
        all_tuned_models_path = os.path.join(model_pack_subfolder_path, f'{model_name}_all.pkl')

        # Serialize models
        su.serialize_objects(best_model_path, best_model, overwrite=True)
        su.serialize_objects(all_tuned_models_path, all_tuned_models, overwrite=True)

        logger.info(f"Models for {model_name} saved successfully.")

    except Exception as e:
        logger.error(f"Error while saving model pack for {model_name}: {e}")
        raise


def load_model_pack(model_name):
    """
    Loads the best model and all tuned models from the storage.

    Parameters:
        model_name (str): The name of the model to load.

    Returns:
        best_model, all_tuned_models: The deserialized models.
    """
    try:
        model_pack_subfolder_path = get_model_pack_subfolder_path(model_name)

        # Define paths for loading the models
        best_model_path = os.path.join(model_pack_subfolder_path, f'{model_name}_best.pkl')
        all_tuned_models_path = os.path.join(model_pack_subfolder_path, f'{model_name}_all.pkl')

        # Deserialize models
        best_model = su.deserialize_objects(best_model_path)
        all_tuned_models = su.deserialize_objects(all_tuned_models_path)

        logger.info(f'Model "{model_name}" has been successfully loaded from storage.')

        return best_model, all_tuned_models

    except FileNotFoundError as fnf:
        logger.error(f"Error: Model pack for {model_name} not found. {fnf}")
        raise
    except Exception as e:
        logger.error(f"Error while loading model pack for {model_name}: {e}")
        raise
