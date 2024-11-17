# Data Visualization
import math
import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix


def plot_pie_distribution(df, categorical_columns, title='Distribution of'):
    num_columns = len(categorical_columns)
    fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(6, 4 * num_columns))
    for i, col in enumerate(categorical_columns):
        category_counts = df[col].value_counts()
        colors = plt.cm.Paired(range(len(category_counts)))
        axes[i].pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
        axes[i].set_title(title + ' ' + col)
        axes[i].axis('equal')
    plt.tight_layout()
    plt.show()


def plot_hist_distribution(df, columns):
    num_cols = len(columns)
    num_rows = math.ceil(num_cols / 2)

    fig, axs = plt.subplots(num_rows, 2, figsize=(18, 4 * num_rows), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    for ax, col in zip(axs.ravel(), columns):
        random_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        sns.histplot(data=df, x=col, ax=ax, color=random_color, kde=True)
        ax.set_title(f'Histogram of {col}', fontsize=20)
        ax.set_xlabel(col, fontsize=17)
        ax.set_ylabel('Count', fontsize=17)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)

    for ax in axs.ravel()[num_cols:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_boxplots_distribution(df, columns):
    num_cols = len(columns)
    num_rows = math.ceil(num_cols / 4)

    fig, axs = plt.subplots(num_rows, 4, figsize=(18, 4 * num_rows), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    for ax, col in zip(axs.ravel(), columns):
        random_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        sns.boxplot(data=df, y=col, ax=ax, color=random_color)
        ax.set_title(f'Boxplot of {col}', fontsize=16)

    for ax in axs.ravel()[num_cols:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_barplot_distribution(column, title='Category Distribution'):
    """
    Displays the value distribution for a categorical column.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    column_name (str): Name of the categorical column to analyze.
    title (str): Title of the plot.

    Returns:
    None
    """

    # Count the values in the specified column
    category_counts = column.value_counts()

    # Create a bar plot for the counts
    plt.figure(figsize=(10, 6))
    sns.barplot(hue=category_counts.index, y=category_counts.values, palette='viridis', legend=False)

    # Add title and axis labels
    plt.title(title, fontsize=16, pad=15)
    plt.xlabel(column.name, fontsize=14)
    plt.ylabel('Count', fontsize=14)

    # Display values on top of each bar
    for i, v in enumerate(category_counts.values):
        plt.text(i, v + max(category_counts.values) * 0.02, str(v), ha='center', fontweight='bold', fontsize=12)

    # Adjust axis tick labels for readability
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, columns, cmap="coolwarm"):
    corr = df[columns].corr()

    sns.heatmap(
        corr,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.tight_layout()
    plt.show()


def plot_roc_curves(models, X_test, Y_test):
    """
    Plots the ROC curves for multiple models.

    Args:
        models (dict): Dictionary where keys are model names and values are model instances.
        X_test (array-like): Testing features.
        Y_test (array-like): True labels for the test set.
    """
    plt.figure(figsize=(10, 8))

    # Loop through models to compute and plot each ROC curve
    for model_dict in models:
        model_name = model_dict['model_name']
        model = model_dict['model']

        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_pred_proba = model.decision_function(X_test)
        else:
            raise ValueError(f"Model {model_name} does not have predict_proba or decision_function.")

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(Y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

    # Plot the diagonal (random guess) line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

    # Configure the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()


def plot_confusion_matrix_models(models, X_test, Y_test):
    """
    Plots the confusion matrix for multiple models.

    Args:
        models (list): A list of dictionaries, where each dictionary contains:
                       "model_name" (str): The name of the model,
                       "model" (object): The trained model.
        X_test: Testing features.
        Y_test: Testing labels.
    """
    # Ensure Y_test is numeric

    for model_info in models:
        model_name = model_info["model_name"]
        model = model_info["model"]

        # Predict and ensure predictions are numeric
        pred = model.predict(X_test)

        # Compute confusion matrix
        cm = confusion_matrix(Y_test, pred)

        # Plot confusion matrix
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix for {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()


def plot_feature_importances(models, feature_names):
    """
    Plots feature importances for multiple models provided in a dictionary format.

    Args:
        models (list): List of dictionaries, each containing:
                       - "model_name": Name of the model (str).
                       - "model": The model instance.
        feature_names (list): List of feature names for labeling.
    """
    for model_entry in models:
        model_name = model_entry["model_name"]
        model = model_entry["model"]

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = importances.argsort()[::-1]

            plt.figure(figsize=(10, 6))
            plt.title(f"Feature Importances: {model_name}")
            plt.bar(range(len(indices)), importances[indices], align="center")
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
            plt.show()
        else:
            print(f"The model '{model_name}' does not support feature importance plotting.")
