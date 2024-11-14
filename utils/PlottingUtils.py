# Data Visualization
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


def plot_barplot_distribution(df, column_name, title='Category Distribution'):
    """
    Displays the value distribution for a categorical column.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    column_name (str): Name of the categorical column to analyze.
    title (str): Title of the plot.

    Returns:
    None
    """
    print(f'Recorded entries: {df.shape[0]}')

    # Count the values in the specified column
    category_counts = df[column_name].value_counts()

    # Create a bar plot for the counts
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')

    # Add title and axis labels
    plt.title(title, fontsize=16, pad=15)
    plt.xlabel(column_name, fontsize=14)
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

    plt.figure(figsize=(1.2 * len(columns), 0.8 * len(columns)))
    sns.heatmap(
        corr,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.title("Correlation Matrix", fontsize=16, pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_roc_curve_model(model, X_test, Y_test):
    """
    Plots the ROC curve for the model.

    Args:
        model: The model to plot the ROC curve for.
        X_test, Y_test: Testing data.
    """
    # Predict probabilities for the positive class
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(Y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


def plot_confusion_matrix_model(model, X_test, Y_test):
    """
    Plots the confusion matrix for the model.

    Args:
        model: The model to plot the confusion matrix for.
        X_test, Y_test: Testing data.
    """
    pred = model.predict(X_test)
    cm = confusion_matrix(Y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_feature_importance(model, feature_names):
    """
    Plots feature importance for models that have a 'feature_importances_' attribute.

    Args:
        model: The model to plot feature importance for.
        feature_names: List of feature names for labeling.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.show()
    else:
        print("The selected model does not support feature importance plotting.")
