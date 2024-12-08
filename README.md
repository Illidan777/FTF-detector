# FTF-detector - financial transactions fraud detector

### 📌 **Overview**  
This project is dedicated to developing a machine learning model to
detect fraud in financial transactions. It aims to provide an 
effective solution for securing financial operations by utilizing
modern machine learning algorithms and data analysis techniques.  

---

### 🎯 **Project Goal**  
To develop and implement a machine learning model for detecting 
anomalies in financial transactions, thereby reducing fraud risks 
and enhancing the security of financial systems.  

---

### 🧩 **Key Processes**  
- Data preprocessing and cleaning.  
- Statistical analysis and data visualization.  
- Development and using machine learning models (both supervised and unsupervised).  
- Hyperparameter optimization.  
- Evaluation of models.  
- Saving and loading models for future use.  

---

### ⚙️ **Technologies and Tools**  
- **Programming Language**: Python  
- **Development Environments**: PyCharm, Jupyter Notebook  
- **Key Libraries**:  
  - **Scikit-learn**: For model building and hyperparameter tuning.  
  - **Pandas, NumPy**: For data manipulation.  
  - **Matplotlib, Seaborn**: For data visualization.  
  - **JobLib, Pickle**: For model serialization.  

---

### Project Structure

The project is organized into several directories, each serving a specific purpose in the machine learning pipeline.

#### Directory Overview:

- **`data/`**: 
  - This folder contains the datasets used for training and testing the models. It also includes snapshots of functions and stages (checkpoint data) to save the intermediate results and facilitate resuming the workflow.
  
- **`model/`**:
  - **`stages/`**: Contains all the Jupyter notebooks that define the stages of the model development process. Each notebook represents a different phase of model building, such as data preprocessing, feature engineering, and model training.
  - **`saved/`**: Stores the trained models along with the necessary parameters. These models are saved after training to be loaded and used for future predictions or evaluation.

- **`services/`**:
  - This directory includes various services that handle specific tasks, such as optimizing the workflow, managing data pipelines, and integrating different stages of model building.
  
- **`utils/`**:
  - Contains utility classes and helper functions used across the project. These are general-purpose tools that support the core functionality, such as data transformation, logging, or visualization.

---

### 🛠️ **Setup Instructions**  

1. **Clone project**:
2. **Open the project dir**
3. **Open folder with run script**
```bash
cd /model/stages
```
4. **Execute run script. Output will be presented in `/model/stages` folder in Jupyter Notebook files.**
```bash
jupyter nbconvert --to notebook --execute run.ipynb
```

---

### Model Performance Metrics

Below are the evaluation metrics for different machine learning models used in the project. These metrics help assess the models' effectiveness in detecting fraudulent financial transactions.

| Model Name              | Accuracy | ROC AUC | Precision | Recall | F1 Score |
|-------------------------|----------|---------|-----------|--------|----------|
| LogisticRegression      | 0.67     | 0.65    | 0.98      | 0.67   | 0.79     |
| GaussianNB              | 0.76     | 0.62    | 0.97      | 0.76   | 0.85     |
| KNeighborsClassifier    | 0.58     | 0.57    | 0.97      | 0.58   | 0.72     |
| RandomForest            | 0.71     | 0.72    | 0.98      | 0.71   | 0.82     |
| DecisionTreeClassifier  | 0.66     | 0.68    | 0.98      | 0.66   | 0.78     |

---

### Workflow Optimization Service

This service is designed to optimize the workflow of model building stages in machine learning projects. It aims to streamline the process, especially for tasks that can take a significant amount of time, such as model training.

In typical workflows, when returning to a previous point in the process, it is often necessary to re-run all prior stages, which can be time-consuming. This service mitigates that issue by providing an efficient way to handle each stage.

#### Key Features:
- **Stage Management:** For each stage (or notebook), an instance of this service is created, and the current and previous stage names are passed to it.
- **Reusing Results:** It allows running the previous stage or retrieving data from it if it has already been executed and saved.
- **Efficient Execution:** The service executes or retrieves results from specific heavyweight functions at each stage.
- **Data Persistence:** It saves necessary data to be passed to the subsequent stage, reducing redundancy and saving time.

This service ensures that repetitive processes are avoided and that you can easily resume work without redoing time-consuming tasks, optimizing both time and computational resources.
