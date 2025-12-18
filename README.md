# Customer Churn Prediction

A machine learning project that predicts customer churn for e-commerce businesses using various classification algorithms and ensemble methods.

## 📋 Project Overview

This project implements multiple machine learning models to predict customer churn based on e-commerce customer data. The models analyze customer behavior, purchase patterns, and demographics to identify customers at risk of churning.

## 🎯 Features

- **Multiple Classification Models**:
  - Logistic Regression
  - Decision Tree Classifier
  - LightGBM Classifier
  - CatBoost Classifier
- **Ensemble Learning**:
  - Stacking Classifier combining multiple base models
  - Voting Classifier for improved predictions
- **Class Imbalance Handling**:

  - SMOTE (Synthetic Minority Over-sampling Technique) implementation
  - Class weight balancing

- **Comprehensive Evaluation**:
  - Accuracy metrics
  - Confusion matrices
  - Detailed classification reports

## 📊 Dataset

The project uses `ecommerce_customer_data_custom_ratios.csv` containing customer information including:

- Customer demographics (Gender, Age, etc.)
- Purchase behavior (Product Category, Payment Method)
- Transaction history
- Churn status (target variable)

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries**:
  - pandas - Data manipulation and analysis
  - scikit-learn - Machine learning algorithms and utilities
  - LightGBM - Gradient boosting framework
  - CatBoost - Gradient boosting library
  - imbalanced-learn - SMOTE implementation

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── ensemble_code.py                          # Ensemble methods with SMOTE
├── dec_tree.py                               # Decision Tree classifier
├── log_reg.py                                # Logistic Regression classifier
├── ecommerce_customer_data_custom_ratios.csv # Dataset
├── ensemble_code_with_smote.txt              # Documentation
├── catboost_info/                            # CatBoost training logs
│   ├── catboost_training.json
│   ├── learn_error.tsv
│   ├── time_left.tsv
│   └── learn/
└── OUTPUTS/                                  # Model outputs and results
```

## 🚀 Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install pandas scikit-learn lightgbm catboost imbalanced-learn
```

### Installation

1. Clone the repository:

```bash
git clone https://github.com/NJVinay/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

2. Ensure the dataset is in the project directory

### Usage

#### Run Decision Tree Model

```bash
python dec_tree.py
```

#### Run Logistic Regression Model

```bash
python log_reg.py
```

#### Run Ensemble Model (Recommended)

```bash
python ensemble_code.py
```

## 📈 Model Pipeline

1. **Data Preprocessing**:

   - Load customer data
   - Drop unnecessary columns (Customer ID, Purchase Date, Customer Name)
   - One-hot encode categorical variables (Product Category, Payment Method, Gender)
   - Handle missing values

2. **Feature Engineering**:

   - Separate features (X) and target variable (y)
   - Split data into training (70%) and testing (30%) sets

3. **Class Imbalance Handling**:

   - Apply SMOTE to balance the training dataset

4. **Model Training**:

   - Train individual base models
   - Combine models using Stacking Classifier with 5-fold cross-validation
   - Use Logistic Regression as the meta-model

5. **Evaluation**:
   - Generate predictions on test set
   - Calculate accuracy, confusion matrix, and classification report

## 📊 Model Performance

The ensemble approach using Stacking Classifier with SMOTE demonstrates:

- Improved handling of imbalanced classes
- Better generalization through model combination
- Enhanced prediction accuracy

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**NJVinay**

- GitHub: [@NJVinay](https://github.com/NJVinay)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project is for educational and research purposes. Ensure compliance with data privacy regulations when using customer data in production environments.
