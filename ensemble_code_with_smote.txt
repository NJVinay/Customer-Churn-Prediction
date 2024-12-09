import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv("ecommerce_customer_data_custom_ratios.csv", sep=';')

# Print columns to verify
print("Available columns in dataset:", df.columns)

# Drop unnecessary columns if they exist
columns_to_drop = ['Customer ID', 'Purchase Date', 'Customer Name']
df_clean = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# One-hot encode categorical variables such as 'Product Category' and 'Payment Method'
df_clean = pd.get_dummies(df_clean, columns=['Product Category', 'Payment Method', 'Gender'], drop_first=True)

# Drop rows with missing values
df_clean = df_clean.dropna()

# Separate features (X) and target (y)
X = df_clean.drop(columns=['Churn'])
y = df_clean['Churn']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------- Apply SMOTE to handle class imbalance ----------
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# ---------- Base Models ----------

# Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)

# Decision Tree Classifier
tree_model = DecisionTreeClassifier(random_state=42)

# LightGBM Classifier
lgbm_model = lgb.LGBMClassifier(random_state=42)

# CatBoost Classifier
catboost_model = CatBoostClassifier(iterations=100, random_state=42, verbose=0)

# ---------- Ensemble Model (Stacking Classifier) ----------
# Defining the base models
estimators = [
    ('logistic', logistic_model),
    ('tree', tree_model),
    ('lgbm', lgbm_model),
    ('catboost', catboost_model)
]

# Defining the stacking classifier with logistic regression as the meta-model
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5  # 5-fold cross-validation
)

# Train the Stacking Classifier on SMOTE-adjusted data
stacking_model.fit(X_train_smote, y_train_smote)

# Make predictions
y_pred_stack = stacking_model.predict(X_test)

# Calculate accuracy and other metrics
accuracy_stack = accuracy_score(y_test, y_pred_stack)
conf_matrix_stack = confusion_matrix(y_test, y_pred_stack)
class_report_stack = classification_report(y_test, y_pred_stack)

# Output results
print("Stacking Classifier Accuracy:", accuracy_stack)
print("Confusion Matrix:\n", conf_matrix_stack)
print("Classification Report:\n", class_report_stack)
