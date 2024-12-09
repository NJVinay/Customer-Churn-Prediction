import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('ecommerce_customer_data_custom_ratios.csv')

# Drop unnecessary columns
df_clean = df.drop(columns=['Customer ID', 'Purchase Date', 'Customer Name'])

# One-hot encode categorical variables such as 'Product Category' and 'Payment Method'
df_clean = pd.get_dummies(df_clean, columns=['Product Category', 'Payment Method', 'Gender'], drop_first=True)

# Drop rows with missing values
df_clean = df_clean.dropna()

# Separate features (X) and target (y)
X = df_clean.drop(columns=['Churn'])
y = df_clean['Churn']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy and other metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Output results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)