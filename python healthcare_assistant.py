import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import shap
import matplotlib.pyplot as plt

# Step 1: Load Dataset
print("Loading dataset...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, header=None, names=columns)

# Step 2: Prepare Data
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Train Model
print("Training model...")
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate Model
print("Evaluating model...")
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 5: Explain Predictions Using SHAP
print("Generating explanations...")
explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test)

# Visualize Global Feature Importance
shap.summary_plot(shap_values, X_test, feature_names=columns[:-1], show=False)
plt.savefig("summary_plot.png")
print("\nFeature importance plot saved as 'summary_plot.png'")

# Visualize Individual Prediction Explanation
# Note: This part may not work properly outside of a Jupyter notebook
# shap.initjs()  # Uncomment if running in a Jupyter notebook
def save_force_plot(explainer, shap_values, instance, feature_names):
    # Convert instance to a 2D array if needed
    if instance.ndim == 1:
        instance = instance.reshape(1, -1)
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0], instance,
                                 feature_names=feature_names, matplotlib=True)
    plt.savefig("force_plot.png")
    print("\nForce plot saved as 'force_plot.png'")

# Save the force plot for the first instance
save_force_plot(explainer, shap_values, X_test[0], columns[:-1])
