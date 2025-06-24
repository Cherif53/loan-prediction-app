import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Chargement et nettoyage
df = pd.read_csv("data/loan_data.csv")
df.dropna(inplace=True)  # Supprime les lignes avec NaN
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})  # Pour la conversion en binaire


# Séparation des features et de la target
X = df[['ApplicantIncome', 'LoanAmount', 'Credit_History']]
y = df['Loan_Status']

# Entraînement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sauvegarde du modèle
joblib.dump(model, 'model.pkl')