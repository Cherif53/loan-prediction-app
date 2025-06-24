import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import streamlit as st

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

# Chargement du modèle
model = joblib.load('model.pkl')

# Interface
st.title("Prédiction de Prêt Bancaire 🏦")
st.write("Entrez vos informations pour savoir si votre prêt sera accepté.")

# Inputs utilisateur
income = st.number_input("Revenu mensuel ($)", min_value=0)
loan_amount = st.number_input("Montant du prêt ($)", min_value=0)
credit_history = st.radio("Avez-vous un bon historique de crédit ?", ("Oui", "Non"))
credit_history = 1 if credit_history == "Oui" else 0

# Bouton de prédiction
if st.button("Prédire"):
    prediction = model.predict([[income, loan_amount, credit_history]])[0]
    if prediction == 1:
        st.success("✅ Prêt accepté !")
    else:
        st.error("❌ Prêt refusé.")
        
        