import streamlit as st
import pandas as pd
import joblib

# Charger le modèle
model_rfc = joblib.load("model_new.joblib")

# Définir les noms des mois
month_names = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

# Définir les noms des segments de marché
market_segments = ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Complementary', 'Groups']

# Définir les noms des types de dépôt
deposit_types = ['No Deposit', 'Non Refund', 'Refundable']

# Fonction pour prétraiter les données
def preprocess_data(df):
    # Convertir le mois en format numérique
    df['arrival_date_month'] = df['arrival_date_month'].apply(lambda x: month_names.index(x) + 1)
    # Convertir le segment de marché en format numérique
    df['market_segment'] = df['market_segment'].apply(lambda x: market_segments.index(x))
    # Convertir le type de dépôt en format numérique
    df['deposit_type'] = df['deposit_type'].apply(lambda x: deposit_types.index(x))
    return df

# Fonction de prédiction
def predict_cancellation(features):
    # Prétraiter les données
    features = preprocess_data(features)
    # Faire la prédiction
    prediction = model_rfc.predict(features)
    return prediction

# Interface Streamlit
st.title("Prédiction d'annulation de réservation")
st.subheader(" Réalisée par Ano N'gozan Louis")
st.subheader(" @2023")

# Entrée des caractéristiques
st.header("Tous les champs sont obligatoires")
arrival_date_month = st.selectbox("Mois d'arrivée", month_names)
market_segment = st.selectbox("Segment de marché", market_segments)
deposit_type = st.selectbox("Type de dépôt", deposit_types)
reservation_status_month = st.number_input("Mois du statut de réservation", min_value=1, max_value=12)
reservation_status_day = st.number_input("Jour du statut de réservation", min_value=1, max_value=31)
lead_time = st.number_input("Temps d'avance (en jours)", min_value=0)
arrival_date_week_number = st.number_input("Numéro de semaine d'arrivée", min_value=1, max_value=53)
arrival_date_day_of_month = st.number_input("Jour du mois d'arrivée", min_value=1, max_value=31)
previous_cancellations = st.number_input("Annulations précédentes", min_value=0)
adr = st.number_input("ADR (Average Daily Rate)")

# Prédiction
features = pd.DataFrame({'arrival_date_month': [arrival_date_month],
                         'market_segment': [market_segment],
                         'deposit_type': [deposit_type],
                         'reservation_status_month': [reservation_status_month],
                         'reservation_status_day': [reservation_status_day],
                         'lead_time': [lead_time],
                         'arrival_date_week_number': [arrival_date_week_number],
                         'arrival_date_day_of_month': [arrival_date_day_of_month],
                         'previous_cancellations': [previous_cancellations],
                         'adr': [adr]})

if st.button("Prédire"):
    # Appeler la fonction de prédiction
    prediction = predict_cancellation(features)
    result = "Annulée" if prediction[0] == 1 else "Non annulée"
    st.success(f"La réservation est prédite comme étant : {result}")



    # Créer un DataFrame avec les caractéristiques et les valeurs prédites
    df_prediction = pd.DataFrame({'Caractéristiques': features.columns,
                                  'Valeurs': features.values[0]})
    df_prediction['Valeurs'] = df_prediction['Valeurs'].astype(str)


    # Ajout de la valeur prédite
    #df_prediction['Valeurs prédites'] = [result] * len(df_prediction)

    # Réinitialisation de l'index
    df_prediction = df_prediction.reset_index(drop=True)
    
     
     
    # Transposer le DataFrame pour l'afficher de manière verticale
    df_prediction = df_prediction.transpose() 



    # Afficher le tableau des valeurs prédites
    st.subheader("Tableau des valeurs prédites")
    st.table(df_prediction)