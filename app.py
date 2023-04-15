# Import dei moduli necessari
from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib


# Caricamento del modello salvato nel file pickle
model = gbm_pickle = joblib.load('model.pkl')

# Inizializzazione dell'app Flask
app = Flask(__name__)

# Definizione della vista per la homepage
@app.route('/')
def home():
    return render_template('home.html')

# Definizione della vista per la gestione della richiesta POST del form
@app.route('/predict', methods=['POST'])
def predict():
    
    # Recupero dei dati inseriti dall'utente nel form
    data = []
    data.append(float(request.form['baseline value']))
    data.append(float(request.form['accelerations']))
    data.append(float(request.form['fetal_movement']))
    data.append(float(request.form['uterine_contractions']))
    data.append(float(request.form['light_decelerations']))
    data.append(float(request.form['severe_decelerations']))
    data.append(float(request.form['prolongued_decelerations']))
    data.append(float(request.form['abnormal_short_term_variability']))
    data.append(float(request.form['mean_value_of_short_term_variability']))
    data.append(float(request.form['percentage_of_time_with_abnormal_long_term_variability']))
    data.append(float(request.form['mean_value_of_long_term_variability']))
    data.append(float(request.form['histogram_width']))
    data.append(float(request.form['histogram_min']))
    data.append(float(request.form['histogram_max']))
    data.append(float(request.form['histogram_number_of_peaks']))
    data.append(float(request.form['histogram_number_of_zeroes']))
    data.append(float(request.form['histogram_mode']))
    data.append(float(request.form['histogram_mean']))
    data.append(float(request.form['histogram_median']))
    data.append(float(request.form['histogram_variance']))
    data.append(float(request.form['histogram_tendency']))

    
    data = np.array(data)

    scaler = joblib.load('scaler.pkl')
    data_scaled = scaler.transform(data.reshape(1, -1))
    
    # Effettuazione della predizione utilizzando il modello
    model = joblib.load('model.pkl')
    prediction = model.predict(data_scaled)

    if prediction == 1:
        prediction = "Normal"
    elif prediction == 2:
        prediction = "Suspect"
    else: 
        prediction = "Pathologic"
    
    # Restituzione dell'etichetta di classe corrispondente come risposta alla richiesta POST
    return str(prediction)

# Avvio dell'app Flask
if __name__ == '__main__':
    app.run(port=3000,debug=True)
