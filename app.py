from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado y el codificador de equipos
model = joblib.load('models/model.pkl')
team_encoder = joblib.load('models/team_encoder.pkl')

# Función de predicción
def predict_result(home_team, away_team, home_rank, away_rank):
    # Codificar los equipos
    home_encoded = team_encoder.transform([home_team])[0]
    away_encoded = team_encoder.transform([away_team])[0]
    
    # Crear los datos de entrada
    X = np.array([[home_encoded, away_encoded, home_rank, away_rank]])
    
    # Hacer la predicción
    result = model.predict(X)[0]
    print(f"Resultado crudo del modelo: {result}")  # Debug

    # Normalizar tipo de dato
    result_str = str(result).lower()

    # Interpretar
    if result_str in ['win', '1']:  # Local gana
        winner = home_team
        loser = away_team
    elif result_str in ['loss', '0']:  # Visitante gana
        winner = away_team
        loser = home_team
    elif result_str in ['draw', '2']:  # Empate
        winner = "Empate"
        loser = "Empate"
    else:
        winner = "Desconocido"
        loser = "Desconocido"

    return result_str, winner, loser

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    winner = None
    loser = None
    home_team = None
    away_team = None

    if request.method == "POST":
        home_team = request.form["home_team"]
        away_team = request.form["away_team"]
        home_rank = int(request.form["home_rank"])
        away_rank = int(request.form["away_rank"])
        
        # Obtener predicción
        prediction, winner, loser = predict_result(home_team, away_team, home_rank, away_rank)
    
    return render_template("index.html",
                           prediction=prediction,
                           winner=winner,
                           loser=loser,
                           home_team=home_team,
                           away_team=away_team)

if __name__ == "__main__":
    app.run(debug=True)
