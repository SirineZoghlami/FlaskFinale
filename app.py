from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
with open('best_modelefinale.pkl', 'rb') as file:
    model = pickle.load(file)

# Define API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    pression_1 = data['pression_1']
    heurefonction = data['heurefonction']
    pression_2 = data['pression_2']
    temperature = data['temperature']
    pointderose = data['pointderose']
    tauxdecharge = data['tauxdecharge']
    id = data['id']
    local_id = data['local_id']
    
    # Make prediction
    input_data = [[id, local_id, pression_1, heurefonction, pression_2, temperature, pointderose, tauxdecharge]]
    predicted_debit = model.predict(input_data)[0]
    
    # Calculate total production
    total_production = predicted_debit * heurefonction
    
    # Return prediction and total production as JSON response
    return jsonify({
        'debit_de_production_air': predicted_debit,
        'total_production': total_production
    })

if __name__ == '__main__':
    app.run(debug=True)
