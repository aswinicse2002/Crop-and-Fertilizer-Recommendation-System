from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv('Crop and fertilizer dataset.csv')

# Define OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(dataset[['District_Name', 'Soil_color']])

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X_encoded, dataset['Crop'], test_size=0.2, random_state=42)
model_crop = RandomForestClassifier(n_estimators=100, random_state=42)
model_crop.fit(X_train, y_train)

# Define route for home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        district = request.form['district']
        soil_color = request.form['soil_color']
        rainfall = float(request.form['rainfall'])
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        pH = float(request.form['pH'])
        temperature = float(request.form['temperature'])

        # Encode user input
        input_data = pd.DataFrame([[district, soil_color]])
        input_data_encoded = encoder.transform(input_data)

        # Make prediction
        predicted_crop = model_crop.predict(input_data_encoded)
        recommended_fertilizer = dataset[dataset['Crop'] == predicted_crop[0]]['Fertilizer'].values[0]

        return render_template('result.html', crop=predicted_crop[0], fertilizer=recommended_fertilizer)

    return render_template('index.html', districts=dataset['District_Name'].unique(), soil_colors=dataset['Soil_color'].unique())

if __name__ == '__main__':
    app.run(debug=True)
