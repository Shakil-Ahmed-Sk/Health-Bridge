from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

app = Flask(__name__, template_folder='templates',static_url_path='/static/')

# Loading the diabetes dataset to a pandas dataframe
diabetes_dataset = pd.read_csv("diabetics.csv")

# Separating features and target variable
X = diabetes_dataset.drop(columns='diabetes', axis=1)
Y = diabetes_dataset['diabetes']

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
Y = diabetes_dataset['diabetes']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

classifier = SVC(kernel='linear')
classifier.fit(X_train, Y_train)

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data:', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data:', test_data_accuracy)

@app.route('/')
def index():
    return render_template('index.html')  
 
@app.route('/input')
def input_page():
    return render_template('input.html') 

@app.route('/submit-data', methods=['POST'])
def submit_data():
    if request.method == 'POST':
        # Get form field values
        age = float(request.form['age'])
        gender = request.form['gender']
        if gender == 'male':
            gender_encoded = 0
        elif gender == 'female':
            gender_encoded = 1
        else:
            gender_encoded = 2  # Encode 'other' as another value

        # Checkbox handling
        hypertension = 1 if 'hypertension' in request.form else 0
        heart_disease = 1 if 'heartDisease' in request.form else 0
        smoking = 1 if 'smoking' in request.form else 0

        bmi = float(request.form['bmi'])
        hba1c = float(request.form['hba1c'])
        blood_glucose = float(request.form['bloodGlucose'])

        # Combine all features into an array
        input_data = np.array([age, gender_encoded, hypertension, heart_disease, smoking, bmi, hba1c, blood_glucose])

        # Reshape and scale the input data
        input_data_reshaped = input_data.reshape(1, -1)
        std_data = scaler.transform(input_data_reshaped)

        # Predict using the loaded model
        prediction = classifier.predict(std_data)

        result = 'The person is diabetic.' if prediction[0] == 1 else 'The person is not diabetic.'

        return render_template('result.html', prediction_result=result)
    else:
        return "Invalid request method"

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
