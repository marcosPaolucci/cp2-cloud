import os
from flask import Flask, redirect, render_template, request, send_from_directory, url_for
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Carregando o dataset Iris e treinando o modelo KNN
iris = load_iris()
X = iris.data  # features
y = iris.target  # labels (0, 1, 2)

# Dividindo em conjunto de treino e teste (aqui usamos todo o conjunto para simplicidade)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

app = Flask(__name__)

@app.route('/')
def index():
    print('Request for index page received')
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recebendo os dados do formulário
        sepal_length = float(request.form.get('sepal_length'))
        sepal_width = float(request.form.get('sepal_width'))
        petal_length = float(request.form.get('petal_length'))
        petal_width = float(request.form.get('petal_width'))

        # Organizando os dados na forma correta para o modelo
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Realizando a predição
        prediction = knn.predict(input_data)[0]

        # Convertendo a predição para o nome da classe
        class_name = iris.target_names[prediction]
        
        print(f'Request for prediction: {class_name}')
        return render_template('result.html', prediction=class_name)
    except Exception as e:
        print(f'Error in prediction: {str(e)}')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run()
