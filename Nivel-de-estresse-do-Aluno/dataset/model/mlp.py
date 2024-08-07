import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.optimizers import Adam

# Carrega o dataset com o delimitador correto
df = pd.read_csv('StressLevelDataset.csv', delimiter=';')

# Verifica os nomes das colunas
print("Colunas no DataFrame:", df.columns)

# Separa features e target
if 'stress_level' not in df.columns:
    raise KeyError("A coluna 'stress_level' não está presente no DataFrame. Verifique o nome da coluna no seu arquivo CSV.")

x = df.drop('stress_level', axis=1)
y = df['stress_level'].values

# Divide os dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Normaliza os dados
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Define o modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=128, activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dropout(0.10),
    tf.keras.layers.Dense(units=1)
])
model.summary()

# Compila o modelo
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mse')

# Treina o modelo
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Realiza previsões
y_pred = model.predict(x_test)
y_pred = y_pred.reshape(y_pred.shape[0])

# Avalia o modelo
print('MSE:', mean_squared_error(y_test, y_pred))
print('R2:', r2_score(y_test, y_pred))

# Plota a curva de perda
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()

# Plota as previsões versus valores reais
plt.plot(y_test[0:10], '-o', color='black')
plt.plot(y_pred[0:10], ':>', color='red')
plt.axis('equal')
plt.legend(['True', "Predicted"])
plt.show()

# Realiza previsão em um novo ponto de dados
new_data_point = np.array([[7.4 , 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]])
new_data_point_scaled = scaler.transform(new_data_point)

predicted_quality = model.predict(new_data_point_scaled)
print('Predicted quality:', predicted_quality[0])


# Avaliação
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Imprimindo o cálculo das métricas de avaliação: acurácia, precisão, recall e F1 score
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()