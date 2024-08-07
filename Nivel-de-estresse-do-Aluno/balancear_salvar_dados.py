import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Carregando os Datasets de treino (input e output)
input_train = pd.read_csv('./dataset/train/input_train.csv', delimiter=';')
output_train = pd.read_csv('./dataset/train/output_train.csv', delimiter=';')
names = pd.read_csv('./dataset/model/StressLevelDataset.csv')

# Removendo valores nulos
input_train = input_train.dropna()
output_train = output_train.dropna()

# Codificação de variáveis categóricas no input_train
label_encoder = LabelEncoder()
for column in input_train.select_dtypes(include=['object']).columns:
    input_train[column] = label_encoder.fit_transform(input_train[column])

# Balanceamento de dados usando SMOTE
sm = SMOTE(random_state=42)
input_train_balanced, output_train_balanced = sm.fit_resample(input_train, output_train)

# Criar o diretório se ele não existir
balanced_dir = './dataset/train/train_balanceados'
os.makedirs(balanced_dir, exist_ok=True)

# Salvar os dados balanceados em novos arquivos CSV
input_train_balanced_df = pd.DataFrame(input_train_balanced, columns=input_train.columns)
input_train_balanced_df.to_csv(os.path.join(balanced_dir, 'input_train_balanced.csv'), index=False)

output_train_balanced_df = pd.DataFrame(output_train_balanced, columns=output_train.columns)
output_train_balanced_df.to_csv(os.path.join(balanced_dir, 'output_train_balanced.csv'), index=False)

# Exibindo a contagem das classes balanceadas
output_train_balanced_array = output_train_balanced_df.values.ravel()  # Converter DataFrame para array unidimensional
print("Contagem das classes balanceadas no output_train_balanced:")
print(pd.Series(output_train_balanced_array).value_counts())

# Pegando os nomes das colunas e seus índices
feature_names = input_train.columns.tolist()
print("Nomes das características e seus índices:")
for index, name in enumerate(feature_names):
    print(f"{index}: {name}")
