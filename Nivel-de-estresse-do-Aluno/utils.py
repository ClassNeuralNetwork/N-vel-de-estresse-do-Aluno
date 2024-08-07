import shap
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# Carregando o Dataset a partir de um arquivo local
# Lendo o arquivo local
df = pd.read_csv("./dataset/model/StressLevelDataset.csv", delimiter=';')

# Dicionário para mapear nomes de colunas em inglês para português
colunas_traduzidas = {
    'anxiety_level': 'nível_de_ansiedade',
    'self_esteem': 'autoestima',
    'mental_health_history': 'histórico_de_saúde_mental',
    'depression': 'depressão',
    'headache': 'dor_de_cabeça',
    'blood_pressure': 'pressão_sanguínea',
    'sleep_quality': 'qualidade_do_sono',
    'breathing_problem': 'problema_de_respiração',
    'noise_level': 'nível_de_ruído',
    'living_conditions': 'condições_de_vivência',
    'safety': 'segurança',
    'basic_needs': 'necessidades_básicas',
    'academic_performance': 'desempenho_acadêmico',
    'study_load': 'carga_de_estudo',
    'teacher_student_relationship': 'relação_professor_aluno',
    'future_career_concerns': 'preocupações_com_a_carreira_futura',
    'social_support': 'apoio_social',
    'peer_pressure': 'pressão_dos_colegas',
    'extracurricular_activities': 'atividades_extracurriculares',
    'bullying': 'bullying',
    'stress_level': 'nível_de_estresse'
}

# Renomeando as colunas do DataFrame
df.rename(columns=colunas_traduzidas, inplace=True)

# Verificando se as colunas estão corretas
print(df.columns)  # Verifica os nomes das colunas
print(df.head())  # Exibe as primeiras linhas do DataSet
print(df.info())  # Tipos de dados e valores nulos
print(df.describe())  # Descreve estatísticas básicas dos dados

# Removendo valores nulos
df = df.dropna()

# Codificação de variáveis categóricas
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

# Definição das features e target
target_column_name = 'nível_de_estresse'
if target_column_name not in df.columns:
    raise KeyError(f"A coluna '{target_column_name}' não foi encontrada no DataFrame.")

X = df.drop(target_column_name, axis=1).values
y = df[target_column_name].values

# Dividindo os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizando as features para terem média 0 e desvio padrão 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Criando e treinando o modelo de Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Previsões
y_pred = model.predict(X_test)

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

# Importância relativa de cada feature no modelo
feature_importances = model.feature_importances_
features = df.columns[df.columns != target_column_name]
indices = np.argsort(feature_importances)

plt.figure(figsize=(10, 6))
plt.title('Importância das Features')
plt.barh(range(len(indices)), feature_importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Importância Relativa')
plt.show()

# Selecionando as features mais importantes 
num_top_features = 5
top_features = [features[i] for i in indices[-num_top_features:]]

print(f'Top {num_top_features} features: {top_features}')

# Usando SHAP para explicar o modelo
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualizando a importância das features com SHAP
shap.summary_plot(shap_values, X_test, feature_names=features)

# Plote o gráfico de força para uma amostra do conjunto de teste
shap.force_plot(explainer.expected_value[1], shap_values[1], X_test[1], feature_names=features)
