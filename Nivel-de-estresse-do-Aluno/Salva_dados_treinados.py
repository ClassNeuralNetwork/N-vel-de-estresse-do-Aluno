import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Carregando o Dataset a partir de um arquivo local
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
print(df.head())  # Exibe as primeiras linhas do DataFrame
print(df.info())  # Tipos de dados e valores nulos

# Removendo valores nulos
df = df.dropna()

# Definindo a coluna alvo (target)
target_column_name = 'nível_de_estresse'

# Verificando se a coluna alvo existe no DataFrame
if target_column_name not in df.columns:
    raise KeyError(f"A coluna '{target_column_name}' não foi encontrada no DataFrame.")

# Separando as features e o target
input_data = df.drop(target_column_name, axis=1).values
output_data = df[target_column_name].values

# Dividindo os dados em conjunto de treino e teste
input_train, input_test, output_train, output_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

# Criando os diretórios se não existirem
os.makedirs('./train', exist_ok=True)
os.makedirs('./test', exist_ok=True)


# Gerando os arquivos CSV
input_train_df = pd.DataFrame(input_train)
input_train_df.to_csv('./train/input_train.csv', index=False)

input_test_df = pd.DataFrame(input_test)
input_test_df.to_csv('./test/input_test.csv', index=False)

output_train_df = pd.DataFrame(output_train, columns=[target_column_name])
output_train_df.to_csv('./train/output_train.csv', index=False)

output_test_df = pd.DataFrame(output_test, columns=[target_column_name])
output_test_df.to_csv('./test/output_test.csv', index=False)
