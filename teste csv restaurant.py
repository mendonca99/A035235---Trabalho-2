# Verificar possíveis problemas no dataset
# Exibir informações gerais do dataset
import pandas as pd
import os

if not os.path.exists('restaurant.csv'):
    print("O ficheiro 'restaurant.csv' não foi encontrado!")
else:
    restaurant_df = pd.read_csv('restaurant.csv')

info = restaurant_df.info()

# Verificar se há valores nulos em alguma coluna
missing_values = restaurant_df.isnull().sum()

# Exibir resultados
info, missing_values
