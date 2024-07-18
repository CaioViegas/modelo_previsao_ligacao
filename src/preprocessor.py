import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from category_encoders import TargetEncoder

def preprocessamento(dataset):
    le = LabelEncoder()
    dataset['conversion_status'] = le.fit_transform(dataset['conversion_status'])
    dataset['conversion_status'] = dataset['conversion_status'].astype('int64')

    oe = OrdinalEncoder()
    colunas_ordinal = ['call_month', 'education_level', 'previous_campaign_outcome']
    for coluna in colunas_ordinal:
        dataset[coluna] = oe.fit_transform(dataset[[coluna]])

    colunas_ohe = ['marital_status', 'communication_channel']
    ohe = OneHotEncoder(sparse_output=False)
    ohe.fit(dataset[colunas_ohe])
    encoded_data = ohe.transform(dataset[colunas_ohe])
    novas_colunas = ohe.get_feature_names_out(colunas_ohe)
    for i, coluna in enumerate(novas_colunas):
        dataset[coluna] = encoded_data[:, i]

    for coluna in colunas_ohe:
        dataset = dataset.drop(coluna, axis=1)

    te = TargetEncoder(cols=['occupation'], smoothing=0.2)
    dataset['occupation'] = te.fit_transform(dataset['occupation'], dataset['conversion_status'])

    return dataset

if __name__ == '__main__':
    df = pd.read_csv("data/dataset.csv")
    df = preprocessamento(df)
    print(df.columns)
    for coluna in df.columns:
        print(f"{coluna}: {df[coluna].value_counts()}")