import pandas as pd
from otimizacao import modelo_otimizado

if __name__ == '__main__':
    df = pd.read_csv("./data/dataset.csv")
    modelo_otimizado(df, "conversion_status")