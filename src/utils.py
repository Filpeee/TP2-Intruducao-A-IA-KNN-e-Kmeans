import pandas as pd

def normalizar_min_max(df, colunas_para_ignorar=['TARGET_5Yrs']):
    """
    Recebe um DataFrame e normaliza suas colunas numéricas para a escala de 0 a 1.
    Ignora colunas específicas (como o nosso alvo) para não alterar a resposta.
    """
    df_norm = df.copy()

    for coluna in df_norm.columns:
        if coluna not in colunas_para_ignorar:
            valor_min = df_norm[coluna].min()
            valor_max = df_norm[coluna].max()

            df_norm[coluna] = (df_norm[coluna] - valor_min) / (valor_max - valor_min)
    
    return df_norm