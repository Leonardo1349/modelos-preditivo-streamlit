from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

import streamlit as st
import joblib
import sklearn
import numpy as np
import pandas as pd
import os


def criarModelo(modelo, base, alvo):
    x_train, x_test, y_train, y_test = train_test_split(base.drop(alvo, axis=1),
                                                        base[alvo],
                                                        test_size=0.3,
                                                        random_state=42)

    modeloRegressor = modelo
    modeloRegressor.fit(x_train, y_train)
    return modeloRegressor


def importarModelo(modeloEscolhido):
    url = 'streamlit/predicao_sem_identificacao_da_empresa/base/dataset_sem_coluna_empresa.xlsx'
    alvo = 'IMPOSTOS_ESTADUAIS'
    base = pd.read_excel(url, engine="openpyxl")
    
    atributosSelecionados = ['RECEITA_VENDAS_BENS_OU_SERVICOS', 'CUSTO_DOS_BENS_OU_SERVICOS_VENDIDOS', 'DESPESAS_RECEITAS_OPERACIONAIS', 
                             'RESULTADO_FINANCEIRO', 'RECEITAS', 'DISTRIBUICAO_DO_VALOR_ADICIONADO', 'IMPOSTOS_ESTADUAIS']

    base = base.loc[:, atributosSelecionados]

    return criarModelo(modeloEscolhido, base, alvo)


modeloET = ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='friedman_mse',
                               max_depth=None, max_features='auto', max_leaf_nodes=None,
                               max_samples=None, min_impurity_decrease=0.0,
                               min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
                               n_estimators=100, n_jobs=-1, oob_score=False,
                               random_state=2444, verbose=0, warm_start=False)


modeloGB = GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, 
                                     init=None, learning_rate=0.1, max_depth=3,
                                     max_features=None, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, 
                                     min_samples_leaf=1, min_samples_split=2,
                                     min_weight_fraction_leaf=0.0, n_estimators=100,
                                     n_iter_no_change=None, random_state=2444, subsample=1.0, tol=0.0001,
                                     validation_fraction=0.1, verbose=0, warm_start=False)

modeloRF = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='friedman_mse',
                                 max_depth=None, max_features='auto', max_leaf_nodes=None,
                                 max_samples=None, min_impurity_decrease=0.0,
                                 min_samples_leaf=1,
                                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                                 n_estimators=100, n_jobs=-1, oob_score=False,
                                 random_state=2444, verbose=0, warm_start=False)


modelo_options = ['Extra Trees', 'Gradient Boosting', 'Random Forest']

modelo_values = {'Extra Trees': modeloET, 'Gradient Boosting': modeloGB, 'Random Forest': modeloRF}


st.header('Modelo de Predição de Sonegadores de Impostos Estaduais (Sem a Identificação da Empresa)')
st.subheader('Preencha as informações solicitadas para obter a predição:')

# Definindo os campos de entrada
modelo_value = st.selectbox('Escolha o Modelo', options=modelo_options)
modeloSelecionado = modelo_values.get(modelo_value)

receita_com_vendas_value = st.number_input('Qual o valor das Receitas com Vendas de Bens ou Serviços ?')

custo_dos_bens_vendidos_value = st.number_input('Qual o valor do Custo dos Bens ou Serviços Vendidos ?')

despesas_receitas_operacionais_value = st.number_input('Qual o valor das Despesas e Receitas Operacionais ?')

resultado_financeiro_value = st.number_input('Qual o valor do Resultado Financeiro ?')

receitas_value = st.number_input('Qual o valor das Receitas ?')

distribuicao_do_valor_adicionado_value = st.number_input('Qual o valor da Distribuição do valor Adicionado ?')

dados = {
    'RECEITA_VENDAS_BENS_OU_SERVICOS': receita_com_vendas_value,
    'CUSTO_DOS_BENS_OU_SERVICOS_VENDIDOS': custo_dos_bens_vendidos_value,
    'DESPESAS_RECEITAS_OPERACIONAIS': despesas_receitas_operacionais_value,
    'RESULTADO_FINANCEIRO': resultado_financeiro_value,
    'RECEITAS': receitas_value,
    'DISTRIBUICAO_DO_VALOR_ADICIONADO': distribuicao_do_valor_adicionado_value,   
}

modelo = importarModelo(modeloSelecionado)
botao = st.button('Efetuar Predição')
if(botao):  
    dadosFormatados = pd.DataFrame([dados])
    resultado = modelo.predict(dadosFormatados)
    prob =  round(resultado[0] / 100, 2)
    st.write('Probabilidade de Sonegação: ', prob, '%')
  
