from sklearn.model_selection import train_test_split

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
    url = 'streamlit/predicao_sem_historico_das_empresas/base/dataset_sem_coluna_empresa.xlsx'
    alvo = 'IMPOSTOS_ESTADUAIS'
    base = pd.read_excel(url, engine="openpyxl")
    
    dados = {
    'RECEITA_VENDAS_BENS_OU_SERVICOS': RECEITA_VENDAS_BENS_OU_SERVICOS,
    'CUSTO_DOS_BENS_OU_SERVICOS_VENDIDOS': CUSTO_DOS_BENS_OU_SERVICOS_VENDIDOS,
    'DESPESAS_RECEITAS_OPERACIONAIS': DESPESAS_RECEITAS_OPERACIONAIS,
    'RESULTADO_FINANCEIRO': RESULTADO_FINANCEIRO,
    'RECEITAS': RECEITAS,
    'DISTRIBUICAO_DO_VALOR_ADICIONADO': DISTRIBUICAO_DO_VALOR_ADICIONADO,   
}
    
    atributosSelecionados = ['RECEITA_VENDAS_BENS_OU_SERVICOS', 'CUSTO_DOS_BENS_OU_SERVICOS_VENDIDOS', 'DESPESAS_RECEITAS_OPERACIONAIS', 
                             'RESULTADO_FINANCEIRO', 'RECEITAS', 'DISTRIBUICAO_DO_VALOR_ADICIONADO', 'IMPOSTOS_ESTADUAIS']

    base = base.loc[:, atributosSelecionados]

    return criarModelo(modeloEscolhido, base, alvo)

st.header('Modelo de Predição de Empresas Sonegadoras de Impostos Estaduais - Sem Histórico das Empresas')
st.subheader('Preencha as informações solicitadas para obter a predição:')

receita_com_vendas_value = st.number_input('Qual o valor das Receitas com Vendas de Bens ou Serviços ?')
receita_com_vendas = receita_com_vendas_value.get(receita_com_vendas_value)

custo_dos_bens_vendidos_value = st.number_input('Qual o valor do Custo dos Bens ou Serviços Vendidos ?')
custo_dos_bens_vendidos = custo_dos_bens_vendidos_value.get(custo_dos_bens_vendidos_value)

despesas_receitas_operacionais_value = st.number_input('Qual o valor das Despesas e Receitas Operacionais ?')
despesas_receitas_operacionais = despesas_receitas_operacionais_value.get(despesas_receitas_operacionais_value)

resultado_financeiro_value = st.number_input('Qual o valor do Resultado Financeiro ?')
esultado_financeiro = resultado_financeiro_value.get(resultado_financeiro_value)

receitas_value = st.number_input('Qual o valor das Receitas ?')
receitas = receitas_value.get(receitas_value)

distribuicao_do_valor_adicionado_value = st.number_input('Qual o valor da Distribuição do valor Adicionado ?')
distribuicao_do_valor_adicionado = distribuicao_do_valor_adicionado_value.get(distribuicao_do_valor_adicionado_value)


dados = {
    'RECEITA_VENDAS_BENS_OU_SERVICOS': RECEITA_VENDAS_BENS_OU_SERVICOS,
    'CUSTO_DOS_BENS_OU_SERVICOS_VENDIDOS': CUSTO_DOS_BENS_OU_SERVICOS_VENDIDOS,
    'DESPESAS_RECEITAS_OPERACIONAIS': DESPESAS_RECEITAS_OPERACIONAIS,
    'RESULTADO_FINANCEIRO': RESULTADO_FINANCEIRO,
    'RECEITAS': RECEITAS,
    'DISTRIBUICAO_DO_VALOR_ADICIONADO': DISTRIBUICAO_DO_VALOR_ADICIONADO,   
}

modelo = importarModelo(modeloSelecionado)
botao = st.button('Efetuar Predição')
if(botao):  
    dadosFormatados = pd.DataFrame([dados])
    resultado = modelo.predict_proba(dadosFormatados)
    prob =  round(resultado[0][0] * 100, 3)
    st.write('Probabilidade de Sonegação: ', prob, ' %')
  
