import streamlit as st
import pandas as pd
import numpy as np

from category_encoders import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

import pickle
import plotly.express as px

st.set_page_config(layout='wide')

st.write("""
# Previsão de status de empréstimo do cliente App

**Esse aplicativo prevê a probalidade de um cliente ter um empréstimo
aprovado com base em algumas informações. ** 

""")

st.sidebar.header('Configure as informações do cliente')

def user_input_features():
    
    Gender = st.sidebar.selectbox('Genêro',('Male','Female'))
    Married = st.sidebar.selectbox('Matrimônio',('No','Yes'))
    Dependents = st.sidebar.selectbox('Dependentes',('0','1','2','3+'))
    Education = st.sidebar.selectbox('Graduação',('Graduate','Not Graduate'))
    Self_Employed =  st.sidebar.selectbox('Autônomo',('No','Yes'))
    Property = st.sidebar.selectbox('Localização da propriedade', ('Urban','Semiurban','Rural'))
    Credit_History = st.sidebar.selectbox("Histórico de crédito não: 0, sim: 1", ('0', '1'))
    ApplicantIncome = st.sidebar.slider('Renda do aplicante anual', 150, 81000, 10000)
    CoapplicantIncome = st.sidebar.slider('Renda do co-aplicante anual', 0.0, 41670.0, 10000.0)
    LoanAmount = st.sidebar.slider('Valor do empréstimo em milhares', 9.0, 700.0, 100.0)
    Loan_Amount_Term = st.sidebar.slider('Duração do empréstimo em meses', 12.0, 480.0, 100.0)
    data = {'Gender': Gender,
                'Married': Married,
                'Dependents': Dependents,
                'Education': Education,
                'Self_Employed': Self_Employed,
                'Property_Area': Property,
                'Credit_History' : Credit_History, 
                'ApplicantIncome': ApplicantIncome,
                'CoapplicantIncome' : CoapplicantIncome,
                'LoanAmount': LoanAmount,
                'Loan_Amount_Term': Loan_Amount_Term}
    features = pd.DataFrame(data, index=[0])
    return features
df0 = user_input_features()

##################################################################################
st.write("Exibindo os dados de entrada", df0)

# Codificando e normalizando os dados de entrada
ordinal = OrdinalEncoder()
df0 = ordinal.fit_transform(df0)

stand = StandardScaler().fit(df0)
#df0 = stand.transform(df0)

##################################################################################

##################################################################################
# Carregando o dataset e separando as features e target
loan = pd.read_csv('emprestimo_app.csv')

if st.sidebar.checkbox("Mostrar todos os dados"):
    st.subheader("Exibindo todos os dados")
    st.write(loan)

X = loan.drop('Loan_Status', axis= 1)
Y = loan['Loan_Status']

# Codificando e normalizar o conjunto de dados
X = ordinal.fit_transform(X)

stand = StandardScaler().fit(X)
X = stand.transform(X)
##################################################################################

##################################################################################
# Carregando nosso modelo de machine learning LGBMClassifier
load_clf = pickle.load(open('mdl_best.pkl', 'rb'))

load_clf.fit(X, Y)

# Prevendendo o status do empréstimo 
prediction = load_clf.predict(df0)

st.write('Prevendo o status do empréstimo', prediction)

p = load_clf.predict_proba(X)[:, 1]

roc = roc_auc_score(Y , p)

st.write("Eficiência do modelo usando a roc_auc: {:.2f} %".format(100*roc))
################################################################################

################################################################################
st.header("Exibindo gráficos")
# Criando filtro para exibir gráfico da distribuição do empréstimo
st.sidebar.subheader('Selecione o valor máximo do empréstimo')
st.subheader("Usando o filtro 'Selecione o valor máximo do empréstimo', podemos controlar a distribuição dos preços")

loanAm_min = int(loan['LoanAmount'].min())
loanAm_max = int(loan['LoanAmount'].max())
loanAm_avg = int(loan['LoanAmount'].mean())

f_loan = st.sidebar.slider('LoanAmount', loanAm_min, loanAm_max, loanAm_avg)
df = loan.loc[loan['LoanAmount'] < f_loan]

# Distribuição do valor do empréstimo
fig = px.histogram(df, x = 'LoanAmount', nbins=50,  color_discrete_sequence=['green'])
st.plotly_chart(fig, use_container_width= True)

st.sidebar.subheader('Selecione o valor máximo da renda do aplicante')
st.subheader("Usando o filtro 'Selecione o valor máximo da renda do aplicante', podemos controlar a distribuição da renda do aplicante")
# Criando filtro para exibir gráfico da renda do aplicante
loanAp_min = int(loan['ApplicantIncome'].min())
loanAp_max = int(loan['ApplicantIncome'].max())
loanAp_avg = int(loan['ApplicantIncome'].mean())

f_loanAp = st.sidebar.slider('ApplicantIncome', loanAp_min, loanAp_max, loanAp_avg)
df = loan.loc[loan['ApplicantIncome'] < f_loanAp]

# Distribuição da renda do aplicante
fig = px.histogram(df, x = 'LoanAmount', nbins=50, color_discrete_sequence=['green'])
st.plotly_chart(fig, use_container_width= True)

st.sidebar.subheader('Selecione o valor máximo da renda do co-aplicante')
st.subheader("Usando o filtro 'Selecione o valor máximo da renda do co-aplicante', podemos controlar a distribuição da renda do co-aplicante")
# Criando filtro para exibir gráfico da renda do co-aplicante
loanCo_min = int(loan['CoapplicantIncome'].min())
loanCo_max = int(loan['CoapplicantIncome'].max())
loanCo_avg = int(loan['CoapplicantIncome'].mean())

f_loanAp = st.sidebar.slider('CoapplicantIncome', loanAp_min, loanAp_max, loanAp_avg)
df = loan.loc[loan['CoapplicantIncome'] < f_loanAp]

# Distribuição da renda do co-aplicante
fig = px.histogram(df, x = 'CoapplicantIncome', nbins=50,  color_discrete_sequence=['green'])
st.plotly_chart(fig, use_container_width= True)

st.sidebar.subheader('Variação da valor do empréstimo com a renda do aplicante')
st.subheader("Usando o filtro 'Selecione', podemos controlar a distribuição média do valor do empéstimo com a renda do aplicante")
f_amount = st.sidebar.slider('Selecione renda',  loanAp_min, loanAp_max, loanAp_avg)
df = loan.loc[loan['ApplicantIncome'] < f_amount]

df = df[['ApplicantIncome', 'LoanAmount']].groupby('ApplicantIncome').mean().reset_index()

fig = px.line(df, x = 'ApplicantIncome', y = 'LoanAmount')
st.plotly_chart(fig, use_container_width=True)
