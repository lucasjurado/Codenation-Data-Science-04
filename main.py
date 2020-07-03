#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Desafio-3" data-toc-modified-id="Desafio-3-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Desafio 3</a></span><ul class="toc-item"><li><span><a href="#Setup-geral" data-toc-modified-id="Setup-geral-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span><em>Setup</em> geral</a></span></li><li><span><a href="#Parte-1" data-toc-modified-id="Parte-1-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Parte 1</a></span><ul class="toc-item"><li><span><a href="#Setup-da-parte-1" data-toc-modified-id="Setup-da-parte-1-1.2.1"><span class="toc-item-num">1.2.1&nbsp;&nbsp;</span><em>Setup</em> da parte 1</a></span></li></ul></li><li><span><a href="#Inicie-sua-análise-a-partir-da-parte-1-a-partir-daqui" data-toc-modified-id="Inicie-sua-análise-a-partir-da-parte-1-a-partir-daqui-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Inicie sua análise a partir da parte 1 a partir daqui</a></span><ul class="toc-item"><li><span><a href="#Normal" data-toc-modified-id="Normal-1.3.1"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>Normal</a></span></li><li><span><a href="#Binomial" data-toc-modified-id="Binomial-1.3.2"><span class="toc-item-num">1.3.2&nbsp;&nbsp;</span>Binomial</a></span></li><li><span><a href="#Diferença-entre-os-quartis" data-toc-modified-id="Diferença-entre-os-quartis-1.3.3"><span class="toc-item-num">1.3.3&nbsp;&nbsp;</span>Diferença entre os quartis</a></span></li></ul></li><li><span><a href="#Questão-1" data-toc-modified-id="Questão-1-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Questão 1</a></span></li><li><span><a href="#Questão-2" data-toc-modified-id="Questão-2-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Questão 2</a></span></li><li><span><a href="#Questão-3" data-toc-modified-id="Questão-3-1.6"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>Questão 3</a></span></li><li><span><a href="#Parte-2" data-toc-modified-id="Parte-2-1.7"><span class="toc-item-num">1.7&nbsp;&nbsp;</span>Parte 2</a></span><ul class="toc-item"><li><span><a href="#Setup-da-parte-2" data-toc-modified-id="Setup-da-parte-2-1.7.1"><span class="toc-item-num">1.7.1&nbsp;&nbsp;</span><em>Setup</em> da parte 2</a></span></li></ul></li><li><span><a href="#Inicie-sua-análise-da-parte-2-a-partir-daqui" data-toc-modified-id="Inicie-sua-análise-da-parte-2-a-partir-daqui-1.8"><span class="toc-item-num">1.8&nbsp;&nbsp;</span>Inicie sua análise da parte 2 a partir daqui</a></span></li><li><span><a href="#Questão-4" data-toc-modified-id="Questão-4-1.9"><span class="toc-item-num">1.9&nbsp;&nbsp;</span>Questão 4</a></span></li><li><span><a href="#Questão-5" data-toc-modified-id="Questão-5-1.10"><span class="toc-item-num">1.10&nbsp;&nbsp;</span>Questão 5</a></span></li></ul></li></ul></div>

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[2]:


#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

#sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[3]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[4]:


dataframe.head(5)


# ### Normal

# In[96]:


normal = dataframe.normal
normal


# In[97]:


normal.describe()


# In[98]:


sct.norm.ppf(0.25, loc=20, scale=4)


# In[99]:


normal_vinte_cinco = normal.describe()[4]
normal_vinte_cinco


# In[100]:


normal_cinquenta = normal.describe()[5]
normal_cinquenta


# In[101]:


normal_setenta_cinco = normal.describe()[6]
normal_setenta_cinco


# ### Binomial

# In[102]:


binomial = dataframe.binomial
binomial.head(5)


# In[103]:


binomial.describe()


# In[104]:


binomial_vinte_cinco = binomial.describe()[4]
binomial_vinte_cinco


# In[105]:


binomial_cinquenta = binomial.describe()[5]
binomial_cinquenta


# In[106]:


binomial_setenta_cinco = binomial.describe()[6]
binomial_setenta_cinco


# ### Diferença entre os quartis

# In[107]:


dif_q1 = (normal_vinte_cinco - binomial_vinte_cinco).round(3)
dif_q2 = (normal_cinquenta - binomial_cinquenta).round(3)
dif_q3 = (normal_setenta_cinco - binomial_setenta_cinco).round(3)


# In[108]:


dif_quartis = (dif_q1, dif_q2, dif_q3)
dif_quartis


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[27]:


def q1():
    return dif_quartis


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[86]:


media = normal.mean()


# In[87]:


desvio_padrao = normal.std()


# In[88]:


probabilidade = ECDF(normal)
dif_intervalo = (probabilidade(media + desvio_padrao) - probabilidade(media - desvio_padrao)).round(3)
dif_intervalo


# In[89]:


def q2():
    return dif_intervalo


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[113]:


m_norm = normal.mean()


# In[114]:


v_norm = normal.var()


# In[115]:


m_binom = binomial.mean()


# In[116]:


v_binom = binomial.var()


# In[117]:


dif_m = (m_binom - m_norm).round(3)


# In[118]:


dif_v = (v_binom - v_norm).round(3)


# In[120]:


dif_norm_binom = (dif_m, dif_v) 
dif_norm_binom


# In[121]:


def q3():
    return dif_norm_binom


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[168]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[169]:


stars.head(10)


# In[170]:


stars.shape


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[173]:


df = stars.query('target == False').mean_profile
df


# In[176]:


df_mean = df.mean()
df_mean


# In[178]:


df_std = df.std()
df_std


# In[181]:


false_pulsar_mean_profile_standardized = (df - df_mean) / df_std
false_pulsar_mean_profile_standardized


# In[188]:


cdf_emp = ECDF(false_pulsar_mean_profile_standardized)
quantis_dist_norm = sct.norm.ppf([0.80, 0.90, 0.95])
probabilidade_quantis = tuple((cdf_emp(quantis_dist_norm)).round(3))
probabilidade_quantis


# In[189]:


def q4():
    return probabilidade_quantis


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[192]:


normal_q1 = sct.norm.ppf(0.25, loc=0, scale=1)
normal_q1


# In[193]:


normal_q2 = sct.norm.ppf(0.5, loc=0, scale=1)
normal_q2


# In[194]:


normal_q3 = sct.norm.ppf(0.75, loc=0, scale=1)
normal_q3


# In[195]:


false_pulsar_q1, false_pulsar_q2, false_pulsar_q3 = false_pulsar_mean_profile_standardized.quantile([0.25, 0.5, 0.75])


# In[198]:


dif_quantile = ((false_pulsar_q1 - normal_q1).round(3), (false_pulsar_q2 - normal_q2).round(3), (false_pulsar_q3 - normal_q3).round(3))
dif_quantile


# In[199]:


def q5():
    return dif_quantile


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
