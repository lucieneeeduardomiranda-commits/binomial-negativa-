# -*- coding: utf-8 -*-
"""
Analisador de Apostas em Escanteios - v7.1 (Negative Binomial Model)
Interface Web (Streamlit)
Modelo: Média Ponderada Original + K Variável (Ritmo de Jogo)
"""
import streamlit as st
import math
from scipy.stats import nbinom

# --- Configuração da Página ---
st.set_page_config(
    page_title="Analisador de Escanteios v7.1",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Funções de Cálculo ---

def get_temporal_factor(minuto, fator_inicio, fator_fim_1t, fator_inicio_2t, fator_fim_jogo):
    if minuto <= 35:
        return fator_inicio
    elif 36 <= minuto <= 45:
        return fator_fim_1t
    elif 46 <= minuto <= 80:
        return fator_inicio_2t
    else:
        return fator_fim_jogo

def calcular_lambda_restante_time_dependent(minutos_jogados, lambda_partida, fator_inicio, fator_fim_1t, fator_inicio_2t, fator_fim_jogo):
    TEMPO_TOTAL = 95
    taxa_base_por_minuto = lambda_partida / TEMPO_TOTAL
    lambda_restante = 0.0
    for minuto in range(minutos_jogados + 1, TEMPO_TOTAL + 1):
        fator = get_temporal_factor(minuto, fator_inicio, fator_fim_1t, fator_inicio_2t, fator_fim_jogo)
        lambda_restante += taxa_base_por_minuto * fator
    return lambda_restante

def neg_binomial_prob(k_count, mu, dispersion_param):
    if k_count < 0:
        return 0.0
    try:
        # p = n / (n + mu)
        n = dispersion_param
        p = n / (n + mu)
        prob = nbinom.pmf(k_count, n, p)
        return float(prob)
    except (OverflowError, ValueError, ZeroDivisionError):
        return 0.0

def calcular_ev_e_prob(minutos_jogados, escanteios_atuais, linha_over, odd_over, odd_under, media_escanteios_por_jogo, fator_inicio, fator_fim_1t, fator_inicio_2t, fator_fim_jogo, dispersion_param):
    TEMPO_TOTAL_REGULAMENTAR = 95
    
    # 1. Cálculo do Lambda Restante
    minutos_restantes = max(0, TEMPO_TOTAL_REGULAMENTAR - minutos_jogados)
    
    if minutos_restantes == 0:
        escanteios_projetados_restantes = 0.0
    else:
        escanteios_projetados_restantes = calcular_lambda_restante_time_dependent(
            minutos_jogados, 
            media_escanteios_por_jogo,
            fator_inicio,
            fator_fim_1t,
            fator_inicio_2t,
            fator_fim_jogo
        )

    # 2. Determinação do Tipo de Linha
    is_half_line = (linha_over * 10) % 10 != 0
    target_over = math.floor(linha_over) + 1 - escanteios_atuais
    
    # 3. Cálculo das Probabilidades
    if is_half_line:
        if target_over <= 0:
            prob_over, prob_under = 1.0, 0.0
        elif minutos_restantes == 0:
            prob_over, prob_under = 0.0, 1.0
        else:
            prob_under_target = sum(neg_binomial_prob(k, escanteios_projetados_restantes, dispersion_param) for k in range(target_over))
            prob_under = prob_under_target
            prob_over = 1.0 - prob_under
        prob_push = 0.0
    else:
        target_push = target_over
        if minutos_restantes == 0:
            prob_over, prob_push, prob_under = 0.0, 0.0, 1.0
        else:
            prob_push = neg_binomial_prob(target_push, escanteios_projetados_restantes, dispersion_param)
            prob_under = sum(neg_binomial_prob(k, escanteios_projetados_restantes, dispersion_param) for k in range(target_push))
            prob_over = 1.0 - prob_push - prob_under
            
            if escanteios_atuais > linha_over:
                prob_over, prob_push, prob_under = 1.0, 0.0, 0.0
            elif escanteios_atuais == linha_over:
                prob_over, prob_push, prob_under = 0.0, 1.0, 0.0

    # 4. Cálculo do EV
    ev_over = (prob_over * (odd_over - 1)) - (prob_under * 1)
    ev_under = (prob_under * (odd_under - 1)) - (prob_over * 1)

    return {
        "minutos_restantes": minutos_restantes,
        "escanteios_projetados_restantes": escanteios_projetados_restantes,
        "is_half_line": is_half_line,
        "prob_over": prob_over,
        "prob_push": prob_push,
        "prob_under": prob_under,
        "ev_over": ev_over,
        "ev_under": ev_under,
        "media_escanteios": media_escanteios_por_jogo,
        "fator_atual": get_temporal_factor(minutos_jogados, fator_inicio, fator_fim_1t, fator_inicio_2t, fator_fim_jogo),
        "k_utilizado": dispersion_param
    }

# --- Interface ---
st.title("⚽ Analisador de Escanteios v7.1")
st.markdown("**Modelo: Binomial Negativa com Volatilidade Ajustável**")

with st.sidebar:
    st.header("📊 Configurações")
    minutos_jogados = st.slider("Minutos Jogados", 0, 95, 45)
    escanteios_atuais = st.number_input("Escanteios Atuais", 0, 30, 5)
    
    st.markdown("---")
    linha_over = st.number_input("Linha de Aposta", 0.5, 20.0, 10.5, 0.5)
    col_o, col_u = st.columns(2)
    odd_over = col_o.number_input("Odd Over", 1.01, 10.0, 1.90)
    odd_under = col_u.number_input("Odd Under", 1.01, 10.0, 1.90)
    
    st.markdown("---")
    st.subheader("🔥 Dinâmica da Partida (K)")
    ritmo = st.selectbox(
        "Ritmo do Jogo",
        ["Cadenciado (Mais Previsível)", "Padrão (Equilibrado)", "Pressão Total (Mais Caótico)"],
        index=1,
        help="Ajusta o modelo para a frequência de 'clusters' de escanteios."
    )
    k_map = {"Cadenciado (Mais Previsível)": 2.0, "Padrão (Equilibrado)": 1.5, "Pressão Total (Mais Caótico)": 1.1}
    dispersion_param = k_map[ritmo]

    st.markdown("---")
    st.subheader("⚙️ Fatores Temporais")
    f_inicio = st.slider("Início (0-35)", 0.5, 1.5, 0.90)
    f_fim1 = st.slider("Fim 1T (36-45)", 0.5, 1.5, 1.10)
    f_ini2 = st.slider("Início 2T (46-80)", 0.5, 1.5, 0.95)
    f_fim2 = st.slider("Fim Jogo (81-95)", 0.5, 1.5, 1.20)

st.subheader("📋 Dados dos Times")
c1, c2, c3, c4 = st.columns(4)
mcf = c1.number_input("Casa Favor", 0.0, 15.0, 5.89)
mcc = c2.number_input("Casa Contra", 0.0, 15.0, 2.95)
mvf = c3.number_input("Visitante Favor", 0.0, 15.0, 4.00)
mvc = c4.number_input("Visitante Contra", 0.0, 15.0, 4.68)

# Sua média ponderada original
media_ponderada_jogo = ((mcf + mvf) + (mcc + mvc)) / 2
st.info(f"Média Ponderada Calculada: **{media_ponderada_jogo:.2f}**")

if st.button("🔍 Calcular Análise", use_container_width=True, type="primary"):
    res = calcular_ev_e_prob(minutos_jogados, escanteios_atuais, linha_over, odd_over, odd_under, media_ponderada_jogo, f_inicio, f_fim1, f_ini2, f_fim2, dispersion_param)
    
    # Exibição simplificada para o usuário
    col1, col2, col3 = st.columns(3)
    col1.metric("Prob. Over", f"{res['prob_over']*100:.1f}%")
    col2.metric("Prob. Under", f"{res['prob_under']*100:.1f}%")
    col3.metric("EV Over", f"{res['ev_over']:.3f}")
    
    if res['ev_over'] > 0.05:
        st.success(f"🎯 VALOR NO OVER {linha_over}")
    elif res['ev_under'] > 0.05:
        st.success(f"🎯 VALOR NO UNDER {linha_over}")
    else:
        st.warning("⚠️ SEM VALOR NO MOMENTO")