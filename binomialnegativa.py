# -*- coding: utf-8 -*-
import streamlit as st
import math
from scipy.stats import nbinom

# --- Configuração de Estilo ---
st.set_page_config(page_title="Scanner Pro v7.2", page_icon="⚽", layout="wide")

# CSS para deixar o visual "Dark" e organizado
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; color: #00ffcc; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #0066cc; color: white; }
    .status-box { padding: 20px; border-radius: 10px; margin-top: 10px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- Funções de Cálculo (Mantidas conforme seu modelo original) ---
def get_temporal_factor(minuto, f_inicio, f_fim1, f_ini2, f_fim2):
    if minuto <= 35: return f_inicio
    elif 36 <= minuto <= 45: return f_fim1
    elif 46 <= minuto <= 80: return f_ini2
    else: return f_fim2

def calcular_lambda_restante(minutos_jogados, lambda_partida, f1, f2, f3, f4):
    tempo_total = 95
    taxa_base = lambda_partida / tempo_total
    lambda_restante = 0.0
    for minuto in range(minutos_jogados + 1, tempo_total + 1):
        fator = get_temporal_factor(minuto, f1, f2, f3, f4)
        lambda_restante += taxa_base * fator
    return lambda_restante

def neg_binomial_prob(k_count, mu, dispersion_param):
    if k_count < 0: return 0.0
    n = dispersion_param
    p = n / (n + mu)
    return float(nbinom.pmf(k_count, n, p))

# --- Interface - Barra Lateral (AQUI ESTÃO SEUS CONTROLES) ---
with st.sidebar:
    st.header("🎮 Parâmetros da Partida")
    minutos = st.slider("Minutos Jogados", 0, 95, 45)
    atuais = st.number_input("Escanteios Atuais", 0, 30, 5)
    
    st.divider()
    st.subheader("🎯 Mercado")
    linha = st.number_input("Linha de Aposta (Over)", 0.5, 20.0, 10.5, 0.5)
    odd_o = st.number_input("Odd Over", 1.01, 10.0, 1.90)
    odd_u = st.number_input("Odd Under", 1.01, 10.0, 1.90)
    
    st.divider()
    st.subheader("🔥 Ritmo (K - Dispersão)")
    ritmo = st.selectbox("Comportamento", ["Cadenciado (K: 2.0)", "Padrão (K: 1.5)", "Pressão Total (K: 1.1)"], index=1)
    k_map = {"Cadenciado (K: 2.0)": 2.0, "Padrão (K: 1.5)": 1.5, "Pressão Total (K: 1.1)": 1.1}
    k_val = k_map[ritmo]

    with st.expander("⚙️ Ajuste Fatores Temporais"):
        f1 = st.slider("0-35 min", 0.5, 1.5, 0.90)
        f2 = st.slider("36-45 min", 0.5, 1.5, 1.10)
        f3 = st.slider("46-80 min", 0.5, 1.5, 0.95)
        f4 = st.slider("81-95 min", 0.5, 1.5, 1.20)

# --- Corpo Principal ---
st.title("⚽ Analisador de Escanteios v7.2")

c1, c2, c3, c4 = st.columns(4)
mcf = c1.number_input("Casa Favor", 0.0, 15.0, 5.5)
mcc = c2.number_input("Casa Contra", 0.0, 15.0, 3.0)
mvf = c3.number_input("Visitante Favor", 0.0, 15.0, 4.5)
mvc = c4.number_input("Visitante Contra", 0.0, 15.0, 4.0)

media_ponderada = ((mcf + mvf) + (mcc + mvc)) / 2
st.info(f"Média Esperada do Confronto: **{media_ponderada:.2f} cantos**")

if st.button("CALCULAR VALOR"):
    # Execução do Modelo
    lambda_r = calcular_lambda_restante(minutos, media_ponderada, f1, f2, f3, f4)
    target_over = math.floor(linha) + 1 - atuais
    
    # Probabilidades
    is_half = (linha * 10) % 10 != 0
    if is_half:
        p_under = sum(neg_binomial_prob(k, lambda_r, k_val) for k in range(max(0, target_over)))
        p_over = 1.0 - p_under
        p_push = 0.0
    else:
        p_push = neg_binomial_prob(target_over, lambda_r, k_val)
        p_under = sum(neg_binomial_prob(k, lambda_r, k_val) for k in range(max(0, target_over)))
        p_over = 1.0 - p_push - p_under

    ev_o = (p_over * (odd_o - 1)) - (p_under * 1)
    odd_justa = 1 / p_over if p_over > 0 else 0

    # --- EXIBIÇÃO DE DADOS AMPLIADA ---
    st.subheader("📊 Análise de Probabilidades")
    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
    
    col_res1.metric("Prob. Over", f"{p_over:.1%}")
    col_res2.metric("Odd Justa", f"{odd_justa:.2f}")
    col_res3.metric("EV (Valor)", f"{ev_o:.3f}", delta=f"{ev_o*100:.1f}%")
    col_res4.metric("Cantos Previstos", f"{lambda_r:.2f}")

    # Detalhes do Modelo para conferência
    with st.expander("🔍 Detalhes do Cálculo e Parâmetros Utilizados"):
        d1, d2, d3 = st.columns(3)
        d1.write(f"**Ritmo (K):** {k_val}")
        d1.write(f"**Minutos Restantes:** {95 - minutos}")
        d2.write(f"**Prob. Push:** {p_push:.1%}")
        d2.write(f"**Prob. Under:** {p_under:.1%}")
        d3.write(f"**Fator Atual:** {get_temporal_factor(minutos, f1, f2, f3, f4):.2f}x")
        d3.write(f"**Linha Alvo:** {linha}")

    # Veredito
    st.divider()
    if ev_o > 0.05:
        st.success(f"### ✅ ENTRADA SUGERIDA: OVER {linha}")
        st.balloons()
    elif ev_o < -0.05 and (1/p_under < odd_u):
        st.info(f"### 📉 POSSÍVEL VALOR NO UNDER {linha}")
    else:
        st.warning("### ⚠️ SEM MARGEM DE SEGURANÇA (EV BAIXO)")