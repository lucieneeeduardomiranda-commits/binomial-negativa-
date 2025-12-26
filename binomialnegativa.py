# -*- coding: utf-8 -*-
import streamlit as st
import math
from scipy.stats import nbinom

# --- Configuração de Estilo ---
st.set_page_config(page_title="Scanner Pro v7.2", page_icon="⚽", layout="wide")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; color: #00ffcc; font-weight: bold; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #0066cc; color: white; font-weight: bold; }
    h3 { padding-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- Funções de Cálculo ---
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

def calcular_odd_minima_para_ev(probabilidade_ganho, probabilidade_perda, ev_alvo=0.05):
    """
    Calcula a Odd Mínima necessária para atingir um EV específico.
    Fórmula: Odd = (EV_alvo + P_perda) / P_ganho + 1
    """
    if probabilidade_ganho <= 0:
        return None
    
    # Odd = (EV_alvo + P_perda) / P_ganho + 1
    odd_minima = ((ev_alvo + probabilidade_perda) / probabilidade_ganho) + 1
    return odd_minima

# --- Interface - Barra Lateral ---
with st.sidebar:
    st.header("🎮 Parâmetros")
    minutos = st.slider("Minutos Jogados", 0, 95, 45)
    atuais = st.number_input("Escanteios Atuais", 0, 30, 5)
    
    st.divider()
    st.subheader("🎯 Mercado")
    linha = st.number_input("Linha de Aposta (Over)", 0.5, 20.0, 10.5, 0.5)
    odd_o = st.number_input("Odd Over", 1.01, 10.0, 1.90)
    odd_u = st.number_input("Odd Under", 1.01, 10.0, 1.90)
    
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

if st.button("CALCULAR ANÁLISE"):
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

    # Cálculos de EV (Valor Esperado)
    # P_perda_over = 1 - P_over - P_push
    # P_perda_under = 1 - P_under - P_push
    p_perda_over = 1.0 - p_over - p_push
    p_perda_under = 1.0 - p_under - p_push
    
    ev_o = (p_over * (odd_o - 1)) - (p_perda_over * 1)
    ev_u = (p_under * (odd_u - 1)) - (p_perda_under * 1)

    # --- EXIBIÇÃO DE RESULTADOS ---
    st.divider()
    st.subheader("📊 Probabilidades e Valor")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("Probabilidade Over", f"{p_over:.1%}")
    col_p2.metric("Probabilidade Under", f"{p_under:.1%}")
    col_p3.metric("Probabilidade Push", f"{p_push:.1%}" if not is_half else "N/A")

    col_ev1, col_ev2 = st.columns(2)
    col_ev1.metric("EV Over", f"{ev_o:.3f}")
    col_ev2.metric("EV Under", f"{ev_u:.3f}")

    # --- VEREDITO FINAL ---
    st.divider()
    if ev_o > 0.05:
        st.success("## ✅ VALOR NO OVER")
    elif ev_u > 0.05:
        st.info("## 📉 VALOR NO UNDER")
    else:
        st.error("## 🛑 EVITAR A ENTRADA")

    # --- NOVO: ODD MÍNIMA PARA EV > 0.05 ---
    st.divider()
    st.subheader("💰 Odd Mínima Necessária (para EV > 0.05)")
    
    # Calcular a Odd Mínima para o Over
    odd_min_over = calcular_odd_minima_para_ev(p_over, p_perda_over, ev_alvo=0.05)
    
    # Calcular a Odd Mínima para o Under
    odd_min_under = calcular_odd_minima_para_ev(p_under, p_perda_under, ev_alvo=0.05)
    
    col_odd1, col_odd2 = st.columns(2)
    
    with col_odd1:
        if odd_min_over is not None and odd_min_over > 0:
            st.metric(
                f"Odd Mínima Over {linha}",
                f"{odd_min_over:.2f}",
                f"Odd Atual: {odd_o:.2f}"
            )
            if odd_o >= odd_min_over:
                st.success(f"✅ Odd atual ({odd_o:.2f}) está acima da mínima ({odd_min_over:.2f})")
            else:
                st.warning(f"⚠️ Odd atual ({odd_o:.2f}) está abaixo da mínima ({odd_min_over:.2f})")
        else:
            st.metric(f"Odd Mínima Over {linha}", "N/A", "Probabilidade insuficiente")
    
    with col_odd2:
        if odd_min_under is not None and odd_min_under > 0:
            st.metric(
                f"Odd Mínima Under {linha}",
                f"{odd_min_under:.2f}",
                f"Odd Atual: {odd_u:.2f}"
            )
            if odd_u >= odd_min_under:
                st.success(f"✅ Odd atual ({odd_u:.2f}) está acima da mínima ({odd_min_under:.2f})")
            else:
                st.warning(f"⚠️ Odd atual ({odd_u:.2f}) está abaixo da mínima ({odd_min_under:.2f})")
        else:
            st.metric(f"Odd Mínima Under {linha}", "N/A", "Probabilidade insuficiente")

    # --- DETALHES TÉCNICOS ---
    with st.expander("🔍 Detalhes Técnicos"):
        st.write(f"**Projeção de Cantos Restantes:** {lambda_r:.2f}")
        st.write(f"**Fator Temporal Atual:** {get_temporal_factor(minutos, f1, f2, f3, f4):.2f}x")
        st.write(f"**Parâmetro de Dispersão (k):** {k_val}")
        st.write(f"**Tipo de Linha:** {'Meio Ponto' if is_half else 'Asiática'}")
        
        st.markdown("**Fórmula da Odd Mínima:**")
        st.latex(r"Odd_{mínima} = \frac{EV_{alvo} + P_{perda}}{P_{ganho}} + 1")
        st.write(f"Onde EV_alvo = 0.05 (5%)")
