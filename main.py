import streamlit as st
import pandas as pd
import data_cleaning as dc
import logic as lo
import graphics as gr
import os

st.set_page_config(page_title="Machine Learning", layout="wide")

@st.cache_data
def load_data():
    file = os.path.join("data", "instagram_users_lifestyle.parquet")
    
    if not os.path.exists(file):
        st.error(f"Arquivo não encontrado: {file}")
        return None
        
    df = pd.read_parquet(file, engine='pyarrow')

    if len(df) > 100000:
        df = df.sample(n=100000, random_state=42)
        
    return dc.cleaning(df)

def main():
    st.title("Projeto de Machine Learning")
    st.subheader("Baseado em redes sociais (Instagram)")

    with st.expander("ℹ️ Sobre o Modelo"):
        st.write("""
            Este projeto utiliza **Random Forest** para prever o comportamento do usuário.
            - **Regressão:** Estima os minutos diários.
            - **Classificação:** Identifica a preferência de conteúdo (Reels, Fotos, Stories).
            - Utilizado dataset com dados do lifestyle de usuários.
        """)
    
    df_clean = load_data()
    
    if df_clean is not None:
        # --- FILTRO POR PAÍSES ---
        st.sidebar.markdown("### Treinamento de modelo por país")
        paises = df_clean["country"].unique()
        pais_selecionado = st.sidebar.selectbox("Selecione o País", paises)

        df_filtered = df_clean[df_clean["country"] == pais_selecionado]
        st.write(f"Exibindo dados para: **{pais_selecionado}**")

        # --- TREINAMENTO ---
        if st.sidebar.button("Treinar modelo"):
            model, features, erro, r2 = lo.treinar_modelo_tempo_gasto(df_filtered)
            st.success("Modelo treinado!!!")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="(MAE) Erro médio", value=f"{erro:.2f}", delta_color="inverse")            
            with col2:
                st.metric(label="(R2 Score) Precisão", value=f"{r2:.2f}")

            st.markdown("---")
            fig = gr.plot_features(model, features)
            st.plotly_chart(fig, use_container_width=True) # use_container_width é o correto para Plotly

        # --- PREDIÇÃO ---
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Gerar predição com base no perfil")
        idade = st.sidebar.slider("Idade", 13, 80, 25)
        genero = st.sidebar.selectbox("Gênero", ["Masculino", "Feminino"])
        renda = st.sidebar.selectbox("Renda", ["Low", "Middle", "High"])
        hobbies = st.sidebar.number_input("Hobbies (Qtd)", 0, 10, 2)

        if st.sidebar.button("Gerar Predição"):
            reg_model, feat_list, erro, r2 = lo.treinar_modelo_tempo_gasto(df_filtered)
            clf_model, target_encoder = lo.treinar_modelo_conteudo(df_filtered)
            
            # Convertendo entradas manuais para o formato do modelo
            gen_n = 1 if genero == "Masculino" else 0
            inc_n = {"Low": 0, "Middle": 1, "High": 2}.get(renda, 1)
            
            entrada = pd.DataFrame([[idade, gen_n, inc_n, hobbies]], columns=feat_list)
            
            tempo_pred = reg_model.predict(entrada)[0] 
            classe_num = clf_model.predict(entrada)[0]
            conteudo_pred = target_encoder.inverse_transform([classe_num])[0]
            
            st.success("Predição concluída!!!")
            st.markdown(f"""
            ### Resultado para o perfil selecionado:
            - **Tempo Estimado:** {tempo_pred:.0f} minutos por dia.
            - **Conteúdo Favorito:** {conteudo_pred}
            """)

if __name__ == "__main__":
    main()