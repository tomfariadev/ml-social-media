import streamlit as st
import pandas as pd
import data_cleaning as dc
import logic as lo
import graphics as gr
import os

@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__)
    file = os.path.join(base_path,"data", "instagram_users_lifestyle.parquet")
    df = pd.read_parquet(file)
    return dc.cleaning(df)

def main():
    with st.expander("ℹ️ Sobre o Modelo"):
        st.write("""
            Este projeto utiliza **Random Forest** para prever o comportamento do usuário.
            - **Regressão:** Estima os minutos diários.
            - **Classificação:** Identifica a preferência de conteúdo (Reels, Fotos, Stories).
            - Utilizado dataset com pouco mais de 1.500.000 registros com dados do lifestyle de usuários
        """)

    st.set_page_config("Machine learning", layout="wide")
    st.title("Projeto de machine learning")
    st.subheader("Baseado em redes sociais(Instagram)")
    
    df_clean = load_data()

    # filtro por paises
    st.sidebar.markdown("### Treinamento de modelo por país")
    paises = df_clean["country"].unique()
    pais_selecionado = st.sidebar.selectbox("Selecione o País", paises)

    df_filtered = df_clean[df_clean["country"] == pais_selecionado]
    st.write(f"Exibindo dados para: **{pais_selecionado}**")

    # Treinamento de modelo por país
    if st.sidebar.button("Treinar modelo"):
        model, features, erro, r2 = lo.treinar_modelo_tempo_gasto(df_filtered)
        st.success("Modelo treinado!!!")

        col1, col2 = st.columns(2)

        # Quanto menor melhor 
        with col1:
            st.metric(label="(MAE) Erro médio", value=f"{erro:.2f}", delta_color="inverse")            

        # Quanto mais proximo de 1.0 melhor
        with col2:
            st.metric(label="(R2 Score) Precisão", value=f"{r2:.2f}")

        st.divider()
        fig = gr.plot_features(model, features)
        st.plotly_chart(fig, width="stretch")

    # Predição com base no perfil de usuário
    st.sidebar.divider()
    st.sidebar.markdown("### Gerar predição com base no perfil de usuário")
    idade = st.sidebar.slider("Idade", 13, 80, 25)
    genero = st.sidebar.selectbox("Gênero", ["Masculino", "Feminino"])
    renda = st.sidebar.selectbox("Renda", ["Low", "Middle", "High"])
    hobbies = st.sidebar.number_input("Hobbies (Qtd)", 0, 10, 2)

    if st.sidebar.button("Gerar Predição"):       
        reg_model, feat_list, erro, r2 = lo.treinar_modelo_tempo_gasto(df_filtered)
        clf_model, target_encoder = lo.treinar_modelo_conteudo(df_filtered)
        st.success("Predição concluída!!!")
        
        entrada = pd.DataFrame([[25, 1, 2, 3]], columns=feat_list)
        
        # modelo de regressao
        tempo_pred = reg_model.predict(entrada)[0] 
        
        # modelo de classificacao
        classe_num = clf_model.predict(entrada)[0]
        conteudo_pred = target_encoder.inverse_transform([classe_num])[0]
        
        st.markdown(f"""
        ### Resultado para o perfil selecionado:
        - **Tempo Estimado:** {tempo_pred:.0f} minutos por dia.
        - **Conteúdo Favorito:** {conteudo_pred}
        """)

if __name__ == "__main__":
    main()