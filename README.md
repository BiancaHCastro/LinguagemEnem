# Projeto Aplicado: Aprendizado de Máquina com Dados do ENEM 2023

## 🎯 Objetivo
Este projeto investiga o potencial e os limites de algoritmos de aprendizado de máquina para prever a nota de **Linguagens, Códigos e suas Tecnologias (`NU_NOTA_LC`)** do ENEM 2023. A regressão é baseada puramente no perfil socioeconômico, educacional e demográfico dos participantes, sem utilizar outras notas da prova.

## 🛠️ Tecnologias e Ferramentas
* **Linguagem:** Python 3.10+
* **Bibliotecas:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Formatos de Dados:** `.parquet` para otimização de leitura e memória

## 📊 Estratégia de Modelagem
Seguindo os requisitos da disciplina, o projeto abrange:
1. **Engenharia de Features:** Criação e seleção de variáveis a partir do questionário socioeconômico (Q001 a Q025).
2. **Prevenção de Vazamento:** Remoção estrita de dados que dependem da realização da prova (notas de outras áreas, competências de redação, gabaritos).
3. **Redução de Dimensionalidade:** Testes utilizando PCA (Análise de Componentes Principais).
4. **Métricas de Avaliação:** RMSE, MAE e R², acompanhados de análise de resíduos por faixas de nota.
5. **Otimização:** Ajuste de hiperparâmetros (GridSearchCV/RandomizedSearchCV) com validação cruzada.

## 🚀 Como executar este projeto
1. Clone o repositório: `git clone https://github.com/seu-usuario/nome-do-repo.git`
2. Instale as dependências: `pip install -r requirements.txt`
3. Siga as instruções em `dados/raw/instrucoes_acesso_dados.md` para baixar a base do INEP.
4. Execute os notebooks sequencialmente na pasta `/notebooks`.