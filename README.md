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
1. Clone o repositório: `git clone https://github.com/BiancaHCastro/LinguagemEnem.git`
2. Instale as dependências: `pip install -r requirements.txt`
3. Baixe os microdados oficiais do INEP e coloque `MICRODADOS_ENEM_2023.csv` em
   `microdados_enem_2023/DADOS/`, conforme o caminho em `src/dados/configuracao.py`.
4. Execute os notebooks sequencialmente na pasta `/notebooks`.

## Executar a modelagem ENEM

O experimento adicional está em `src/modelagem/KNN_Boosting.py` e usa o conjunto
C07 em `dados/processed/enem_2023_lc_c07.parquet`. Com o CSV oficial do ENEM
disponível no caminho indicado em `src/dados/configuracao.py`, gere o conjunto:

```powershell
python -m src.conjuntos.exportar_conjunto C07
```

Na raiz do repositório, execute a análise completa ou retome uma execução
interrompida:

```powershell
python -m src.modelagem.KNN_Boosting completo
python -m src.modelagem.KNN_Boosting retomar
```

As métricas e mensagens ficam no terminal e são registradas em
`resultados/resultados_execucao.txt` e `resultados/resultados_execucao.csv`.
Também são gerados arquivos individuais por modelo, como
`resultados/resultados_knn.csv` e `resultados/resultados_boosting.txt`, para KNN,
Boosting, Linear e Ridge. As figuras são salvas em `resultados/figuras/` e as
específicas incluem o identificador do modelo no nome. Bases Parquet e
microdados são locais e estão excluídos do Git; esses arquivos são separados
dos `resultados/experimentos.csv` existentes.
