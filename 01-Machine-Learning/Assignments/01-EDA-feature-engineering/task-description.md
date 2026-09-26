# Machine Learning

**Pos-Graduacao em IA Generativa e Aplicacoes com LLMs, PUC Minas, Prof. Mauricio Rodrigues da Silva**

## Tarefa 1: Analise Exploratoria de Dados (EDA) e Feature Engineering

*Unidade 1, Ciclo de Vida de ML e Feature Engineering*

Nesta tarefa, cada aluno vai aplicar, individualmente e sobre um dataset real, o mesmo processo de Analise Exploratoria de Dados (EDA) demonstrado em sala na pratica guiada: entendimento do problema, inspecao dos dados, analise de distribuicoes e outliers, leitura de correlacao, Feature Engineering com apoio de IA Generativa, tratamento de dados ausentes, codificacao e escalonamento, e separacao correta de treino e teste. Esta primeira entrega nao exige treinar nenhum modelo: o objetivo e consolidar bem cada etapa do processo antes de partir para a modelagem, que comeca na Aula 2 (Regressao) em diante.

## 1. Objetivos de Aprendizagem

Ao final desta tarefa, o aluno deve ser capaz de:

- Carregar e inspecionar um dataset real, identificando tipos de dado, valores ausentes e problemas de qualidade que nao aparecem em dados sinteticos.
- Diagnosticar a forma de uma distribuicao (simetrica, assimetrica a direita ou a esquerda) usando media, mediana e histograma.
- Detectar outliers pelo metodo do Intervalo Interquartil (IQR) e discutir, com base no contexto de negocio, se devem ser removidos, mantidos ou investigados.
- Interpretar uma matriz de correlacao e relacionar os achados ao problema de churn.
- Usar um LLM como copiloto de Feature Engineering, aplicando um checklist de auditoria a cada sugestao antes de implementa-la.
- Tratar dados ausentes, codificar variaveis categoricas e escalonar variaveis numericas, sempre depois da separacao entre treino e teste.

## 2. O Dataset: Telco Customer Churn

Vamos usar o dataset real Telco Customer Churn (IBM, disponibilizado publicamente no Kaggle), arquivo `WA_Fn-UseC_-Telco-Customer-Churn.csv`. Sao 7.043 clientes de uma operadora de telecomunicacoes, com colunas numericas (`tenure`, `MonthlyCharges`, `TotalCharges`) e categoricas (`Contract`, `PaymentMethod`, `InternetService`, entre outras), e a coluna alvo `Churn` (Yes/No), indicando se o cliente cancelou o servico.

O arquivo sera disponibilizado no portal, junto com o notebook-base desta tarefa. Nao e necessario baixar nada do Kaggle.

## 3. Roteiro da Tarefa

O notebook-base (`tarefa_1_eda.ipynb`) ja traz a teoria, os graficos de referencia e o codigo de exemplo de cada etapa. Complete os exercicios marcados dentro dele, na ordem:

1. Carregar os dados e fazer a primeira inspecao (shape, tipos, valores ausentes), incluindo a correcao do tipo de `TotalCharges`.
2. Analisar distribuicoes de pelo menos duas variaveis numericas, com histograma, media e mediana.
3. Detectar outliers pelo metodo IQR em pelo menos duas variaveis, e escrever uma frase de interpretacao para cada resultado.
4. Ler o heatmap de correlacao e explicar, com suas palavras, o que a correlacao mais forte revela sobre o problema de churn.
5. Fazer Feature Engineering com apoio de um LLM: aplicar o prompt sugerido (ou um proprio), auditar cada sugestao com o Checklist do Auditor, e implementar pelo menos 3 features validadas.
6. Tratar os dados ausentes reais de `TotalCharges`, codificar as variaveis categoricas (One-Hot Encoding) e escalonar as variaveis numericas (`StandardScaler`), sempre depois do split.
7. Separar treino e teste com `train_test_split(stratify=y)` e confirmar que a proporcao de churn ficou parecida nos dois conjuntos.

> **O que NAO e exigido nesta tarefa:** treinar, ajustar ou comparar qualquer modelo de Machine Learning. O notebook termina com as bases de treino e teste prontas (`X_train_final`, `X_test_final`), que serao reaproveitadas a partir da Aula 2.

## 4. Prompt Log (obrigatorio)

Documente, em uma celula de markdown no final do notebook (ou em um arquivo separado `prompt_log.md`), todos os prompts usados na Etapa 5, a ferramenta de IA escolhida (ChatGPT, Claude, Gemini etc.), e uma frase curta sobre o que o Checklist do Auditor reprovou ou aprovou em cada sugestao recebida. O prompt log tambem sera exigido, de forma acumulada, na entrega do Trabalho Orientado ao final da disciplina.

## 5. Entrega

- **Formato:** o proprio notebook `tarefa_1_eda.ipynb`, com todas as celulas executadas (Kernel > Restart & Run All antes de exportar, para garantir que roda do zero sem erro).
- **Envio:** pelo portal da disciplina, no espaco desta tarefa. [definir data e horario limite de entrega]
- **Trabalho individual.** Discussao entre colegas e bem-vinda, mas o notebook entregue deve refletir a execucao e as respostas de cada aluno.

## 6. Duvidas Frequentes

**“Preciso baixar o dataset no Kaggle?”**
Nao, o CSV ja sera disponibilizado no portal junto com o notebook.

**“Qual ferramenta de IA generativa devo usar no prompt de Feature Engineering?”**
Qualquer uma (ChatGPT, Claude, Gemini, entre outras). O que importa e documentar o prompt e aplicar o Checklist do Auditor ao resultado.

**“Posso usar outras variaveis do dataset alem das ja sugeridas no notebook?”**
Pode, e e ate incentivado. O dataset real tem mais colunas do que as usadas nos exemplos (`gender`, `InternetService`, `OnlineSecurity`, entre outras); explora-las conta a favor da Etapa 5.
