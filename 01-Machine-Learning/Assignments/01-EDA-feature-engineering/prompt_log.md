# Prompt Log — Prática 01: EDA e Feature Engineering com GenAI

**Aluno:** Lucas Santos
**Disciplina:** Machine Learning · Pós-Graduação em IA Generativa e Aplicações com LLMs · PUC Minas
**Professor:** Maurício Rodrigues da Silva
**Dataset:** Telco Customer Churn (7.043 clientes, 21 colunas)
**Metodologia:** PBL Aumentado: a IA sugere, eu audito, eu implemento.

## Status

| # | Etapa | Seção do notebook | Status |
|---|---|---|---|
| 1 | Primeira inspeção | 3 | ✅ Concluído |
| 2 | Distribuições | 4 | ✅ Concluído |
| 3 | Outliers (IQR) | 5 | ✅ Concluído |
| 4 | Correlação | 6 | ✅ Concluído |
| 5 | Brainstorming de features | 7 | ✅ Concluído |
| 6 | Prompt Battle (dupla) | 7 | ⬜ Não realizado (exercício em dupla, fora do checklist final) |
| 7 | Features finais implementadas | 7 | ✅ Concluído |
| 8 | Pré-processamento e split | 8 e 9 | ✅ Concluído |

Legenda: ⬜ Pendente · 🟨 Em andamento · ✅ Concluído

---

## 2. Distribuições (Seção 4)

**Minha leitura (sem IA):**

| Variável | Média | Mediana | Relação | Formato | Precisa de transformação? |
|---|---|---|---|---|---|
| `tenure` | 32,37 | 29,00 | média > mediana | Assimetria à direita leve (skew = 0,24), com picos nas pontas (clientes novos e clientes com 70–72 meses)\* | Não |
| `MonthlyCharges` | 64,76 | 70,35 | média < mediana | Assimetria à esquerda (leve, skew = −0,22) e bimodal | Não\*: assimetria leve; log ou Box-Cox não resolvem a bimodalidade |
| `TotalCharges` | 2.283,30 | 1.397,48 | média > mediana | Assimetria à direita (skew = 0,96) | Opcional\*: só para modelos lineares. Raiz quadrada reduz o skew para 0,31; log inverte para −0,74. Não aplicada nesta tarefa |

\* Preenchido pela IA (Claude Code · Claude Opus 5.5 · 27/09/2026), a pedido: "faça essa parte e três campos da tabela de distribuições: média, mediana e formato de tenure, e 'Precisa de transformação?' de MonthlyCharges e TotalCharges".

Observações (multimodalidade, picos, etc.): `MonthlyCharges` tem dois picos: um grande perto de 20 (clientes com plano básico) e outro em torno de 80–90. A leve assimetria à esquerda vem dessa mistura de grupos, não de uma cauda longa.

**Uso de IA:**
- Ferramenta / modelo / data: Claude Code (Claude Opus 5.5) · 27/09/2026
- Prompt:
  > Vamos começar com o exercicio 4 onde o monthlycharges teve uma assimetria à esquerda, e o totalCharges, assimetria a direita. Agora preciso que vc faça o grafico pra ver se o resultado será o mesmo
- Resposta (resumo): A IA gerou o histograma + KDE de `MonthlyCharges` com linhas de média e mediana (a célula de `TotalCharges` eu já tinha feito). Confirmou as duas leituras: `MonthlyCharges` média 64,76 < mediana 70,35 (skew −0,22, esquerda); `TotalCharges` média 2.283 > mediana 1.397 (skew 0,96, direita). Apontou também que `MonthlyCharges` é bimodal.
- Decisão: mantive minha leitura, que o gráfico confirmou; registrei a bimodalidade de `MonthlyCharges` como observação extra.

**Comparação: minha leitura × IA:**
- Onde concordamos: direção da assimetria: `MonthlyCharges` à esquerda e `TotalCharges` à direita (minha leitura, confirmada pelo gráfico)
- Onde divergimos (e quem estava certo): sem divergência

---

## 3. Outliers com IQR (Seção 5)

**Minha leitura (sem IA):**

| Variável | Q1 | Q3 | IQR | Limite inf. | Limite sup. | Nº outliers |
|---|---|---|---|---|---|---|
| `MonthlyCharges` | 35,50 | 89,85 | 54,35 | −46,02 | 171,38 | 0 |
| `tenure` | 9,00 | 55,00 | 46,00 | −60,00 | 124,00 | 0 |
| `TotalCharges` | 401,45 | 3.794,74 | 3.393,29 | −4.688,48 | 8.884,67 | 0 |

Minha leitura: nenhuma das três variáveis tem outlier pelo método IQR.

**Minha interpretação (sem IA):**
- Talvez a faixa de outlier (1,5 × IQR) esteja larga demais; com uma faixa menor poderiam aparecer outliers.
- `tenure`: os clientes parecem seguir um padrão de "tempo de casa". Valeria entender por que nenhum cliente fica mais tempo do que os dados mostram (máximo de 72 meses).
- `TotalCharges`: poucos clientes são cobrados em alto volume. A ausência de outliers sugere que cobrança excessiva talvez não seja o principal motivo de churn.

**Revisão da IA sobre a minha interpretação** (Claude Code · Claude Opus 5.5 · 27/09/2026):
- Prompt:
  > no 3 a minha interpretacao é que talvez o range de outlier está muito grande, se fosse um pouco meno. Mas ignorando esse fato, pro tenure, os clientes parecem ter um padrão de "tempo de casa", talvez seria interessante entender por que nenhum fica mais tempo como cliente que os dados mostram. E sobre o totalCharges, dá pra ver que poucos clientes são cobrados em alto volume e que no geral, e a ausencia de outliers indica que talvez a cobranca excessiva nao seja o motivo principal dos churns
- Faixa menor: em `tenure`, nem com 0,5 × IQR aparece outlier (faixa de −14 a 78 contra dados de 0 a 72). Em `TotalCharges`, com 1,0 × IQR surgem 272 outliers e com 0,5 × IQR surgem 935, mas são só a cauda longa de clientes antigos, não erros. Reduzir o multiplicador até "aparecer" outlier é arbitrário. Para dado assimétrico, a alternativa é aplicar o IQR depois de uma transformação log.
- Máximo de 72 meses: 362 clientes têm exatamente 72 meses e 651 têm 70 ou mais. Um pico colado no máximo indica provável **corte do dataset** (janela de coleta de ~6 anos), não que ninguém fique mais tempo. Esses clientes quase não cancelam (1,7% de churn, contra 52,9% em quem tem até 6 meses).
- Cobrança excessiva: a conclusão não se sustenta pela análise de outliers, que não olha para `Churn`. `TotalCharges` cresce com o tempo de casa, então quem cancela tem `TotalCharges` **menor** (mediana 703,55 contra 1.683,60), porque sai cedo. O sinal de preço está em `MonthlyCharges`: quem cancela paga mais por mês (mediana 79,65 contra 64,43). Ou seja, preço pode sim pesar no churn. A Seção 6 (correlação) ajuda a confirmar.

**Decisão de negócio (manter ou remover):** manter

**Uso de IA:**
- Ferramenta / modelo / data: Claude Code (Claude Opus 5.5) · 27/09/2026
- Prompt:
  > na 5, nao encontrei outlier pra tenure, nem pra totalCharges, agora faça a sua análise pra ver se eu acertei
- Resposta (resumo): Confirmou 0 outliers nas três variáveis. `tenure` vai de 0 a 72, dentro de [−60; 124]. O máximo de `TotalCharges` é 8.684,80, abaixo do limite superior de 8.884,67; é o caso que chega mais perto do limite. Os 11 `NaN` de `TotalCharges` ficam fora da conta: `quantile` ignora `NaN`, e comparar `NaN` com o limite sempre dá `False`. Adicionou ao notebook a célula de IQR + boxplot para `tenure`, que ainda não existia.
- Concordou com minha decisão? Sim, 0 outliers em `tenure` e `TotalCharges`.

---

## 4. Correlação (Seção 6)

**Minha leitura (sem IA):**
- Maior correlação: `tenure` × `TotalCharges` (r = 0,83), conforme o texto do notebook
- O que significa, com minhas palavras: talvez porque as cobranças mensais de quem cancelou estavam muito altas?
  - Revisão (versão final): `TotalCharges` é praticamente `tenure × MonthlyCharges`: quanto mais tempo de casa, mais o cliente acumula de cobrança. Por isso as duas andam juntas, e usar ambas no modelo seria redundante.
- Correlação de cada variável com `Churn`:
  - `tenure` −0,35: quanto mais tempo de casa, menos cancela.
  - `TotalCharges` −0,20: é reflexo do `tenure`.
  - `MonthlyCharges` +0,19: quem paga mais cancela um pouco mais.
- Risco de multicolinearidade e o que fazer: decisões tomadas no pré-processamento (Seção 8): retirar `Contract` (repetida por `contract_commitment_months`); juntar "No internet service" em "No" em `OnlineSecurity` e `TechSupport` (repetiam `InternetService_No`); `tenure_stage` talvez retirar

**Uso de IA:**
- Ferramenta / modelo / data: Claude Code (Claude Opus 5.5) · 27/09/2026
- Prompt:
  > 4 maior correlacao, tenure e totalCharges, conforme explicado no texto. Talvez por que as cobrancas mensais de quem churnou estavam muito altas?
- Resposta (resumo): Correlações calculadas: `tenure`×`TotalCharges` 0,826 · `MonthlyCharges`×`TotalCharges` 0,651 · `tenure`×`MonthlyCharges` 0,248. Com `Churn`: `tenure` −0,352 · `TotalCharges` −0,199 · `MonthlyCharges` +0,193. A correlação de 0,83 é **estrutural**: `TotalCharges` ≈ `tenure × MonthlyCharges` (correlação de 0,9996 entre os dois). Quanto mais meses o cliente fica, mais ele acumula de cobrança. Não tem relação com churn nem com mensalidade alta. A hipótese da mensalidade alta é outro achado, que aparece em `MonthlyCharges`×`Churn` (+0,19, fraca). Para `Churn`, o achado mais forte é `tenure` (−0,35): quanto mais tempo de casa, menor o churn. Multicolinearidade: `tenure` e `TotalCharges` carregam quase a mesma informação. Opções: descartar `TotalCharges` ou substituí-la por uma razão (ex.: `TotalCharges / tenure`).

**Pergunta de acompanhamento:**
- Prompt:
  > outra coisa, os dados me dizem que a tenure e o num_chum não tem relação, ou seja, quem cancela mais, é quem tem menos tempo de casa
- Resposta (resumo): "Quem tem menos tempo de casa cancela mais" **é** uma relação: uma correlação negativa (r = −0,35). "Sem relação" seria r ≈ 0. Taxa de churn por faixa de `tenure`: 0–6 meses 52,9% · 7–12 35,9% · 13–24 28,7% · 25–48 20,4% · 49–72 9,5%. O churn cai em todas as faixas. O r de −0,35 parece "fraco" porque o Pearson mede relação linear e `Churn` só vale 0 ou 1; a queda é forte, concentrada no início do contrato e não linear.

**Comparação: minha leitura × IA:** acertei o par com maior correlação. A explicação estava errada: confundi a relação entre duas features com a relação delas com o churn.

---

## 5. Brainstorming de features com IA (Seção 7)

- **Ferramenta / modelo / data:** Claude Code (Claude Opus 5.5) · 27/09/2026
- **Prompt usado:** (prompt padrão do notebook, sem alteração)
  > Atue como um Cientista de Dados Sênior especialista em TELECOM.
  > Dataset: tenure, MonthlyCharges, TotalCharges, Contract, PaperlessBilling, PaymentMethod.
  > Objetivo: prever CHURN.
  >
  > Crie 5 NOVAS FEATURES que ajudem a prever churn. Para cada uma, explique: nome e fórmula,
  > por que ajuda, código Pandas, e sinalize risco de DATA LEAKAGE.
- **Resposta bruta:** sessão do Claude Code de 27/09/2026 (a resposta completa, com o código Pandas, está na conversa)

**Features sugeridas:**

| # | Nome | Fórmula | Por que ajuda (segundo a IA) | Leakage apontado pela IA |
|---|---|---|---|---|
| 1 | `charge_trend_ratio` | `MonthlyCharges / (TotalCharges / tenure)`, com `tenure = 0` → `NaN` | > 1 indica que a mensalidade atual está acima da média histórica (reajuste ou upgrade), o que gera insatisfação com preço | Baixo. Os 11 `NaN` precisam ser imputados com estatística do treino. Premissa: `TotalCharges` é a foto do mesmo momento das outras colunas |
| 2 | `tenure_stage` | `pd.cut(tenure, [-1, 6, 12, 24, 48, 72])` | Captura o risco não linear: o churn se concentra nos primeiros meses | Nenhum com faixas fixas. Se usar `qcut` (quantis), as faixas precisam sair só do treino |
| 3 | `contract_commitment_months` | `Contract` → {Month-to-month: 1, One year: 12, Two year: 24} | Transforma o contrato em escala ordinal de fidelização: quanto menor o compromisso, menor o custo de sair | Nenhum (mapeamento fixo) |
| 4 | `is_manual_digital_payer` | `(PaymentMethod == "Electronic check") & (PaperlessBilling == "Yes")` | Cliente digital, mas sem débito automático: todo mês decide ativamente se paga, e cancelar tem pouca fricção | Nenhum (regra fixa por linha) |
| 5 | `charge_vs_contract_avg` | `MonthlyCharges / média de MonthlyCharges do mesmo Contract` | Cliente que paga bem acima dos pares do mesmo contrato é alvo de concorrente | **Alto** se a média for calculada no dataset inteiro. A média por contrato deve vir só do treino (depois do split) e ser aplicada ao teste |

**Checklist do Auditor:**

| # | Feature | 1. Dados existem? | 2. Faz sentido p/ negócio? | 3. Sem leakage? | 4. Viável? | Decisão | Motivo |
|---|---|---|---|---|---|---|---|
| 1 | `charge_trend_ratio` | ✅ | ✅ | ✅ (imputar `NaN` com estatística do treino) | ✅ | ✅ Aceita | Faz sentido para o negócio; cálculo simples |
| 2 | `tenure_stage` | ✅ | ✅ | ✅ (faixas fixas) | ✅ | ✅ Aceita | Faz sentido para o negócio; cálculo simples |
| 3 | `contract_commitment_months` | ✅ | ✅ | ✅ (mapeamento fixo) | ✅ | ✅ Aceita | Faz sentido para o negócio; cálculo simples |
| 4 | `is_manual_digital_payer` | ✅ | ✅ | ✅ (regra por linha) | ✅ | ✅ Aceita | Faz sentido para o negócio; cálculo simples |
| 5 | `charge_vs_contract_avg` | ✅ | ❌ | ⚠️ (só é segura se a média vier do treino) | ✅ | ❌ Descartada | Não faz sentido para o negócio |

Decisão: ✅ Aceita · 🔧 Ajustada · ❌ Descartada

Critério de viabilidade: nenhuma feature exige cálculo complexo ou combinação pesada de colunas.

**Prompt da auditoria:**
> sobre as features pode aprovar todas menos a 5, acho que todas fazem sentido pro negocio menos a 5 tambem, se é viavel acho e sim, pois nao sao calculos muito complexos e misturados

**Alucinações encontradas:** nenhuma.

---

## 6. Prompt Battle (exercício em dupla)

| | Prompt A (meu) | Prompt B (dupla: _) |
|---|---|---|
| Ferramenta / modelo | _ | _ |
| Prompt | _ | _ |
| Features sugeridas | _ | _ |
| Aprovadas no checklist | _ / 5 | _ / 5 |
| Alucinações | _ | _ |
| Riscos de leakage | _ | _ |
| Especificidade (1–5) | _ | _ |

**Veredito:** qual prompt venceu e por quê: _

**O que tornou o melhor prompt melhor** (contexto, restrições, formato de saída, persona): _

---

## 7. Features finais implementadas (Seção 7)

| Feature | Origem (exemplo do notebook / IA / própria) | Código Pandas | Cuidado tratado |
|---|---|---|---|
| `avg_monthly_spend` | Exemplo do notebook | `TotalCharges / tenure.replace(0, nan)` | Divisão por zero em `tenure = 0` |
| `is_high_risk_segment` | Exemplo do notebook | `(MonthlyCharges > 70) & (Contract == "Month-to-month")` | Limiar fixo, sem estatística global |
| `tenure_to_charge_ratio` | Exemplo do notebook | `tenure / MonthlyCharges` | Sem estatística global |
| `charge_trend_ratio` | IA (feature 1 do brainstorming) | `(MonthlyCharges / (TotalCharges / tenure.replace(0, nan))).fillna(1.0)` | `tenure = 0` (11 clientes) → 1,0 (valor fixo: cliente sem histórico, mensalidade = média), sem estatística do dataset |
| `tenure_stage` | IA (feature 2 do brainstorming) | `pd.cut(tenure, [-1, 6, 12, 24, 48, 72]).astype(str)` | Faixas fixas; entra como categórica (One-Hot) |
| `contract_commitment_months` | IA (feature 3 do brainstorming) | `Contract.map({Month-to-month: 1, One year: 12, Two year: 24})` | Mapeamento fixo |
| `is_manual_digital_payer` | IA (feature 4 do brainstorming) | `(PaymentMethod == "Electronic check") & (PaperlessBilling == "Yes")` | Regra fixa por linha |

Implementação: célula "Espaço para as features que foi validada com o Checklist do Auditor" (Seção 7); as 4 features foram incluídas no split (Seção 9). Bases finais, após todos os ajustes da Seção 8: `X_train_final` 4.930 × 27 e `X_test_final` 2.113 × 27, sem `NaN`.

---

## 8. Pré-processamento e split (Seções 8 e 9)

**Minhas decisões:**
- Dados ausentes em `TotalCharges`: estratégia **constante 0** (`SimpleImputer(strategy="constant", fill_value=0)`; proposta inicial era a mediana) · justificativa: os 11 ausentes são clientes com `tenure = 0`, que ainda não pagaram nada, então 0 é o valor real. Por ser constante, não usa estatística do dataset e pode ser aplicado antes do split sem leakage
- Encoding: **One-Hot** em `PaymentMethod`, `PaperlessBilling`, `InternetService`, `OnlineSecurity` e `TechSupport` (categorias sem ordem). **`Contract` retirado**: a informação já está em `contract_commitment_months` (1/12/24), que preserva a ordem do compromisso. `OnlineSecurity` e `TechSupport`: categoria "No internet service" juntada em "No" (evita 3 colunas idênticas a `InternetService_No`). `tenure_stage`: talvez retirar (em avaliação; por enquanto continua no One-Hot)
- Escalonamento: **`StandardScaler`** nas features numéricas contínuas, mantido como no notebook · justificativa: me parece bom. **Exceção:** `is_high_risk_segment` e `is_manual_digital_payer` ficam **fora do scaler** (lista `features_binarias`) e entram em `X_*_final` como 0/1
- Split: `test_size` = 0,3 · `stratify=y` · churn treino = 26,5% · churn teste = 26,5%
- Como evitei leakage (`fit` só no treino): split antes de qualquer fit. `fit` só no treino, `transform` no treino e no teste. Não usei a variável alvo (`Churn`) para criar features. A imputação de `TotalCharges` com 0 é feita antes do split porque é uma constante e não usa estatística do dataset

**Uso de IA:**
- Ferramenta / modelo / data: Claude Code (Claude Opus 5.5) · 27/09/2026
- Prompt:
  > agora vamos pra 8. Pré-processamento e split (Seções 8 e 9)
  > totalcharges talvez fazer com a mediana dos dados pros dados ausentes.
  > Sobre o processo de encoding ainda vou decidir
- Resposta (resumo): Os 11 ausentes são todos clientes com `tenure = 0` (nenhum cancelou). A mediana geral (1.397,48) daria a um cliente que nunca pagou um total maior que o de 50% da base; a mediana de quem tem 1 mês é 49,75. O ausente é sistemático, não aleatório, e o valor coerente com o negócio é 0 (estratégia atual do notebook, sem estatística, sem leakage). Se usar a mediana, ela precisa vir só do treino, depois do split (mediana do treino = 1.389,20; 7 ausentes no treino e 4 no teste). Impacto numérico pequeno (0,16% das linhas). Alternativa: `MonthlyCharges` (valor da 1ª fatura).
- O que aceitei / rejeitei:
  > verdade pode ser 0

  Rejeitei a mediana e aceitei a constante 0.
- Prompt (encoding e escalonamento):
  > PaymentMethod, PaperlessBilling, onehot
  > contract retirar
  > tenurestage talvez retirar, sobre os dados de escalonamento acho que pode manter, me parece bom
- Resultado: `Contract` removido de `features_categoricas` (célula do split, Seção 9). `X_train_final` 4.930 × 20 e `X_test_final` 2.113 × 20.
- Prompt (binárias):
  > tirar  is_high_risk_segment e is_manual_digital_payer  do scaller
- Resultado: nova lista `features_binarias` (célula do split, Seção 9); na célula de escalonamento/encoding essas colunas pulam o `StandardScaler` e são concatenadas sem transformação. Shape final igual (20 colunas), sem `NaN`.
- Prompt (novas categóricas):
  > incluir InternetService, OnlineSecurity e TechSupport em features_categoricas.
- Resultado: as 3 colunas foram incluídas no One-Hot (célula do split, Seção 9). `X_train_final` 4.930 × 29 e `X_test_final` 2.113 × 29, sem `NaN`. Taxa de churn: `InternetService` Fiber optic 41,9% · DSL 19,0% · sem internet 7,4%; `OnlineSecurity` No 41,8% · Yes 14,6%; `TechSupport` No 41,6% · Yes 15,2%. Alerta: `OnlineSecurity_No internet service` e `TechSupport_No internet service` são idênticas a `InternetService_No` (3 colunas iguais).
- O que aceitei / rejeitei (colunas repetidas):
  > 1 juntar
- Resultado: na célula do split (Seção 9), `OnlineSecurity` e `TechSupport` trocam "No internet service" por "No" antes do split (mapeamento fixo, sem leakage). `X_train_final` 4.930 × 27 e `X_test_final` 2.113 × 27, sem `NaN`.

