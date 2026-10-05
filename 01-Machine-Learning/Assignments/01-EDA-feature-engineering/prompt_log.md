# Prompt Log — Tarefa 1: Etapa 5 (Feature Engineering com IA)

**Aluno:** Lucas Santos
**Disciplina:** Machine Learning · Pós-Graduação em IA Generativa e Aplicações com LLMs · PUC Minas
**Professor:** Maurício Rodrigues da Silva
**Dataset:** Telco Customer Churn (7.043 clientes, 21 colunas)
**Metodologia:** PBL Aumentado: a IA sugere, eu audito, eu implemento.

---

## Brainstorming de features com IA (Seção 7 do notebook)

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


**Checklist do Auditor, 1ª rodada (27/09/2026):** só os 4 critérios do notebook, sem olhar os dados.

| # | Feature | 1. Dados existem? | 2. Faz sentido p/ negócio? | 3. Sem leakage? | 4. Viável? | Decisão |
|---|---|---|---|---|---|---|
| 1 | `charge_trend_ratio` | ✅ | ✅ | ✅ | ✅ | ✅ Aceita |
| 2 | `tenure_stage` | ✅ | ✅ | ✅ | ✅ | ✅ Aceita |
| 3 | `contract_commitment_months` | ✅ | ✅ | ✅ | ✅ | ✅ Aceita |
| 4 | `is_manual_digital_payer` | ✅ | ✅ | ✅ | ✅ | ✅ Aceita |
| 5 | `charge_vs_contract_avg` | ✅ | ❌ | ⚠️ | ✅ | ❌ Descartada |

- Prompt da auditoria:
  > sobre as features pode aprovar todas menos a 5, acho que todas fazem sentido pro negocio menos a 5 tambem, se é viavel acho e sim, pois nao sao calculos muito complexos e misturados

Ao revisar, a 1ª rodada se mostrou rasa: o mesmo motivo para quatro features e nenhum teste com os
dados. A `charge_trend_ratio` foi aprovada mesmo sendo derivada de `TotalCharges ≈ tenure × MonthlyCharges`,
relação que eu já tinha registrado na Seção 6. Por isso refiz a auditoria com evidência.

**Checklist do Auditor, 2ª rodada (05/10/2026):** os 4 critérios do notebook mais um **5º critério
próprio, evidência nos dados** (a feature varia e separa churn de não churn?). Números calculados
**só no conjunto de treino** (mesmo split da Seção 9; 4.930 clientes, churn de 26,5%).

- Ferramenta / modelo / data: Claude Code (Claude Opus 5.5) · 05/10/2026
- Prompt (pedido das evidências):
  > sim, então vamos melhorar. Faça um plano pra gente implementar essa melhoria [...]

  Resposta (resumo): a IA calculou, para cada sugestão, variação, taxa de churn por faixa/categoria e
  correlação com colunas existentes, sem tomar decisão. As decisões abaixo são minhas.

| # | Feature | 1 | 2 | 3 | 4 | 5. Evidência nos dados | Decisão |
|---|---|---|---|---|---|---|---|
| 1 | `charge_trend_ratio` | ✅ | ✅ | ✅ | ✅ | ❌ 94,4% dos clientes entre 0,9 e 1,1; churn por faixa de 25,8% a 28,7% (base 26,5%); correlação com churn de 0,008 | ❌ Descartada (mudou) |
| 2 | `tenure_stage` | ✅ | ✅ | ✅ | ✅ | ✅ churn de 52,2% em 0–6m, 38,5% em 7–12m, 27,3% em 13–24m, 19,9% em 25–48m e 10,1% em 49–72m | ✅ Aceita |
| 3 | `contract_commitment_months` | ✅ | ✅ | ✅ | ✅ | ✅ churn de 42,9% (1 mês), 10,9% (12) e 3,0% (24); correlação de −0,395 | ✅ Aceita |
| 4 | `is_manual_digital_payer` | ✅ | ✅ | ✅ | ✅ | ✅ churn de 50,2% com a flag contra 18,7% sem; só `Electronic check` dá 45,6% | ✅ Aceita |
| 5 | `charge_vs_contract_avg` | ✅ | ✅ | ⚠️ só se a média vier do treino | ✅ | ❌ médias por contrato quase iguais (66,61 · 65,34 · 60,85); correlação de 0,996 com `MonthlyCharges` | ❌ Descartada (motivo revisto) |

Decisão: ✅ Aceita · 🔧 Ajustada · ❌ Descartada

- Prompt (minhas decisões):
  > 1) Acho que faz pouco sentido pois não nos traz correlação nenhuma com o churn. Se os dados nao sustentam, nao compensa.
  > 2) Acho que compensa
  > 3) Compensa
  > 4) Compensa
  > 5) Até compensa mas é redundante, deixa de fora. Pois outros dados ja dizem isso.

**Motivos:**
1. `charge_trend_ratio`: a ideia de negócio (reajuste gera insatisfação) é boa, mas neste dataset a
   razão fica perto de 1 para quase todos e não tem correlação com o churn. Se os dados não
   sustentam, não compensa.
2. `tenure_stage`: compensa; captura a queda não linear do churn nos primeiros meses.
3. `contract_commitment_months`: compensa; é o sinal mais forte das cinco e preserva a ordem do
   compromisso (por isso substitui `Contract` no encoding).
4. `is_manual_digital_payer`: compensa; a combinação separa mais que `Electronic check` sozinho.
5. `charge_vs_contract_avg`: até compensa pela ideia, mas é redundante: repete `MonthlyCharges`, que
   já está no modelo. Na 1ª rodada eu tinha descartado por "não fazer sentido para o negócio"; o
   motivo real é a redundância, somada ao risco de leakage.

**Alucinações encontradas:** nenhuma. Conferido assim: as 6 colunas usadas existem no dataset; os
valores de `Contract` (Month-to-month, One year, Two year), `PaymentMethod` (Electronic check) e
`PaperlessBilling` (Yes/No) batem com os usados pela IA nas fórmulas; o código das cinco sugestões
rodou sem erro, e o mapeamento de `Contract` não gerou `NaN`.

**Observação sobre os exemplos do notebook:** `avg_monthly_spend` também tem correlação de 0,996 com
`MonthlyCharges`. Mantida por ser exemplo da aula, registrada como candidata a remoção na modelagem.

A auditoria completa também está no notebook, na célula "Auditoria das features sugeridas pela IA"
(Seção 7).

---

## Features finais implementadas (Seção 7 do notebook)

| Feature | Origem (exemplo do notebook / IA / própria) | Código Pandas | Cuidado tratado |
|---|---|---|---|
| `avg_monthly_spend` | Exemplo do notebook | `TotalCharges / tenure.replace(0, nan)` | Divisão por zero em `tenure = 0`; redundante com `MonthlyCharges` (0,996) |
| `is_high_risk_segment` | Exemplo do notebook | `(MonthlyCharges > 70) & (Contract == "Month-to-month")` | Limiar fixo, sem estatística global |
| `tenure_to_charge_ratio` | Exemplo do notebook | `tenure / MonthlyCharges` | Sem estatística global |
| `tenure_stage` | IA (feature 2 do brainstorming) | `pd.cut(tenure, [-1, 6, 12, 24, 48, 72]).astype(str)` | Faixas fixas; entra como categórica (One-Hot) |
| `contract_commitment_months` | IA (feature 3 do brainstorming) | `Contract.map({Month-to-month: 1, One year: 12, Two year: 24})` | Mapeamento fixo |
| `is_manual_digital_payer` | IA (feature 4 do brainstorming) | `(PaymentMethod == "Electronic check") & (PaperlessBilling == "Yes")` | Regra fixa por linha |

Implementação: célula "Espaço para as features que foi validada com o Checklist do Auditor" (Seção 7);
as 3 features da IA foram incluídas no split (Seção 9). `charge_trend_ratio` foi removida do notebook
após a 2ª rodada. Notebook reexecutado do zero (Restart & Run All) em 05/10/2026, sem erros. Bases
finais: `X_train_final` 4.930 × 26 e `X_test_final` 2.113 × 26, sem `NaN`; churn de 26,5% em treino
e teste.
