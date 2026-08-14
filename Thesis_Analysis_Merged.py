# -*- coding: utf-8 -*-
"""
MERGED ANALYSIS FILE — single source for every number in the thesis.

This file combines, without changing any logic:

  PART A  = James's Data Analysis.py
            (cleaning + pooled hits, hit %, overshoots, undershoots,
             hallucinations, omissions, and MAPE per variable),
            now looped over all three runs so Metrics1/2/3.xlsx and
            Mape1/2/3.xlsx come from one execution instead of three
            manual edits.

  PART B  = Data Analysis Extended.py
            (per-variable hit rates, RQ1 Spearman + consumer cost,
             RQ2 consistency + variance ratio, RQ3 sparbank +
             aggregator Fisher tests).

Run it from inside the "James_ Code Data and Analysis" folder:

    python "Thesis_Analysis_Merged.py"

It reads ground_truth.xlsx and 1_Claude.xlsx ... 3_Perplexity.xlsx from
the current folder and writes all output spreadsheets next to them.

Definitions used everywhere (no tolerances anywhere in this file):
  hit           = AI value exactly equals ground truth value (both non-missing)
  overshoot     = AI value > ground truth value
  undershoot    = AI value < ground truth value
  hallucination = AI returns a value where ground truth has none
  omission      = AI returns nothing where ground truth has a value
  MAPE          = mean(|AI - truth| / truth) per variable, truth != 0

@author: james + matiss
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, fisher_exact

# =============================================================================
# 0.  LOAD AND CLEAN
# =============================================================================

Truth_raw = pd.read_excel("ground_truth.xlsx")

RAW_FILES = {
    1: {"Claude": "1_Claude.xlsx", "Gemini": "1_Gemini.xlsx",
        "OpenAI": "1_OpenAI_gpt5.xlsx", "Perplexity": "1_Perplexity.xlsx"},
    2: {"Claude": "2_Claude.xlsx", "Gemini": "2_Gemini.xlsx",
        "OpenAI": "2_OpenAI_gpt5.xlsx", "Perplexity": "2_Perplexity.xlsx"},
    3: {"Claude": "3_Claude.xlsx", "Gemini": "3_Gemini.xlsx",
        "OpenAI": "3_OpenAI_gpt5.xlsx", "Perplexity": "3_Perplexity.xlsx"},
}
RAW = {run: {m: pd.read_excel(path) for m, path in files.items()}
       for run, files in RAW_FILES.items()}

cols1 = ["Bank", "Min APR", "Max APR", "Max Loan Amount", "Min Loan Amount",
         "Min Age", "Max Term"]
cols  = ["Min APR", "Max APR", "Max Loan Amount", "Min Loan Amount",
         "Min Age", "Max Term"]

# Keep only digits/dot/minus, force to numeric.
def clean(df):
    df = df[cols1].sort_values(by="Bank").reset_index(drop=True)
    banks = df["Bank"].copy()
    df = df[cols].replace(r"[^\d\.\-]", "", regex=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.insert(0, "Bank", banks)
    return df

 # ground truth, cleaned, with Bank

T = clean(Truth_raw)                      
RUNS = {run: {m: clean(df) for m, df in models.items()}
        for run, models in RAW.items()}
MODELS = ["Claude", "Gemini", "OpenAI", "Perplexity"]


# =============================================================================
# PART A — FOUNDATIONAL METRICS  (from Data Analysis.py, looped over runs)
# =============================================================================
# For each run: hits, hit %, overshoots, undershoots, hallucinations,
# omissions, and MAPE per variable. Writes Metrics{run}.xlsx and 
# Mape{run}.xlsx.


Truth_nums = T[cols]                       # numeric-only ground truth

for run_id in (1, 2, 3):
    frames = [Truth_nums,
              RUNS[run_id]["Claude"][cols],
              RUNS[run_id]["Perplexity"][cols],
              RUNS[run_id]["OpenAI"][cols],
              RUNS[run_id]["Gemini"][cols]]

    Hits, Hallucinations, Omissions, High, Low, MapeResultsList = \
        [], [], [], [], [], []

    for df in frames:
        HitMask = df.eq(Truth_nums) & df.notna() & Truth_nums.notna()
        Hits.append(HitMask.sum().sum())

        HallMask = df.notna() & Truth_nums.isna()
        Hallucinations.append(HallMask.sum().sum())

        OmMask = df.isna() & Truth_nums.notna()
        Omissions.append(OmMask.sum().sum())

        HiMask = (df > Truth_nums) & df.notna() & Truth_nums.notna()
        High.append(HiMask.sum().sum())

        LoMask = (df < Truth_nums) & df.notna() & Truth_nums.notna()
        Low.append(LoMask.sum().sum())

        MapeMask = Truth_nums != 0
        mape = ((df - Truth_nums).abs() / Truth_nums).where(MapeMask)
        MapeResultsList.append(mape.mean() * 100)

    Hit_Percent = [x / Truth_nums.size * 100 for x in Hits]

    Metrics = pd.DataFrame([Hits, Hit_Percent, High, Low,
                            Hallucinations, Omissions])
    Metrics.columns = ["Ground Truth", "Claude", "Perplexity",
                       "Open AI", "Gemini"]
    Metrics.index = ["Hits", "Hit Percent (%)", "Overshoots", "Undershoots",
                     "Hallucinations", "Omissions"]

    MapeResults = pd.DataFrame(MapeResultsList)
    MapeResults.index = ["Ground Truth", "Claude", "Perplexity",
                         "Open AI", "Gemini"]
    MapeResults.columns = [cols]

    Metrics.to_excel(f"Metrics{run_id}.xlsx")
    MapeResults.to_excel(f"Mape{run_id}.xlsx")

    print("=" * 70)
    print(f"PART A — RUN {run_id}: pooled metrics "
          "(hits / overshoots / undershoots / hallucinations / omissions)")
    print("=" * 70)
    print(Metrics.round(1).to_string())


# =============================================================================
# PART B / 1.  PER-VARIABLE HIT RATE  (thesis Table 1)
# =============================================================================

def per_variable_hits(model_df, truth_df=T):
    """Return a Series: variable -> hit rate (%) ignoring NaNs in either side."""
    out = {}
    for c in cols:
        a = model_df[c]
        t = truth_df[c]
        mask = a.notna() & t.notna()
        if mask.sum() == 0:
            out[c] = np.nan
        else:
            out[c] = 100.0 * (a[mask] == t[mask]).sum() / mask.sum()
    return pd.Series(out)

per_var_rows = []
for run_id, models in RUNS.items():
    for m, df in models.items():
        s = per_variable_hits(df)
        s["Run"] = run_id
        s["Model"] = m
        per_var_rows.append(s)
PerVarHits = pd.DataFrame(per_var_rows).set_index(["Run", "Model"])[cols]
PerVarHits.to_excel("PerVariableHits.xlsx")


# =============================================================================
# PART B / 2.  RQ1 — RANKING DISTORTION (Spearman) and CONSUMER COST (SEK)
# =============================================================================

def spearman_min_apr(model_df, truth_df=T):
    a = model_df["Min APR"]
    t = truth_df["Min APR"]
    mask = a.notna() & t.notna()
    if mask.sum() < 5:
        return np.nan, np.nan, mask.sum()
    rho, p = spearmanr(a[mask], t[mask])
    return rho, p, mask.sum()

rank_rows = []
for run_id, models in RUNS.items():
    for m, df in models.items():
        rho, p, n = spearman_min_apr(df)
        rank_rows.append({"Run": run_id, "Model": m,
                          "Spearman rho": rho, "p-value": p, "n banks": n})
RankCorr = pd.DataFrame(rank_rows)
RankCorr.to_excel("RankingCorrelation.xlsx", index=False)


# Consumer cost: standard annuity, 200,000 SEK over 60 months.
PRINCIPAL = 200_000
MONTHS    = 60

def total_interest(apr_pct, P=PRINCIPAL, n=MONTHS):
    """Total interest on an annuity loan, given APR as a percentage."""
    if apr_pct is None or pd.isna(apr_pct) or apr_pct < 0:
        return np.nan
    r = (apr_pct / 100.0) / 12.0
    if r == 0:
        return 0.0
    payment = P * r / (1 - (1 + r) ** (-n))
    return payment * n - P

def cheapest_bank(df):
    sub = df.dropna(subset=["Min APR"])
    if sub.empty:
        return None, np.nan
    row = sub.loc[sub["Min APR"].idxmin()]
    return row["Bank"], float(row["Min APR"])

truth_bank, truth_apr = cheapest_bank(T)
truth_interest = total_interest(truth_apr)

cost_rows = []
for run_id, models in RUNS.items():
    for m, df in models.items():
        ai_bank, ai_apr = cheapest_bank(df)
        if ai_bank is not None and ai_bank in T["Bank"].values:
            real_apr_for_ai_pick = float(
                T.loc[T["Bank"] == ai_bank, "Min APR"].iloc[0])
        else:
            real_apr_for_ai_pick = np.nan
        phantom_cost = total_interest(ai_apr) - truth_interest
        real_cost    = total_interest(real_apr_for_ai_pick) - truth_interest
        cost_rows.append({
            "Run": run_id, "Model": m,
            "AI's cheapest bank": ai_bank,
            "AI's reported APR": ai_apr,
            "True APR of that bank": real_apr_for_ai_pick,
            "True cheapest bank": truth_bank,
            "True best APR": truth_apr,
            "Phantom cost (SEK)": round(phantom_cost, 0)
                                   if not pd.isna(phantom_cost) else np.nan,
            "Real cost if followed (SEK)": round(real_cost, 0)
                                   if not pd.isna(real_cost) else np.nan,
        })
ConsumerCost = pd.DataFrame(cost_rows)
ConsumerCost.to_excel("ConsumerCost.xlsx", index=False)


# =============================================================================
# PART B / 3.  RQ2 — CONSISTENCY ACROSS RUNS, AND VARIANCE RATIO
# =============================================================================

def consistency_by_variable(model_name):
    r1 = RUNS[1][model_name].set_index("Bank")
    r2 = RUNS[2][model_name].set_index("Bank")
    r3 = RUNS[3][model_name].set_index("Bank")
    out = {}
    for c in cols:
        triple = pd.concat([r1[c], r2[c], r3[c]], axis=1)
        triple.columns = ["r1", "r2", "r3"]
        triple = triple.dropna()
        if len(triple) == 0:
            out[c] = (np.nan, 0)
            continue
        agree = ((triple["r1"] == triple["r2"]) &
                 (triple["r2"] == triple["r3"])).sum()
        out[c] = (100.0 * agree / len(triple), len(triple))
    return out

consistency_rows = []
for m in MODELS:
    res = consistency_by_variable(m)
    for c, (pct, n) in res.items():
        consistency_rows.append({
            "Model": m, "Variable": c,
            "All-three-runs-agree (%)": pct,
            "n banks (answered all 3 runs)": n
        })
Consistency = pd.DataFrame(consistency_rows)
Consistency.to_excel("Consistency.xlsx", index=False)


def variance_ratio_min_apr(model_df, truth_df=T):
    a = model_df["Min APR"]
    t = truth_df["Min APR"]
    mask = a.notna() & t.notna()
    if mask.sum() < 5:
        return np.nan, np.nan, np.nan
    sd_ai    = float(np.std(a[mask], ddof=1))
    sd_truth = float(np.std(t[mask], ddof=1))
    return sd_ai, sd_truth, sd_ai / sd_truth if sd_truth > 0 else np.nan

var_rows = []
for run_id, models in RUNS.items():
    for m, df in models.items():
        sd_ai, sd_t, ratio = variance_ratio_min_apr(df)
        var_rows.append({"Run": run_id, "Model": m,
                         "SD AI": sd_ai, "SD Truth": sd_t,
                         "Ratio (AI / Truth)": ratio})
VarianceRatio = pd.DataFrame(var_rows)
VarianceRatio.to_excel("VarianceRatio.xlsx", index=False)


# =============================================================================
# PART B / 4.  RQ3 — SPARBANK vs NON-SPARBANK and CITATION SOURCE
# =============================================================================

def is_sparbank(name):
    return ("sparbank" in name.lower()) if isinstance(name, str) else False

T["IsSparbank"] = T["Bank"].apply(is_sparbank)

def hit_on_min_apr(model_df, truth_df=T):
    """Return a DataFrame indexed by Bank with a boolean 'hit' column."""
    merged = model_df[["Bank", "Min APR"]].merge(
        truth_df[["Bank", "Min APR", "IsSparbank"]],
        on="Bank", suffixes=("_ai", "_truth"))
    merged["hit"] = (merged["Min APR_ai"].notna()
                     & merged["Min APR_truth"].notna()
                     & (merged["Min APR_ai"] == merged["Min APR_truth"]))
    return merged

long_rows = []
for run_id, models in RUNS.items():
    for m, df in models.items():
        merged = hit_on_min_apr(df)
        for _, r in merged.iterrows():
            long_rows.append({
                "Run": run_id, "Model": m, "Bank": r["Bank"],
                "Hit": bool(r["hit"]),
                "IsSparbank": bool(r["IsSparbank"]),
            })
HitsLong = pd.DataFrame(long_rows)

sparbank_rows = []
for m in MODELS:
    sub = HitsLong[HitsLong["Model"] == m]
    a = ((sub["IsSparbank"]) & (sub["Hit"])).sum()
    b = ((sub["IsSparbank"]) & (~sub["Hit"])).sum()
    c = ((~sub["IsSparbank"]) & (sub["Hit"])).sum()
    d = ((~sub["IsSparbank"]) & (~sub["Hit"])).sum()
    if (a + b) == 0 or (c + d) == 0:
        odds, p = np.nan, np.nan
    else:
        odds, p = fisher_exact([[a, b], [c, d]])
    sparbank_rows.append({
        "Model": m,
        "Sparbank hit %":     100.0 * a / (a + b) if (a + b) > 0 else np.nan,
        "Non-sparbank hit %": 100.0 * c / (c + d) if (c + d) > 0 else np.nan,
        "Fisher OR": odds, "p-value": p,
        "n_sparbank": a + b, "n_non_sparbank": c + d,
    })
SparbankFisher = pd.DataFrame(sparbank_rows)
SparbankFisher.to_excel("SparbankFisher.xlsx", index=False)


# Modal-baseline check.
sparbank_truth = T[T["IsSparbank"] & T["Min APR"].notna()]
if len(sparbank_truth) > 0:
    modal_apr   = sparbank_truth["Min APR"].mode().iloc[0]
    modal_share = (sparbank_truth["Min APR"] == modal_apr).mean()
else:
    modal_apr, modal_share = np.nan, np.nan

baseline_rows = [{
    "Sparbank count (truth, with Min APR)": len(sparbank_truth),
    "Modal Min APR (%)": modal_apr,
    "Share of sparbanks at modal APR": modal_share,
    "Modal-baseline accuracy (%)": 100 * modal_share,
}]
ModalBaseline = pd.DataFrame(baseline_rows)
ModalBaseline.to_excel("ModalBaseline.xlsx", index=False)


# Citation source: was an aggregator cited?
AGGREGATOR_DOMAINS = [
    "kreditkoll", "lanekoll", "trygga.com", "enklare.se", "bank24.se",
    "mystep.se", "finansvalp", "finansfreak", "compricer", "consector",
    "lendo", "zmarta", "lanesite", "rocker", "lana-pengar",
    "thebanks.eu", "ekonomifokus", "lexly.se",
]

def cited_aggregator(cell):
    if not isinstance(cell, str):
        return False
    s = cell.lower()
    return any(dom in s for dom in AGGREGATOR_DOMAINS)

CITATION_COL_CANDIDATES = [
    "Cited URLs\n(shaped the answer)",      # Claude, OpenAI, Perplexity
    "Cited Domains\n(shaped the answer)",   # Gemini (exposes only domains)
    "Cited URLs",
    "Cited Domains",
    "Citations",
    "All Result URLs\n(returned by search engine)",
]

def read_citation_column(raw):
    for c in CITATION_COL_CANDIDATES:
        if c in raw.columns:
            return raw[["Bank", c]].rename(columns={c: "Citations"})
    return raw[["Bank"]].assign(Citations=np.nan)

cit_rows = []
for run_id, models in RAW.items():
    for m, raw in models.items():
        cit = read_citation_column(raw)
        cit["Aggregator"] = cit["Citations"].apply(cited_aggregator)
        ai_clean = RUNS[run_id][m]
        merged = cit.merge(ai_clean[["Bank", "Min APR"]], on="Bank", how="left")
        merged = merged.merge(T[["Bank", "Min APR"]],
                              on="Bank", suffixes=("_ai", "_truth"))
        merged["Hit"] = (merged["Min APR_ai"].notna()
                         & merged["Min APR_truth"].notna()
                         & (merged["Min APR_ai"] == merged["Min APR_truth"]))
        for _, r in merged.iterrows():
            cit_rows.append({"Run": run_id, "Model": m, "Bank": r["Bank"],
                             "Aggregator cited": bool(r["Aggregator"]),
                             "Hit": bool(r["Hit"])})
CitLong = pd.DataFrame(cit_rows)

cit_fisher = []
for m in MODELS:
    sub = CitLong[CitLong["Model"] == m]
    a = ((sub["Aggregator cited"]) & (sub["Hit"])).sum()
    b = ((sub["Aggregator cited"]) & (~sub["Hit"])).sum()
    c = ((~sub["Aggregator cited"]) & (sub["Hit"])).sum()
    d = ((~sub["Aggregator cited"]) & (~sub["Hit"])).sum()
    if (a + b) == 0 or (c + d) == 0:
        odds, p = np.nan, np.nan
    else:
        odds, p = fisher_exact([[a, b], [c, d]])
    cit_fisher.append({
        "Model": m,
        "Hit % (aggregator cited)":     100.0 * a / (a + b) if (a + b) > 0 else np.nan,
        "Hit % (aggregator NOT cited)": 100.0 * c / (c + d) if (c + d) > 0 else np.nan,
        "Fisher OR": odds, "p-value": p,
        "n_aggregator": a + b, "n_not_aggregator": c + d,
    })
CitationFisher = pd.DataFrame(cit_fisher)
CitationFisher.to_excel("CitationFisher.xlsx", index=False)


# =============================================================================
# 5.  SUMMARY PRINTOUT
# =============================================================================
print("\n" + "=" * 70)
print("PER-VARIABLE HIT RATE (%)  — by run and model  [thesis Table 1]")
print("=" * 70)
print(PerVarHits.round(1))

print("\n" + "=" * 70)
print("RQ1a — RANKING CORRELATION (Min APR, Spearman)  [thesis Table 2]")
print("=" * 70)
print(RankCorr.round(3).to_string(index=False))

print("\n" + "=" * 70)
print("RQ1b — CONSUMER COST (200,000 SEK / 60 months)  [thesis Table 3]")
print("=" * 70)
print(ConsumerCost.to_string(index=False))

print("\n" + "=" * 70)
print("RQ2b — VARIANCE RATIO (AI SD / Truth SD, Min APR)  [thesis Table 4]")
print("=" * 70)
print(VarianceRatio.round(3).to_string(index=False))

print("\n" + "=" * 70)
print("RQ2a — RUN-TO-RUN CONSISTENCY (all 3 runs match)  [thesis Table 5]")
print("=" * 70)
print(Consistency.round(1).to_string(index=False))

print("\n" + "=" * 70)
print("RQ3a — SPARBANK Fisher test (Min APR)  [thesis Table 6]")
print("=" * 70)
print(SparbankFisher.round(3).to_string(index=False))

print("\n" + "=" * 70)
print("RQ3a (defence) — Modal-APR baseline among sparbanks")
print("=" * 70)
print(ModalBaseline.to_string(index=False))

print("\n" + "=" * 70)
print("RQ3b — AGGREGATOR-cited Fisher test (Min APR)  [thesis Table 7]")
print("=" * 70)
print(CitationFisher.round(3).to_string(index=False))
