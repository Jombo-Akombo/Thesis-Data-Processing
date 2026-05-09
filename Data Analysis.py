# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 00:10:51 2026

@author: james
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re


# Preprocessing

# Import file and format into dataframe

Truth = pd.read_excel("ground_truth.xlsx")

CLAUDE1 = pd.read_excel("1_Claude.xlsx")
PERPLEXITY1 = pd.read_excel("1_Perplexity.xlsx")
OPENAI1 = pd.read_excel("1_OpenAI_gpt5.xlsx")
GEMINI1 = pd.read_excel("1_Gemini.xlsx")

CLAUDE2 = pd.read_excel("2_Claude.xlsx")
PERPLEXITY2 = pd.read_excel("2_Perplexity.xlsx")
OPENAI2 = pd.read_excel("2_OpenAI_gpt5.xlsx")
GEMINI2 = pd.read_excel("2_Gemini.xlsx")

CLAUDE3 = pd.read_excel("3_Claude.xlsx")
PERPLEXITY3 = pd.read_excel("3_Perplexity.xlsx")
OPENAI3 = pd.read_excel("3_OpenAI_gpt5.xlsx")
GEMINI3 = pd.read_excel("3_Gemini.xlsx")



AIS = [Truth, CLAUDE3, PERPLEXITY3, OPENAI3, GEMINI3]

cols1 = ["Bank", "Min APR", "Max APR", "Max Loan Amount", "Min Loan Amount",\
        "Min Age", "Max Term"]
    
cols = ["Min APR", "Max APR", "Max Loan Amount", "Min Loan Amount",\
            "Min Age", "Max Term"]
    
Hits = Difference = [None] * len(AIS)

for i, df in enumerate(AIS):
    df = df[cols1]
    df = df.sort_values(by='Bank')
    df = df[cols]
    df = df.reset_index(drop=True)
    df = df.replace(r"[^\d\.\-]", "", regex=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    AIS[i] = df

Truth = AIS[0] # Needed to update Truth

# Creating mask for each AI and each metric

DiffMask = HitMask = HallMask = OmMask = HiMask = LowMask = AIS[0] * np.nan 

    
for i, df in enumerate(AIS):

    # Creating a mask to find differences where neither = 0
    # Could be a problem here

    DiffMask = Truth.notna() & df.notna()
    
    # Simple subtraction at points where mask = True
    
    Difference[i] = (Truth - df).where(DiffMask, np.nan)    

    # print(Difference[i])
    # Difference.to_excel("Comparison.xlsx", index=False)



# Frequency analysis
# Failure to Retrieve


# Number of Hits

Hits = []
Hallucinations = []
Omissions = []
High = []
Low = []
MapeResultsList = []

for df in AIS:
    
    HitMask = df.eq(Truth) & df.notna() & Truth.notna()
    HitCount = HitMask.sum().sum()
    
    Hits.append(HitCount)
    
    HallMask = df.notna() & Truth.isna()
    HallCount = HallMask.sum().sum()
    
    Hallucinations.append(HallCount)
    
    OmMask = df.isna() & Truth.notna()
    OmCount = OmMask.sum().sum()
    
    Omissions.append(OmCount)
    
    HiMask = (df > Truth) & df.notna() & Truth.notna()
    HiCount = HiMask.sum().sum()
    
    High.append(HiCount)
    
    LoMask = (df < Truth) & df.notna() & Truth.notna()
    LoCount = LoMask.sum().sum()
    
    Low.append(LoCount)

    MapeMask = Truth != 0
    
    mape = (
        (df - Truth).abs() / Truth
    ).where(MapeMask)
    
    mape_col = mape.mean() * 100

    MapeResultsList.append(mape_col)

Hit_Percent = [x / (AIS[0].size)*100 for x in Hits] 


Metrics = pd.DataFrame([Hits, Hit_Percent, High, Low, Hallucinations,
                        Omissions])

Metrics.columns = ["Ground Truth", "Claude", "Perplexity", "Open AI", 
                   "Gemini"]

Metrics.index = ["Hits", "Hit Percent (%)", "Overshoots", "Undershoots", 
                   "Hallucinations", "Omissions"]

MapeResults = pd.DataFrame(MapeResultsList)

MapeResults.index = ["Ground Truth", "Claude", "Perplexity", "Open AI", 
                   "Gemini"]


MapeResults.columns = [cols]

Metrics.to_excel("Metrics3.xlsx")
MapeResults.to_excel("Mape3.xlsx")


