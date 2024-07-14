# Import Libraries
import pandas as pd
import numpy as np

# import yfinance as yf

import matplotlib.pyplot as plt
import plotly.express as px

import streamlit as st

# Define Custom Functions

def cosine_similarity(a:np.array, b:np.array) -> float:
    result = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return result

# def load_data(ticker:str, days:int=10, period:str="10y") -> pd.DataFrame:
#     df_ = yf.download(ticker, period=period)
#     close_ = df_[["Close"]].reset_index()
#     now_ = close_.iloc[-days:].reset_index(drop=True)
#     past_ = close_.iloc[:-1].reset_index(drop=True)
#     return now_, past_

# Streamlit
st.title("Stock Prediction")

col1, col2, col3, col4 = st.columns(4)

with col1:
    TICKER = st.selectbox(
        "Select stock code:",
        ["AAPL"]
    )

with col2:
    DAYS_BEFORE = st.selectbox(
        "Select analysis period (no. days):",
        [10]
    )

with col3:
    DAYS_AFTER = st.selectbox(
        "Select prediction period (no. days):",
        [5]
    )

with col4:
    THRESHOLD = st.selectbox(
        "Select threshold:",
        [0.99, 0.98, 0.97]
    )

st.divider()
    
# now, past = load_data(ticker=TICKER, days=DAYS_BEFORE)
csv_path = "./data/AAPL.csv"
df = pd.read_csv(csv_path)
close = df[["Date", "Close"]]
now = close.iloc[-DAYS_BEFORE:]
past = close.iloc[:-1]

TODAY = now.iloc[-1]["Date"]
st.write("Date:", TODAY)

now_max = np.max(now["Close"].values)
now_min = np.min(now["Close"].values)
now["Minmax"] = (now["Close"] - now_min) / (now_max - now_min)

dic = {}
for idx, _ in past.iterrows():
    x = past.iloc[idx-DAYS_BEFORE:idx].reset_index(drop=True)
    y = past.iloc[idx-1:idx+DAYS_AFTER-1].reset_index(drop=True)
    if len(x) == DAYS_BEFORE:
        if len(y) == DAYS_AFTER:
            x_max = np.max(x["Close"].values)
            x_min = np.min(x["Close"].values)
            x["Minmax"] = (x["Close"].values - x_min) / (x_max - x_min)
            y["Minmax"] = (y["Close"].values - x_min) / (x_max - x_min)
            y_return = (y.iloc[4]["Close"] / y.iloc[0]["Close"]) - 1
            dic[idx] = (x, y, y_return)
    else:
        pass

cos_sim = {}
for key, val in dic.items():
    result = cosine_similarity(now["Minmax"].values, val[0]["Minmax"].values)
    cos_sim[key] = result
    
high_sim_idx = [key for key, val in cos_sim.items() if val >= THRESHOLD]

fig1 = px.line(now, x=now.index, y="Minmax")
for idx in high_sim_idx:
    x_df, y_df = dic[idx][0], dic[idx][1]
    fig1.add_scatter(x=x_df.index, y=x_df["Minmax"], mode="lines", line=dict(color="gray"), opacity=0.5)

    y_return = y_df.iloc[DAYS_AFTER-1]["Close"] / y_df.iloc[0]["Close"] - 1

    if y_return > 0:
        fig1.add_scatter(x=x_df.index+(DAYS_BEFORE-1), y=y_df["Minmax"], mode="lines", line=dict(color="red"))
    else:
        fig1.add_scatter(x=x_df.index+(DAYS_BEFORE-1), y=y_df["Minmax"], mode="lines", line=dict(color="blue"))

st.plotly_chart(fig1)

# win_array = np.array()
# for idx in high_sim_idx:
#     dic[idx][2] = y_return
#     if y_return > 0:
#         case_ = "Win"
#     elif y_return == 0:
#         case_ = "Draw"
#     else:
#         case_ = "Lose"
#     tup_ = np.array(tuple(case, y_return))
#     np.append(win_array, tup_)
    
# win_df = pd.DataFrame(win_array, columns=["Case", "Returns"])
    
# fig2 = px.pie(win_df, values="Returns", names="Case", color="Case")

# st.plotly_chart(fig2)
