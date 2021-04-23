import yfinance as yf
import streamlit as st

st.write("""
# Simple stock price app

shown are the stock closing **price** and ***volume*** of google

""")
#define the ticker symbol
tickerSymbol = 'AAPL'
#get data on this ticker
tickerData  = yf.Ticker(tickerSymbol)
#get the historical prices for this ticker
#tickerDf = tickerData.history(period='id',start='2010-5-31',end='2020-5-31')
#open high low close volume dividends stock splits
tickerData.recommendations
