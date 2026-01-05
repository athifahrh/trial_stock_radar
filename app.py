# app.py ----------------------------------------
import streamlit as st

st.set_page_config(page_title="Stock Radar Suite", page_icon="📊", layout="wide")

st.title("📊 Stock Radar Suite")

st.markdown("""
Selamat datang di **Stock Radar Suite**.

Gunakan sidebar untuk pindah menu:

- 🚀 **Stock Radar NEW (Dynamic Only — Daily → 5m)**  
  Scan **Full IDX otomatis** (GitHub) → **daily prefilter** (cari yang “hidup”) → **5m scan** (volume flow).  

- 🧱 **Stock Radar OLD (Swing & Short-Term)**  
  Versi lama untuk screening swing/short-term yang masih pakai indikator teknikal.

**Catatan:**
- Data sumber: Yahoo Finance (yfinance) + Universe: GitHub Dataset-Saham-IDX
- Kalau hasil kosong, biasanya karena data 5m `.JK` lagi bolong / belum kebaca intraday → coba **Refresh** atau ganti period 10d.
""")
