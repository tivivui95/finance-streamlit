import streamlit as st

st.markdown("# Pandas Profiling ❄️")
st.sidebar.markdown("# Pandas Profiling ❄️")

import pandas as pd
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

df = pd.read_csv("data/final.csv")
pr = df.profile_report()

st_profile_report(pr)