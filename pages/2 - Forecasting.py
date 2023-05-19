import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

st.sidebar.markdown("# Forecasting")

st.title("Dự đoán tăng trưởng GDP các năm tới của Việt Nam")
st.markdown("Chi tiết về các mô hình được đính kèm ở link dưới đây:")
st.markdown("[Google Colab](https://colab.research.google.com/drive/1Rx56PbUX8i1IiRF78OWhLsarkfiA2MEz?usp=sharing)")
df_raw = pd.read_csv('./data/final.csv')
df = df_raw[df_raw['Country Code'] == 'VNM']
date_time = pd.to_datetime(df.pop('Year'), format='%Y')
plot_cols = ['GDP (current US$)', 'GDP per capita (current US$)', 'GDP growth (annual %)']
df = df[plot_cols]

st.markdown("Ở đây, chúng ta sẽ xét 3 trường chính về GDP của Việt Nam như sau: `GDP (current US$)`, `GDP per capita (current US$)`, `GDP growth (annual %)`")
st.dataframe(df.head(), use_container_width=True)

st.markdown("Biểu đồ về dữ liệu của 3 trường trên qua các năm:")
plot_cols = ['GDP (current US$)', 'GDP per capita (current US$)', 'GDP growth (annual %)']
plot_features = df['GDP (current US$)']
plot_features.index = date_time
# st.pyplot(plot_features.plot(subplots=True))
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 12))
for i in range(3):
    ax = axes[i]
    ax.plot(date_time, df[plot_cols[i]])
    ax.set_xlabel("Years")
    ax.set_ylabel(plot_cols[i])
st.pyplot(fig)

st.markdown("Mô tả dữ liệu")
st.dataframe(df.describe(), use_container_width=True)

st.markdown("### Dự đoán từ các mô hình")

st.markdown("Từ các trường dữ liệu về GDP, ta thấy được `GDP growth` có thể sử dụng mô hình Time-series Forecasting để dự đoán.")
st.markdown("Ta sẽ so sánh hiệu quả giữa các mô hình bằng độ đo MAE (Trung bình độ lỗi), vì thế, mô hình nào có độ lỗi càng thấp càng tốt")
models = Image.open('./images/multi.png')
st.image(models, caption="Performance of all tested models")

st.markdown("Ta thấy được dense và conv (convolutional) là 2 mô hình có độ lỗi thấp nhất, vì thế ta sẽ dùng 2 mô hình này dự đoán, đồng thời ta sẽ dùng dense để dự đoán mẫu trước 1 năm.")
img1 = Image.open('./images/img1.png')
st.image(img1, caption="Dự đoán tăng trưởng GDP năm 2022")

img2 = Image.open('./images/img2.png')
st.image(img2, caption="Dự đoán tăng trưởng GDP năm 2022-2028 bằng Dense")

img3 = Image.open('./images/img3.png')
st.image(img3, caption="Dự đoán tăng trưởng GDP năm 2022 bằng Conv")