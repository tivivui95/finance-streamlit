import pandas as pd
import streamlit as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class Toc:

    def __init__(self):
        self._items = []
        self._placeholder = None
    
    def title(self, text):
        self._markdown(text, "h1")

    def header(self, text):
        self._markdown(text, "h2", " " * 2)

    def subheader(self, text):
        self._markdown(text, "h3", " " * 4)

    def placeholder(self, sidebar=False):
        self._placeholder = st.sidebar.empty() if sidebar else st.empty()

    def generate(self):
        if self._placeholder:
            self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)
    
    def _markdown(self, text, level, space=""):
        import re
        key = re.sub('[^0-9a-zA-Z]+', '-', text).lower()

        st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")


toc = Toc()

st.markdown("# Trực quan dữ liệu")
st.markdown("Dữ liệu được trực quan tại Power BI ở file dưới đây.")
with open("./data/final.pbix", "rb") as file:
    btn = st.download_button(
            label="Download PBIX",
            data=file,
            file_name="final.pbix"
          )

st.markdown("# Phân tích dữ liệu")
st.sidebar.markdown("# Phân tích dữ liệu")

st.sidebar.title("Mục lục")
toc.placeholder(sidebar=True)

df = pd.read_csv("data/final.csv")

vn_df = df.loc[df['Country Name']=='Vietnam']
vn_df.head(5)


toc.header("1. Thống kê mô tả")
st.markdown('Đây là mô tả chung của dữ liệu, với 5 dòng đầu tiên,' 
             +' để có thể biết thêm chi tiết về dữ liệu, sử dụng trang **Pandas Profiling**.')
st.dataframe(df.head(5), use_container_width=True)

st.markdown('Chúng ta sẽ lấy dữ liệu của Việt Nam để phân tích,' 
             +' Dữ liệu sẽ như sau.')
st.dataframe(vn_df, use_container_width=True)

st.markdown("Mô tả dữ liệu")
st.dataframe(vn_df.describe())
#   Phân tích hồi quy, giải thích và dự đoán: 10%– Đề xuất giải pháp:

toc.header("2. Phân tích dữ liệu đơn giản")
st.markdown("Đây là bản tương quan giữa các thuộc tính, để tìm ra thuộc tính có sự tương quan với nhau.")
st.table(vn_df.corr())

st.markdown("Từ bảng trên, ta tìm được các thuộc tính có liên quan tới nhau\n" +
            "- GDP và GDP per capita đều tăng trường theo năm (Year)\n" +
            "- Import và Export goods đều có tương quan với nhau và với GDP")
vn_df1 = vn_df.copy()

toc.header("3. Phân tích hồi quy và giải thích")

st.markdown("Đầu tiên, ta dùng mô hình hồi quy OLS để phân tích tương quan giữa GDP với Import, Export qua các năm")
model1 = smf.ols("Q('GDP (current US$)') ~ Year + Q('Imports of goods and services (% of GDP)') + Q('Exports of goods and services (% of GDP)')", data=vn_df).fit()
LRresult = (model1.summary2())
st.dataframe(LRresult.tables[0], use_container_width=True)
st.dataframe(LRresult.tables[1], use_container_width=True)
st.dataframe(LRresult.tables[2], use_container_width=True)

st.markdown("Kiểm tra bằng phương pháp chi bình phương với **GDP growth (annual %)**, ta có các kết quả sau:")
from scipy.stats import chi2_contingency
label = 'GDP growth (annual %)'
y_label = vn_df[label]
result = []
for header in vn_df.columns:
  if header == label: continue
  contingency_table = pd.crosstab(vn_df[header], vn_df[label])
  chi2, p_value, dof, expected = chi2_contingency(contingency_table)
  result.append({'feature': header, 'chi-square': chi2, 'p-value': p_value, 'expected': expected}) 
result = sorted(result, key=lambda item: item['chi-square'], reverse=True)
for i in range(len(result)):
  feature = result[i]['feature']
  chi2 = result[i]['chi-square']
  pVal = result[i]['p-value']
  st.text(f'{i+1}: {feature} [chi2 = {round(chi2)}, p-value = {round(pVal, 2)}]')

st.markdown("Từ kết quả trên, chúng tôi xác định rằng có 9 tính năng hàng đầu thể hiện sự liên kết nhiều nhất với Gdp. Vì theo mô hình ta quan sát dc `Gdp per capita` có xu hướng overfit với label `Gdp` nên ta sẽ loại bỏ thuộc tính này")
st.markdown("Đây là những dòng dữ liệu mẫu sau khi lược bỏ:")
features = [result[i]['feature'] for i in range(9)]
features.remove('GDP (current US$)')
features.remove('GDP per capita (current US$)')
for col in vn_df.columns:
  if col in features: continue
  vn_df = vn_df.drop(col, axis=1)
st.dataframe(vn_df.head(5), use_container_width=True)

st.markdown("Sau khi sơ lượt Linear Regression thì ta có những tầm nhìn là với hầu hết thuộc tính thì ta quan sát dc Gdp tăng theo hệ số của Linear là dương, và ngược lại.")
from sklearn.linear_model import LinearRegression
negative_slope = []
positive_slope = []
lr_list = []

for feature in features:
  x = np.array(vn_df[feature])
  y = np.array(y_label)
  lr = LinearRegression()
  lr.fit(x.reshape(-1, 1), y)
  slope = lr.coef_[0]
  if slope < 0:
    negative_slope.append(feature)
  else: positive_slope.append(feature)
  lr_list.append({'feature': feature, 'lr': lr})

features = positive_slope if len(positive_slope)>=len(negative_slope) else negative_slope
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
index = 0
for i in range(2):
    for j in range(3):
        ax = axes[i][j]
        if index>=len(features):
          break
        x = np.array(vn_df[features[index]])
        y = np.array(y_label)
        plt.scatter(x, y)
        m, b = np.polyfit(x, y, 1)
        line = m*x + b
        # plot the scatter plot and the regression line
        ax.scatter(x, y)
        ax.plot(x, line, color='red')
        ax.set_xlabel(features[index])
        ax.set_ylabel(label)
        index+=1

plt.tight_layout()
st.pyplot(fig)
st.markdown("Ở đây ta sẽ thử sử dụng Linear Regression để dự đoán, và thu được kết quả như sau:")
feature_pred = {'Year': 2023, 'Exports of goods and services (% of GDP)': 155}

for key in feature_pred.keys():
  for pack in lr_list:
    if key!=pack['feature']: continue
    lr = pack['lr']
    val = lr.predict(np.array(feature_pred[key]).reshape(1, -1))
    st.text(f'Predict of {key} is {val}')

toc.header("4. Dự đoán và đề xuất giải pháp")
st.markdown("Chi tiết xem tại trang Forecasting")
st.markdown("Đây là kết quả dự đoán dựa trên mô hình Forecasting")
img1 = Image.open('./images/img1.png')
st.image(img1, caption="Dự đoán tăng trưởng GDP năm 2022")
st.markdown("Dựa trên kết quả ta thấy được tăng trưởng 2022 sẽ tăng lên 6% so với dữ liệu thực tế là 8%")
img2 = Image.open('./images/img2.png')
st.image(img2, caption="Dự đoán tăng trưởng GDP năm 2022-2028 bằng Dense")

img3 = Image.open('./images/img3.png')
st.image(img3, caption="Dự đoán tăng trưởng GDP năm 2022 bằng Conv")
st.markdown("Dựa trên kết quả ta thấy những năm tiếp theo sẽ tăng trưởng mạnh")
toc.generate()