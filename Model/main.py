import os
import streamlit as st
import pandas as pd
import numpy as np


st.title('Ứng dụng Streamlit đầu tiên của tôi')

st.write("Đây là một ứng dụng Streamlit đơn giản. Hãy cùng thêm một biểu đồ.")

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['A', 'B', 'C'])

st.line_chart(chart_data)

if st.checkbox('Hiển thị dữ liệu thô'):
    st.subheader('Dữ liệu thô')
    st.write(chart_data)