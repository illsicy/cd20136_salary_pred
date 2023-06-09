import pandas as pd
import plotly as px
import streamlit as st

st.set_page_config(page_title='电子游戏销售数据仪表板',
                   layout='wide',
                   initial_sidebar_state='collapsed')


@st.cache_data
def get_data_from_csv():
    df = pd.read_csv('data/vgsales.csv')
    df.dropna(subset=['Year'], inplace=True)
    df['Year'] = df['Year'].astype('int')
    print('get data')
    return df

df = get_data_from_csv()

# 构建筛选栏
st.sidebar.header('数据过滤')
genre = st.sidebar.multiselect(
    '请选择题材',
    options=df['Genre'].unique(),
    default=df['Genre'].unique(),
)
platform = st.sidebar.multiselect(
    '请选择平台',
    options=df['Platform'].unique(),
    default=df['Platform'].unique(),
)

df_selection = df.query(
    'Genre == @genre & Platform == @platform'
)

year_range = str(df_selection['Year'].min()) + ' - ' + str(df_selection['Year'].max())
st.title(f'电子游戏销售数据仪表板({year_range})')

# KPI计算
round_digit = 2
na_total = round(df_selection['NA_Sales'].sum(), round_digit)
eu_total = round(df_selection['EU_Sales'].sum(), round_digit)
jp_total = round(df_selection['JP_Sales'].sum(), round_digit)
other_total = round(df_selection['Other_Sales'].sum(), round_digit)
global_total = round(df_selection['Global_Sales'].sum(), round_digit)

first_column, second_column, third_column, fourth_column, fifth_column  = st.columns(5)
with first_column:
    st.subheader('全球')
    st.subheader(f'US $ {global_total}M')
with second_column:
    st.subheader('北美')
    st.subheader(f'US $ {na_total}')
with third_column:
    st.subheader('欧洲')
    st.subheader(f'US $ {eu_total}')
with fourth_column:
    st.subheader('日本')
    st.subheader(f'US $ {jp_total}')
with fifth_column:
    st.subheader('其他')
    st.subheader(f'US $ {other_total}')

st.markdown('---')

# 各地区销售额柱状图
sales_by_region = df_selection[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum().to_frame()
sales_by_region.columns = ['Total']
fig_region_sales = px.bar(
    sales_by_region,
    x='Total',
    y=sales_by_region.index,
    title='<b>各地区销售数据</b>'
)
fig_region_sales.update_layout(
    xaxis=dict(title='销售额'),
    yaxis=dict(title='区域')
)

# 按年绘制
sales_by_year = df_selection.groupby(by='Year').sum()[['Global_Sales']]

fig_year_sales = px.bar(
    sales_by_year,
    y='Global_Sales',
    x=sales_by_year.index,
    title='<b>各年销售数据</b>'
)
fig_year_sales.update_layout(
    xaxis=dict(title='年份'),
    yaxis=dict(showgrid=False, title='销售额'),

)
left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_region_sales, use_container_width=True)
right_column.plotly_chart(fig_year_sales, use_container_width=True)

hide_st_style = """
<style>
#MainMenu {display: none;}
footer {display: none;}
header {display: none;}
</style>
"""

st.markdown(hide_st_style, unsafe_allow_html=True)
