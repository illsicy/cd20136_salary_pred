#graf lib
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
#preprocess lib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#ml lib (d tree)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title='Data Science Job Salary Prediction',
                   layout='wide',
                   initial_sidebar_state='collapsed')

@st.cache_data
def get_data_from_csv():
    df = pd.read_csv('ds_salaries.csv')
    #st.write(df.isnull().sum())
    df.dropna(inplace=True)
    df = df.drop(["index", "work_year", "salary", "salary_currency", "employee_residence", "company_location"], axis=1)
    df['remote_ratio'] = df['remote_ratio'].replace({0: 'Non-Remote', 50: 'Partial', 100: 'Remote'})
    #df.dropna(subset=['Year'], inplace=True)
    #df['Year'] = df['Year'].astype('int')
    #print('get data')
    return df

df = get_data_from_csv()

# 构建筛选栏
st.sidebar.header('Data Filter')
employment_type = st.sidebar.multiselect(
    'Select Employment Type',
    options=df['employment_type'].unique(),
    default=df['employment_type'].unique(),
)
experience_level = st.sidebar.multiselect(
    'Select Experience Level',
    options=df['experience_level'].unique(),
    default=df['experience_level'].unique(),
)
remote_ratio = st.sidebar.multiselect(
    'Select Remote Working Ratio',
    options=df['remote_ratio'].unique(),
    default=df['remote_ratio'].unique(),
)
company_size = st.sidebar.multiselect(
    'Select company size',
    options=df['company_size'].unique(),
    default=df['company_size'].unique(),
)
job_title = st.sidebar.multiselect(
    'Select a Job Tittle',
    options=df['job_title'].unique(),
    default=df['job_title'].unique(),
)



df_selection = df.query(
    'company_size == @company_size & remote_ratio == @remote_ratio & employment_type == @employment_type & job_title == @job_title & experience_level == @experience_level'
)

st.title(f'Data Science Job Salary Prediction Application \n base on data 2020-2022 (Made by Lim Chen Gen - CD20136)')

st.markdown('---')


sales_by_year = df_selection.groupby('experience_level')['salary_in_usd'].mean().reset_index()


fig_year_sales = px.bar(
    sales_by_year,
    y='salary_in_usd',
    x='experience_level',
    title='<b>Average Salary base on Experience Level</b>'
)
fig_year_sales.update_layout(
    xaxis=dict(title='Experience Level'),
    yaxis=dict(showgrid=False, title='Average Salary'),

)

sales_by_year = df_selection.groupby('employment_type')['salary_in_usd'].mean().reset_index()


fig_region_sales = px.bar(
    sales_by_year,
    y='salary_in_usd',
    x='employment_type',
    title='<b>Average Salary base on Employment Type</b>'
)
fig_region_sales.update_layout(
    xaxis=dict(title='Employment type'),
    yaxis=dict(showgrid=False, title='Average Salary'),

)

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_region_sales, use_container_width=True)
right_column.plotly_chart(fig_year_sales, use_container_width=True)

left_column.write("CT = Contact, FL = Freelancer, FT = Full time, PT = Part Time")
right_column.write("EN = Entry,  EX = Executive, MI = Mid, SE = Senior")

################machine learning stuff#########################
################input deal#########################
left_column2, right_column2 = st.columns(2)
selected_employment_type = left_column2.selectbox("Select Employment Type", df['employment_type'].unique())
selected_experience_level = right_column2.selectbox("Select Experience Level", df['experience_level'].unique())
selected_remote_ratio = left_column2.selectbox("Select Remote Working Ratio", df['remote_ratio'].unique())
selected_company_size = right_column2.selectbox("Select Company Size", df['company_size'].unique())
selected_job_title = st.selectbox("Select a Job Tittle", df['job_title'].unique())
expectation_salary = st.number_input("Enter your expectation salary")
#email = st.text_input("Enter your email address")
#st.write("Name:", name)
job_title_index = np.where(df['job_title'].unique() == selected_job_title)[0][0]
employment_type_index = np.where(df['employment_type'].unique() == selected_employment_type)[0][0]
experience_level_index = np.where(df['experience_level'].unique() == selected_experience_level)[0][0]
remote_ratio_index = np.where(df['remote_ratio'].unique() == selected_remote_ratio)[0][0]
company_size_index = np.where(df['company_size'].unique() == selected_company_size)[0][0]
#st.write("selected_job_title:", job_title_index)

################mconvert deal#########################
#predict 1 if user expect salary is able to achived base on the condition input,else 0
df['salary_in_usd'] = df['salary_in_usd'].apply(lambda x: 1 if x >= expectation_salary else 0)
label_encoder = LabelEncoder()
df['experience_level'] = label_encoder.fit_transform(df['experience_level'])
df['employment_type'] = label_encoder.fit_transform(df['employment_type'])
df['job_title'] = label_encoder.fit_transform(df['job_title'])
df['remote_ratio'] = label_encoder.fit_transform(df['remote_ratio'])
df['company_size'] = label_encoder.fit_transform(df['company_size'])


################train test data#########################
X = df[['experience_level', 'employment_type','job_title','remote_ratio', 'company_size']]
y = df['salary_in_usd']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# check data converted to code by labelencoder or not
#st.write(y_train)
#st.write(df['job_title'])

# Create a Decision Tree classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

st.markdown('---')

# Create an input sample with the selected values
input_sample = [job_title_index, employment_type_index, experience_level_index, remote_ratio_index, company_size_index]

# Convert the input sample into a DataFrame with a single row
input_df = pd.DataFrame([input_sample], columns=X.columns)

# Make the prediction
prediction = clf.predict(input_df)

# Print the predicted result
st.subheader("Decision Tree")
#st.write("The predicted salary is:", prediction[0])
if prediction[0] == 1:
    st.write("Congratulations! Your expected salary of", expectation_salary, "$ can be achieved based on the machine learning model.")
else:
    st.write("Sorry, it seems that your expected salary of", expectation_salary, "may not be achievable based on the machine learning model.")

#confusion
# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate precision, recall, F-score, and accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f_score = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Display the results
st.write("Confusion Matrix:")
st.write(cm)
st.write("Precision:", precision)
st.write("Recall:", recall)
st.write("F-Score:", f_score)
st.write("Accuracy:", accuracy)

st.markdown('---')
############################################

# Create the Random Forest classifier
classifier = RandomForestClassifier()

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred_rf = classifier.predict(X_test)

# Create an input sample with the selected values
input_sample = [job_title_index, employment_type_index, experience_level_index, remote_ratio_index, company_size_index]

# Convert the input sample into a DataFrame with a single row
input_df = pd.DataFrame([input_sample], columns=X.columns)

# Make the prediction
prediction_rf = classifier.predict(input_df)

# Print the predicted result
st.subheader("Random Forest")
#st.write("The predicted salary is:", prediction[0])
if prediction_rf[0] == 1:
    st.write("Congratulations! Your expected salary of", expectation_salary, "$ can be achieved based on the machine learning model.")
else:
    st.write("Sorry, it seems that your expected salary of", expectation_salary, "may not be achievable based on the machine learning model.")

#confusion
# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)

# Calculate precision, recall, F-score, and accuracy
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
f_score = f1_score(y_test, y_pred_rf)
accuracy = accuracy_score(y_test, y_pred_rf)

# Display the results
st.write("Confusion Matrix:")
st.write(cm)
st.write("Precision:", precision)
st.write("Recall:", recall)
st.write("F-Score:", f_score)
st.write("Accuracy:", accuracy)

hide_st_style = """
<style>
#MainMenu {display: none;}
footer {display: none;}
header {display: none;}
</style>
"""

st.markdown(hide_st_style, unsafe_allow_html=True)
