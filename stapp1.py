import streamlit as st
import langchain as lc
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, classification_report, confusion_matrix
import streamlit as st
import openai
from langchain import PromptTemplate, LLMChain
from langchain_community.chat_models import ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community import embeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os
import time
import pandas as pd
import numpy as np
from langchain_community.vectorstores import Chroma

import textwrap
import streamlit as st

st. set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
# Suppress warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load data
df = pd.read_csv('Enter your .csv file path')
st.sidebar.markdown("<h1 style='text-align: left; color: red ;'>Telco AI Solutions</h1>", unsafe_allow_html=True)
with st.sidebar:
    groq_api_key = 'Enter your groq key'

    llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name='mixtral-8x7b-32768'
    )

    rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    rag_chain = (
    {"context":  RunnablePassthrough(), "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
    )

def process_question(user_question):
    start_time = time.time()

    # Directly using the user's question as input for rag_chain.invoke
    response = rag_chain.invoke(user_question)

    # Measure the response time
    end_time = time.time()
    response_time = f"Response time: {end_time - start_time:.2f} seconds."

    # Combine the response and the response time into a single string
    full_response = f"{response}\n\n{response_time}"

    return full_response

# Setup the Streamlit interface
# st.title("GROQ CHAT")
user_question = st.sidebar.text_input("Type your question here...")
if user_question:
    full_response = process_question(user_question)
    st.sidebar.markdown(full_response)

col1, col2 = st.columns([1, 1])
with col1:
    st.title('Telco Customer Churn Analysis')

    # Display data shape
    st.write(f"Data Shape: {df.shape}")

    # Display missing values matrix
    st.write("Missing Values Matrix:")
    msno.matrix(df)
    st.pyplot()

    # Drop customerID column
    df = df.drop(['customerID'], axis=1)

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')

    # Drop rows with tenure == 0
    df = df.drop(df[df['tenure'] == 0].index)

    # Fill null values in TotalCharges with mean
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

    # Map SeniorCitizen to categorical
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Pie charts for Gender and Churn
    g_labels = ['Male', 'Female']
    c_labels = ['No', 'Yes']

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])

    fig.add_trace(go.Pie(labels=g_labels, values=df['gender'].value_counts(), name="Gender"), 1, 1)
    fig.add_trace(go.Pie(labels=c_labels, values=df['Churn'].value_counts(), name="Churn"), 1, 2)

    fig.update_traces(hole=.4, hoverinfo="label+percent+name", textfont_size=16)

    fig.update_layout(
        title_text="Gender and Churn Distributions",
        annotations=[dict(text='Gender', x=0.16, y=0.5, font_size=20, showarrow=False),
                     dict(text='Churn', x=0.84, y=0.5, font_size=20, showarrow=False)])

    st.plotly_chart(fig)

    # Distribution of monthly charges by churn (KDE Plot)
    st.write("Distribution of monthly charges by churn:")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(df.MonthlyCharges[(df["Churn"] == 'No')], color="Red", shade=True, ax=ax)
    sns.kdeplot(df.MonthlyCharges[(df["Churn"] == 'Yes')], color="Blue", shade=True, ax=ax)
    ax.legend(["Not Churn", "Churn"], loc='upper right')
    ax.set_ylabel('Density')
    ax.set_xlabel('Monthly Charges')
    ax.set_title('Distribution of monthly charges by churn')
    st.pyplot(fig, dpi=100)

    # Distribution of total charges by churn (KDE Plot)
    st.write("Distribution of total charges by churn:")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(df.TotalCharges[(df["Churn"] == 'No')], color="Gold", shade=True, ax=ax)
    sns.kdeplot(df.TotalCharges[(df["Churn"] == 'Yes')], color="Green", shade=True, ax=ax)
    ax.legend(["Not Churn", "Churn"], loc='upper right')
    ax.set_ylabel('Density')
    ax.set_xlabel('Total Charges')
    ax.set_title('Distribution of total charges by churn')
    st.pyplot(fig, dpi=100)

    # Customer contract distribution
    st.write("Customer contract distribution:")
    fig = px.histogram(df, x="Churn", color="Contract", barmode="group", title="<b>Customer contract distribution<b>")
    st.plotly_chart(fig)

    # Payment Method Distribution
    st.write("Payment Method Distribution:")
    labels = df['PaymentMethod'].unique()
    values = df['PaymentMethod'].value_counts()
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(title_text="<b>Payment Method Distribution</b>")
    st.plotly_chart(fig)

    # Customer Payment Method distribution w.r.t. Churn
    st.write("Customer Payment Method distribution w.r.t. Churn:")
    fig = px.histogram(df, x="Churn", color="PaymentMethod",
                       title="<b>Customer Payment Method distribution w.r.t. Churn</b>")
    st.plotly_chart(fig)

    # Dependents distribution
    st.write("Dependents distribution:")
    color_map = {"Yes": "#FF97FF", "No": "#AB63FA"}
    fig = px.histogram(df, x="Churn", color="Dependents", barmode="group", title="<b>Dependents distribution</b>",
                       color_discrete_map=color_map)
    st.plotly_chart(fig)

    # Partners distribution
    st.write("Churn distribution w.r.t. Partners:")
    color_map = {"Yes": '#FFA15A', "No": '#00CC96'}
    fig = px.histogram(df, x="Churn", color="Partner", barmode="group",
                       title="<b>Churn distribution w.r.t. Partners</b>", color_discrete_map=color_map)
    st.plotly_chart(fig)

    # Senior Citizen distribution
    st.write("Churn distribution w.r.t. Senior Citizen:")
    color_map = {"Yes": '#00CC96', "No": '#B6E880'}
    fig = px.histogram(df, x="Churn", color="SeniorCitizen", title="<b>Churn distribution w.r.t. Senior Citizen</b>",
                       color_discrete_map=color_map)
    st.plotly_chart(fig)

    # Online Security distribution
    st.write("Churn distribution w.r.t. Online Security:")
    color_map = {"Yes": "#FF97FF", "No": "#AB63FA"}
    fig = px.histogram(df, x="Churn", color="OnlineSecurity", barmode="group",
                       title="<b>Churn w.r.t Online Security</b>", color_discrete_map=color_map)
    st.plotly_chart(fig)

    # Paperless Billing distribution
    st.write("Churn distribution w.r.t. Paperless Billing:")
    color_map = {"Yes": '#FFA15A', "No": '#00CC96'}
    fig = px.histogram(df, x="Churn", color="PaperlessBilling",
                       title="<b>Churn distribution w.r.t. Paperless Billing</b>", color_discrete_map=color_map)
    st.plotly_chart(fig)

    # TechSupport distribution
    st.write("Churn distribution w.r.t. TechSupport:")
    fig = px.histogram(df, x="Churn", color="TechSupport", barmode="group",
                       title="<b>Churn distribution w.r.t. TechSupport</b>")
    st.plotly_chart(fig)

    # Phone Service distribution
    st.write("Churn distribution w.r.t. Phone Service:")
    color_map = {"Yes": '#00CC96', "No": '#B6E880'}
    fig = px.histogram(df, x="Churn", color="PhoneService", title="<b>Churn distribution w.r.t. Phone Service</b>",
                       color_discrete_map=color_map)
    st.plotly_chart(fig)

    # Box plot of tenure vs churn
    st.write("Box plot of tenure vs churn:")
    fig = px.box(df, x='Churn', y='tenure')
    fig.update_layout(width=750, height=600, title='<b>Tenure vs Churn</b>')
    st.plotly_chart(fig)

    # Correlation heatmap
    st.write("Correlation Heatmap:")
    plt.figure(figsize=(25, 10))
    corr = df.apply(lambda x: pd.factorize(x)[0]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    ax = sns.heatmap(corr, mask=mask, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, linewidths=.2,
                     cmap='coolwarm', vmin=-1, vmax=1)
    st.pyplot()

    # Feature Encoding
    st.write("Feature Encoding:")


    def object_to_int(dataframe_series):
        if dataframe_series.dtype == 'object':
            dataframe_series = LabelEncoder().fit_transform(dataframe_series)
        return dataframe_series


    df_encoded = df.apply(lambda x: object_to_int(x))
    # st.write(df_encoded.head())

    # Train-test split
    X = df_encoded.drop(columns=['Churn'])
    y = df_encoded['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40, stratify=y)

    # Standardize numerical columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Modeling
    st.write("Modeling:")
    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=11),
        "Support Vector Machine": SVC(random_state=1),
        "Random Forest": RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=-1,
                                                random_state=50, max_features='sqrt', max_leaf_nodes=30),
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"{name} accuracy: {accuracy}")
        st.write(classification_report(y_test, y_pred))

        if name == "Random Forest":
            # Confusion Matrix for Random Forest
            plt.figure(figsize=(4, 3))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", linecolor="k", linewidths=3)
            plt.title("RANDOM FOREST CONFUSION MATRIX")
            st.pyplot()

            # ROC Curve for Random Forest
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr, tpr, label='Random Forest', color="r")
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Random Forest ROC Curve')
            st.pyplot()

        if name == "Logistic Regression":
            # Confusion Matrix for Logistic Regression
            plt.figure(figsize=(4, 3))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", linecolor="k", linewidths=3)
            plt.title("LOGISTIC REGRESSION CONFUSION MATRIX")
            st.pyplot()

            # ROC Curve for Logistic Regression
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr, tpr, label='Logistic Regression', color="r")
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Logistic Regression ROC Curve')
            st.pyplot()

        if name == "Gradient Boosting":
            # Confusion Matrix for Gradient Boosting
            plt.figure(figsize=(4, 3))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", linecolor="k", linewidths=3)
            plt.title("GRADIENT BOOSTING CONFUSION MATRIX")
            st.pyplot()

with col2:
    st.write("""
    ## About Telco AI Solutions

    Telco AI Solutions is a pioneer in leveraging artificial intelligence and data analytics 
    to optimize telecommunications services. With a focus on customer satisfaction and 
    retention, we employ cutting-edge technologies to drive business insights and improve 
    operational efficiency. 🚀

    ### Our Mission:
    - **Customer Centricity:** We prioritize the needs and preferences of our customers. 💡
    - **Innovation:** We continuously innovate to deliver superior services and experiences. 🌟
    - **Data-driven Decision Making:** We harness the power of data to make informed decisions. 📊

    ### Contact Us:
    Have questions or inquiries? Contact our team at [info@telcoai.com](mailto:info@telcoai.com). 📧

    All copyrights to Telco AI Solutions©️ 
""")