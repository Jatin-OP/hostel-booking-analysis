# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from langchain.document_loaders import DataFrameLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.chains import RetrievalQA
# from langchain.memory import ConversationBufferMemory
# from langchain.schema import AIMessage
# from langchain.prompts import PromptTemplate
# from langchain_community.llms import Ollama
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# # Load Data
# df = pd.read_csv("D:\Systematic\hotel_bookings.csv")

# # Data Cleaning
# df.fillna({"children": 0, "country": "Unknown", "agent": 0, "company": 0}, inplace=True)
# df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"])
# df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
# df["revenue"] = df["adr"] * df["total_nights"]
# df = df[(df["adr"] > 0) & (df["total_nights"] > 0)]

# df["text"] = df.apply(
#     lambda row: f"Hotel: {row['hotel']}, Country: {row['country']}, Revenue: {row['revenue']}, "
#                  f"Lead Time: {row['lead_time']}, Status: {'Canceled' if row['is_canceled'] else 'Not Canceled'}",
#     axis=1
# )
# df = df.tail(10000)

# df['reservation_status_date'] = df['reservation_status_date'].astype(str)

# def revenue_trends():
#     df["arrival_date"] = pd.to_datetime(
#         df["arrival_date_year"].astype(str) + "-" +
#         df["arrival_date_month"] + "-" +
#         df["arrival_date_day_of_month"].astype(str)
#     )
#     revenue_trend = df.groupby("arrival_date")["revenue"].sum()
#     fig, ax = plt.subplots(figsize=(12, 6))
#     revenue_trend.plot(ax=ax)
#     ax.set_title("Revenue Trends Over Time")
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Revenue")
#     st.pyplot(fig)

# def cancellation_rate():
#     total_bookings = len(df)
#     cancellations = df[df["is_canceled"] == 1].shape[0]
#     cancellation_rate = (cancellations / total_bookings) * 100
#     st.write(f"## Cancellation Rate: {cancellation_rate:.2f}%")

# def geographical_distribution():
#     country_distribution = df["country"].value_counts().head(10)
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.barplot(x=country_distribution.index, y=country_distribution.values, ax=ax)
#     ax.set_title("Top 10 Countries by Bookings")
#     ax.set_xlabel("Country")
#     ax.set_ylabel("Number of Bookings")
#     st.pyplot(fig)

# def lead_time_distribution():
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.histplot(df["lead_time"], bins=50, kde=True, ax=ax)
#     ax.set_title("Booking Lead Time Distribution")
#     ax.set_xlabel("Lead Time (Days)")
#     ax.set_ylabel("Frequency")
#     st.pyplot(fig)

# # Embeddings and Retrieval Setup
# loader = DataFrameLoader(df, page_content_column="text")
# documents = loader.load()
# embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
# vector_store = Chroma.from_documents(documents, embeddings)
# retriever = vector_store.as_retriever()

# llm = Ollama(model="mistral", temperature=0.7)

# prompt_template = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are a helpful and friendly Hotel booking assistant.
# You provide information about Hotel bookings, cancellations, and revenue in a polite and easy-to-understand way.
# If you don't know the answer, politely say so. Keep responses concise and clear.

# Context: {context}
# Question: {question}
# Answer:
# """.strip(),
# )

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# documents_sample = [
#     "Hotel bookings are managed through our online portal.",
#     "Cancellations can be made up to 24 hours before check-in.",
#     "Revenue is calculated based on bookings and cancellations.",
# ]

# huggingface_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
# vectorstore = Chroma.from_texts(documents_sample, huggingface_embeddings)
# retriever = vectorstore.as_retriever()

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": prompt_template},
# )

# # Streamlit UI
# st.title("Hotel Booking Analysis & Assistant")

# st.sidebar.title("Navigation")
# option = st.sidebar.radio("Select an option", ["Data Analysis", "Chatbot"])

# if option == "Data Analysis":
#     st.subheader("Revenue Trends")
#     revenue_trends()
#     st.subheader("Cancellation Rate")
#     cancellation_rate()
#     st.subheader("Geographical Distribution")
#     geographical_distribution()
#     st.subheader("Lead Time Distribution")
#     lead_time_distribution()

# elif option == "Chatbot":
#     st.write("### Chat with the Hotel Booking Assistant")
#     user_input = st.text_input("You:")
#     if user_input:
#         response = qa_chain({"query": user_input})
#         ai_message = AIMessage(content=response["result"])
#         memory.save_context({"input": user_input}, {"output": str(ai_message.content)})
#         st.write(f"**Assistant:** {ai_message.content}")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load Data
df = pd.read_csv(r"D:\Systematic\buyogo\hotel_bookings.csv")

# Data Cleaning
df.fillna({"children": 0, "country": "Unknown", "agent": 0, "company": 0}, inplace=True)
df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"])
df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
df["revenue"] = df["adr"] * df["total_nights"]
df = df[(df["adr"] > 0) & (df["total_nights"] > 0)]

df["text"] = df.apply(
    lambda row: f"Hotel: {row['hotel']}, Country: {row['country']}, Revenue: {row['revenue']}, "
                 f"Lead Time: {row['lead_time']}, Status: {'Canceled' if row['is_canceled'] else 'Not Canceled'}",
    axis=1
)
df = df.tail(10000)
df['reservation_status_date'] = df['reservation_status_date'].astype(str)

# Data Analysis Functions
def revenue_trends():
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" +
        df["arrival_date_month"] + "-" +
        df["arrival_date_day_of_month"].astype(str)
    )
    revenue_trend = df.groupby("arrival_date")["revenue"].sum()
    fig, ax = plt.subplots(figsize=(12, 6))
    revenue_trend.plot(ax=ax)
    ax.set_title("Revenue Trends Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    st.pyplot(fig)

def cancellation_rate():
    total_bookings = len(df)
    cancellations = df[df["is_canceled"] == 1].shape[0]
    cancellation_rate = (cancellations / total_bookings) * 100
    st.write(f"## Cancellation Rate: {cancellation_rate:.2f}%")

def geographical_distribution():
    country_distribution = df["country"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=country_distribution.index, y=country_distribution.values, ax=ax)
    ax.set_title("Top 10 Countries by Bookings")
    ax.set_xlabel("Country")
    ax.set_ylabel("Number of Bookings")
    st.pyplot(fig)

def lead_time_distribution():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["lead_time"], bins=50, kde=True, ax=ax)
    ax.set_title("Booking Lead Time Distribution")
    ax.set_xlabel("Lead Time (Days)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# Embeddings and Retrieval Setup
loader = DataFrameLoader(df, page_content_column="text")
documents = loader.load()
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
vector_store = Chroma.from_documents(documents, embeddings)
retriever = vector_store.as_retriever()

llm = Ollama(model="mistral", temperature=0.7)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful and friendly Hotel booking assistant.
You provide information about Hotel bookings, cancellations, and revenue in a polite and easy-to-understand way.
If you don't know the answer, politely say so. Keep responses concise and clear.

Context: {context}
Question: {question}
Answer:
""".strip(),
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

documents_sample = [
    "Hotel bookings are managed through our online portal.",
    "Cancellations can be made up to 24 hours before check-in.",
    "Revenue is calculated based on bookings and cancellations.",
]

huggingface_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma.from_texts(documents_sample, huggingface_embeddings)
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template},
)

# Streamlit UI
st.title("Hotel Booking Analysis & Assistant")

st.sidebar.title("Navigation")
option = st.sidebar.radio("Select an option", ["Data Analysis", "Chatbot"])

if option == "Data Analysis":
    st.subheader("Revenue Trends")
    revenue_trends()
    st.subheader("Cancellation Rate")
    cancellation_rate()
    st.subheader("Geographical Distribution")
    geographical_distribution()
    st.subheader("Lead Time Distribution")
    lead_time_distribution()

elif option == "Chatbot":
    st.write("### Chat with the Hotel Booking Assistant")

    # Retrieve and display past chat history
    if "chat_history" in memory.load_memory_variables({}):
        for message in memory.load_memory_variables({})["chat_history"]:
            if isinstance(message, AIMessage):
                st.write(f"**Assistant:** {message.content}")
            elif isinstance(message, HumanMessage):
                st.write(f"**You:** {message.content}")

    user_input = st.text_input("You:")
    
    if user_input:
        response = qa_chain({"query": user_input})
        ai_message = AIMessage(content=response["result"])

        # Save chat history
        memory.save_context({"input": user_input}, {"output": ai_message.content})

        # Display user and assistant messages
        st.write(f"**You:** {user_input}")
        st.write(f"**Assistant:** {ai_message.content}")
