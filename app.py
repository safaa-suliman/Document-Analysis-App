import streamlit as st
import os
import pandas as pd
import fitz  # PyMuPDF for PDF processing
import shutil  # For clearing temporary files
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans


# Set page configuration
st.set_page_config(page_title="Document Analysis Webpage", page_icon="📄", layout="wide")
st.subheader("Hi, This is a web for analyzing documents :wave:")
st.title("A Data Analyst From Sudan")
st.write("I am passionate about Data Science")
st.write("[My GitHub >](https://github.com/safa-suliman)")

# Define the function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error processing {pdf_path}: {e}")
        return ""

def clear_temp_folder(folder="temp"):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

# Function to perform text analysis
def analyze_texts(pdf_texts, top_n):
    # Combine all text for global word count
    all_text = " ".join([doc["text"] for doc in pdf_texts])

    # Preprocess and tokenize
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(re.sub(r'\W+', ' ', all_text.lower()))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    # Count word frequencies
    word_counts = Counter(filtered_words)
    top_words = word_counts.most_common(top_n)

    return top_words, word_counts

# Topic Modeling with LDA (scikit-learn)
def topic_modeling(texts, num_topics=3):
    vectorizer = CountVectorizer(stop_words="english")
    doc_term_matrix = vectorizer.fit_transform(texts)
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    return topics

# Clustering using KMeans
def clustering(pdf_texts, num_clusters=3):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([doc["text"] for doc in pdf_texts])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(tfidf_matrix)
    return kmeans.labels_

# Streamlit App
st.title("📂 Document Analysis - Enhanced Features")

# File uploader for multiple PDFs
uploaded_files = st.file_uploader("Upload multiple PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Clear old temporary files
    clear_temp_folder()

    pdf_texts = []

    for uploaded_file in uploaded_files:
        # Save uploaded file to a temporary path
        pdf_path = os.path.join("temp", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Check file extension and process PDFs
        if uploaded_file.name.endswith(".pdf"):
            # Extract text from the PDF
            text = extract_text_from_pdf(pdf_path)
            if text.strip():  # Check if extracted text is not empty
                pdf_texts.append({"filename": uploaded_file.name, "text": text})
            else:
                st.warning(f"No text extracted from {uploaded_file.name}.")
        else:
            st.warning(f"Skipping non-PDF file: {uploaded_file.name}")

    # Check if pdf_texts contains data
    if not pdf_texts:
        st.error("No text could be extracted from the uploaded PDFs. Please check the files.")
    else:
        # Convert to a DataFrame
        pdf_df = pd.DataFrame(pdf_texts)

        # Display the DataFrame
        st.write("### Extracted Data:")
        st.dataframe(pdf_df)

        # Option to download the DataFrame as a CSV
        csv_data = pdf_df.to_csv(index=False)
        st.download_button(label="Download CSV", data=csv_data, file_name="extracted_texts.csv", mime="text/csv")

        # Input for number of top words
        top_n = st.slider("Select the number of top words to display", min_value=1, max_value=20, value=10)

        if st.button("Analyze Texts"):
            # Perform analysis
            top_words, word_counts = analyze_texts(pdf_texts, top_n)

            # Display top words in a table
            st.write("### Top Words Across All Documents:")
            st.table(pd.DataFrame(top_words, columns=["Word", "Frequency"]))

            # Input for specific word analysis
            specific_word = st.text_input("Enter a word to analyze its frequency:")
            if specific_word:
                specific_word_count = word_counts.get(specific_word.lower(), 0)
                st.write(f"The word **'{specific_word}'** appears **{specific_word_count}** times.")

        # Perform Topic Modeling
        num_topics = st.slider("Select the Number of Topics:", 2, 10, 3)
        topics = topic_modeling([doc["text"] for doc in pdf_texts], num_topics=num_topics)
        st.write("### Topic Modeling Results:")
        for topic in topics:
            st.write(topic)

        # Perform Clustering
        num_clusters = st.slider("Select the Number of Clusters:", 2, 10, 3)
        clusters = clustering(pdf_texts, num_clusters=num_clusters)
        pdf_df["Cluster"] = clusters
        st.write("### Clustered Documents:")
        st.dataframe(pdf_df)

else:
    st.info("Please upload multiple PDF files.")
