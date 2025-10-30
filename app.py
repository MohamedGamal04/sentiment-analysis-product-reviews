import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from num2words import num2words
from nltk.corpus import wordnet
import contractions
from nltk.corpus import stopwords
import re 
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

#Preprocess
def preprocess_text(text):
    text = contractions.fix(text)
    text = text.replace('.', ' . ')
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)
    text = "".join(num2words(int(word)) if word.isdigit() else word for word in text)
    word_tokens = word_tokenize(text)
    text = [w for w in word_tokens if not w in stop_words]
    tagged = nltk.tag.pos_tag(text)
    lemmatized_words = []

    for word, tag in tagged:
        wordnet_pos = get_wordnet_pos(tag) or wordnet.NOUN
        lemmatized_words.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    return ' '.join(lemmatized_words)

# Streamlit App
import pickle
@st.cache_resource
def load_model():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, clf = load_model()
st.set_page_config(page_title="Sentiment Analysis", page_icon="🎭", layout="wide")

st.title("🎭 Sentiment Analysis on Reviews")
st.markdown("### Analyze the sentiment of movie reviews using Machine Learning")

# Sidebar for model info
with st.sidebar:
    st.header("Model Information")
    st.info("""
    **Model:** Logistic Regression
    
    **Accuracy:** ~90%
    
    **Dataset:** IMDB 50k Movie Reviews
    
    **Classes:**
    - Positive (1)
    - Negative (0)
    """)
    
    st.header("About")
    st.markdown("""
    This app uses a trained Logistic Regression model with TF-IDF vectorization 
    to predict sentiment from text reviews.
    
    The preprocessing includes:
    - Contraction expansion
    - HTML tag removal
    - Stopword removal
    - Lemmatization
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Model Insights"])

with tab1:
    st.header("Single Review Analysis")
    
    # Text input
    user_input = st.text_area(
        "Enter your review:",
        height=150,
        placeholder="Type or paste your review here..."
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        predict_button = st.button("🔍 Analyze Sentiment", type="primary")
    
    if predict_button and user_input:
        with st.spinner("Processing..."):
            # Preprocess
            processed_text = preprocess_text(user_input)
            
            # For demo purposes - you'll need to load your actual trained model
            # vectorizer and clf should be loaded from saved files
            st.success("✅ Analysis Complete!")
            
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                prediction = clf.predict(vectorizer.transform([processed_text]))[0]
                confidence = clf.predict_proba(vectorizer.transform([processed_text]))[0]
                st.metric("Sentiment", "Positive" if prediction == 1 else "Negative", delta= "High Confidence" if confidence.max() > 0.75 else "Low Confidence")
                print (confidence)
            with col2:
                st.metric("Confidence Score", f"{confidence.max()*100:.2f}%")
            
            # Show processed text
            with st.expander("🔎 View Preprocessed Text"):
                st.text(processed_text)
    
    elif predict_button and not user_input:
        st.warning("⚠️ Please enter some text to analyze.")

with tab2:
    st.header("Batch Review Analysis")
    
    uploaded_file = st.file_uploader("Upload a CSV file with reviews", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("🚀 Analyze All Reviews") and df['review'].notna().any():
            with st.spinner("Processing batch..."):
                # Process all reviews
                df['processed'] = df['review'].apply(preprocess_text)
                predictions = clf.predict(vectorizer.transform(df['processed']))
                df['predicted_sentiment'] = predictions
                
                st.success("✅ Batch analysis complete!")
                st.dataframe(df)
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Reviews", len(df))
                with col2:
                    st.metric("Positive", len(df[df['predicted_sentiment'] == 1]))
                with col3:
                    st.metric("Negative", len(df[df['predicted_sentiment'] == 0]))
        else:
            st.info("ℹ️ Please upload a CSV file with a 'review' column to analyze.")

with tab3:
    st.header("Model Insights & Feature Importance")
    
    st.markdown("""
    ### Top Positive & Negative Features
    These words have the strongest influence on sentiment prediction.
    """)
    
    # Example feature importance (replace with actual from your model)
    top_positive = ['excellent', 'great', 'perfect', 'wonderful', 'brilliant']
    top_negative = ['waste', 'awful', 'bad', 'worst', 'terrible']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("**Top Positive Words**")
        for i, word in enumerate(top_positive, 1):
            st.write(f"{i}. {word}")
    
    with col2:
        st.error("**Top Negative Words**")
        for i, word in enumerate(top_negative, 1):
            st.write(f"{i}. {word}")
    
    st.markdown("---")
    st.info("""
    **To use your trained model:**
    
    ```
    # Save your model and vectorizer:
    import pickle
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    
    # Load in Streamlit:
    @st.cache_resource
    def load_model():
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model
    
    vectorizer, clf = load_model()
    ```
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>"
    "<p>Built with Streamlit | Powered by Logistic Regression & TF-IDF</p>"
    "</div>",
    unsafe_allow_html=True

)

