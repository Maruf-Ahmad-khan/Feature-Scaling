import streamlit as st
import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# ✅ Download NLTK Resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# ✅ Load dataset (Update path)
file_path = r"C:\Users\mk744\OneDrive - Poornima University\Desktop\Feature Scaling\Basic_statistics\Email_spam_data.csv"
df = pd.read_csv(file_path)

# ✅ Handle missing values
df.dropna(inplace=True)  # Remove NaN rows
df['message'] = df['message'].astype(str)  # Convert all messages to string
df['message'] = df['message'].replace('', 'NoText')  # Replace empty messages

# ✅ Convert labels (assuming the column name is 'label')
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df.dropna(subset=['label'], inplace=True)  # Ensure no NaN labels
df['label'] = df['label'].astype(int)  # Convert labels to integer

# ✅ Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# ✅ Function for text preprocessing
def preprocess_text(text):
    """Preprocess text: Lowercase, remove punctuation, tokenize, remove stopwords, apply lemmatization & stemming."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))  # Lowercase & remove punctuation
    words = word_tokenize(text)  # Tokenization
    words = [stemmer.stem(lemmatizer.lemmatize(word)) for word in words if word not in stop_words]  # Remove stopwords & lemmatize/stem
    return " ".join(words) if words else "NoText"  # Ensure no empty text

# ✅ Apply text preprocessing
df['cleaned_message'] = df['message'].apply(preprocess_text)

# ✅ Improve TF-IDF to prevent errors
vectorizer = TfidfVectorizer(
    max_features=3000,  # Limit vocabulary size
    max_df=0.8,  # Ignore words in more than 80% of messages
    min_df=1  # Keep words appearing in at least 1 message
)
X = vectorizer.fit_transform(df['cleaned_message'])

# ✅ Define target variable
y = df['label']

# ✅ Perform stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # Ensures class balance
)

# ✅ Train a Multinomial Naïve Bayes classifier with regularization
model = MultinomialNB(alpha=0.5)  # Reduce over-reliance on specific words
model.fit(X_train, y_train)

# ✅ Predict & evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

# ✅ Convert classification report to a DataFrame
report_df = pd.DataFrame({
    "Class": ["Ham (0)", "Spam (1)"],
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1,
    "Support": support
})

# -----------------------------------------------
# 🌟 STREAMLIT UI
# -----------------------------------------------

st.title("📩 Spam Email Classifier App")

st.sidebar.header("Navigation")
option = st.sidebar.radio("Select Option", ["Dataset Overview", "EDA & Visualization", "Spam Prediction"])

# ✅ 1. Dataset Overview
if option == "Dataset Overview":
    st.header("📌 Dataset Summary")
    st.write(df.head())

    # Spam vs. Ham distribution
    st.subheader("Spam vs. Ham Count")
    fig, ax = plt.subplots()
    sns.countplot(x=df['label'], palette='viridis', ax=ax)
    ax.set_xticklabels(["Ham (0)", "Spam (1)"])
    st.pyplot(fig)

    # Message Length Distribution
    df['message_length'] = df['message'].apply(len)
    st.subheader("Message Length Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['message_length'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

# ✅ 2. EDA & Visualization
elif option == "EDA & Visualization":
    st.header("📊 Exploratory Data Analysis")

    # ✅ Handle empty word clouds safely
    spam_messages = df[df['label'] == 1]['message']
    ham_messages = df[df['label'] == 0]['message']
    
    spam_words = " ".join(spam_messages) if not spam_messages.empty else "NoSpamMessages"
    ham_words = " ".join(ham_messages) if not ham_messages.empty else "NoHamMessages"

    # WordCloud for Spam
    spam_wc = WordCloud(width=800, height=400, background_color="black").generate(spam_words)
    st.subheader("Most Common Words in Spam Messages")
    fig, ax = plt.subplots()
    ax.imshow(spam_wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    # WordCloud for Ham
    ham_wc = WordCloud(width=800, height=400, background_color="white").generate(ham_words)
    st.subheader("Most Common Words in Ham Messages")
    fig, ax = plt.subplots()
    ax.imshow(ham_wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    # Show Model Accuracy & Classification Report in Table
    st.subheader("📌 Model Performance")
    st.write(f"**Accuracy:** {accuracy:.2%}")
    st.dataframe(report_df)  # Show classification report as a table

# ✅ 3. Spam Prediction
elif option == "Spam Prediction":
    st.header("🤖 Spam Email Prediction")
    
    user_input = st.text_area("Enter an email message below:")
    
    if st.button("Predict"):
        if user_input.strip():
            # Preprocess input message
            cleaned_input = preprocess_text(user_input)
            transformed_input = vectorizer.transform([cleaned_input])
            
            # Predict
            prediction = model.predict(transformed_input)[0]
            result = "✅ Ham (Not Spam)" if prediction == 0 else "🚨 Spam Detected!"
            
            # Show result
            st.subheader("Prediction Result:")
            st.write(f"**{result}**")
        else:
            st.warning("⚠️ Please enter some text to classify.")
