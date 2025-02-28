import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Travel Chatbot 🌍", page_icon="🌍", layout="centered")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

def load_data(file_path):
    try:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return None
        df = pd.read_csv(file_path)
        if df.empty:
            st.error("The dataset is empty.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def preprocess_data(df):
    df = df.fillna("")
    df['Destination'] = df['Destination'].str.lower()
    df['Itinerary'] = df['Itinerary'].str.lower()
    return df

def create_vectorizer(df):
    vectorizer = TfidfVectorizer()
    place_vectors = vectorizer.fit_transform(df['Destination'] + " " + df['Itinerary'])
    return vectorizer, place_vectors

def find_best_match(user_query, vectorizer, place_vectors, df):
    query_vector = vectorizer.transform([user_query.lower()])
    similarities = cosine_similarity(query_vector, place_vectors).flatten()
    best_index = similarities.argmax()
    best_score = similarities[best_index]
    if best_score > 0.3:
        return df.iloc[best_index]['Itinerary']
    else:
        return None

def configure_generative_model(api_key):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Error configuring AI model: {e}")
        return None

def refine_answer(generative_model, user_query, matched_answer):
    try:
        context = """
        You are a travel and tourism chatbot. Improve the following recommendation with detailed insights,
        including best time to visit, activities, and estimated costs.
        """
        prompt = f"{context}\n\nUser Query: {user_query}\nMatched Answer: {matched_answer}\nRefined Answer:"
        response = generative_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error refining answer: {e}"
    
    
def travel_chatbot(df, vectorizer, place_vectors, generative_model):
    st.title("Travel Chatbot 🌍")
    st.write("Ask me about travel destinations!")

    st.markdown("### Conversation History")
    for message in st.session_state.conversation:
        role, content = message["role"], message["content"]

        # Update colors for better visibility
        if role == "User":
            color = "#1e90ff"  # Blue for user messages
            text_color = "#ffffff"  # White text
        else:
            color = "#333333"  # Dark gray for bot responses
            text_color = "#f8f8f8"  # Light gray text

        st.markdown(
            f"<div style='background-color: {color}; color: {text_color}; padding: 10px; "
            f"border-radius: 10px; margin: 5px 0;'>"
            f"<strong>{role}:</strong> {content}</div>", 
            unsafe_allow_html=True
        )

    user_query = st.text_input("User:", placeholder="Ask about a travel destination...", key="user_input")

    if user_query:
        st.session_state.conversation.append({"role": "User", "content": user_query})
        matched_answer = find_best_match(user_query, vectorizer, place_vectors, df)

        if matched_answer:
            with st.spinner("Refining answer..."):
                refined_answer = refine_answer(generative_model, user_query, matched_answer)
                st.session_state.conversation.append({"role": "Bot", "content": refined_answer})

                # Updated bot response style
                st.markdown(
                    f"<div style='background-color: #333333; color: #f8f8f8; padding: 10px; "
                    f"border-radius: 10px;'>"
                    f"<strong>Bot:</strong> {refined_answer}</div>", 
                    unsafe_allow_html=True
                )
        else:
            try:
                context = """
                You are a travel chatbot. Suggest a travel destination with details including ideal visit time,
                key attractions, and estimated cost.
                """
                prompt = f"{context}\n\nUser: {user_query}\nBot:"
                response = generative_model.generate_content(prompt)
                st.session_state.conversation.append({"role": "Bot", "content": response.text})

                # Updated bot response style
                st.markdown(
                    f"<div style='background-color: #333333; color: #f8f8f8; padding: 10px; "
                    f"border-radius: 10px;'>"
                    f"<strong>Bot:</strong> {response.text}</div>", 
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error generating response: {e}")


def main():
    file_path = "travel_toursim_data.csv"
    df = load_data(file_path)
    if df is None:
        return
    df = preprocess_data(df)
    vectorizer, place_vectors = create_vectorizer(df)
    API_KEY = "your_API_key"
    if not API_KEY:
        st.error("API key not found.")
        return
    generative_model = configure_generative_model(API_KEY)
    if generative_model is None:
        return
    travel_chatbot(df, vectorizer, place_vectors, generative_model)

if __name__ == "__main__":
    main()
