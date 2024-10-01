from transanalyer import Transanalyer
import matplotlib.pyplot as plt
import streamlit as st

st.title("Video Sentiment Analyzer")
st.write("Enter video link.")

url = st.text_input("Enter a URL")

if st.button("Evaluate"):
    if url:
        st.write(f"Entered URL: {url}")

        analyzer = Transanalyer()
        sentiment_results = analyzer.pipeline(url)

        if sentiment_results:
            st.write("Sentiment Counts:")
        
        fig, ax = plt.subplots()
        categories = list(sentiment_results.keys())
        values = list(sentiment_results.values())
        
        ax.bar(categories, values, color=['green', 'red', 'gray'])
        ax.set_title("Sentiment Analysis Results")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Frequency")
        
        fig.tight_layout()  
        
        st.pyplot(fig)
    else:
        st.write("URL Invalid")