import streamlit as st
import matplotlib.pyplot as plt
from transanalyer import Transanalyer

# Streamlit title and description
st.title("Sentiment Analysis Based on URL-Provided Document")
st.write("Enter a URL linking to a document and click 'Evaluate' to perform sentiment analysis.")

# Input field for URL
url = st.text_input("Enter a URL")

# Evaluate button
if st.button("Evaluate"):
    if url:
        # Display the URL entered by the user
        st.write(f"Entered URL: {url}")

        # Initialize the Transanalyer class and perform sentiment analysis
        analyzer = Transanalyer()
        sentiment_results = analyzer.pipeline(url)

        # Display sentiment counts
        if sentiment_results:
            st.write("Sentiment Counts:")
            # for sentiment, count in sentiment_results.items():
            #     st.write(f"{sentiment.capitalize()}: {count}")

        # Plot the bar chart for sentiment results
        fig, ax = plt.subplots()
        categories = list(sentiment_results.keys())
        values = list(sentiment_results.values())
        ax.bar(categories, values, color=['green', 'red', 'gray'])
        ax.set_title("Sentiment Analysis Results")
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Frequency")
        fig.tight_layout()  # This ensures that the plot is laid out properly so the title shows fully

        # Display the bar chart on the page
        st.pyplot(fig)
    else:
        st.write("Please enter a valid URL.")



















# import streamlit as st
# import numpy as np

# import matplotlib.pyplot as plt
# from transanalyer import Transanalyer

# # Streamlit title and description
# st.title("URL Input and Random Histogram Generator")
# st.write("Enter a URL and click 'Evaluate' to display the URL and generate a random histogram.")

# # Input field for URL
# url = st.text_input("Enter a URL")



# # Evaluate button
# if st.button("Evaluate"):
#     if url:
#         # Display the URL entered by the user
#         st.write(f"Entered URL: {url}")

#         # Generate a random histogram
#         data = np.random.randn(1000)  # Generate random data

#         # Plot the histogram
#         fig, ax = plt.subplots()
#         ax.hist(data, bins=30, alpha=0.7, color='blue')
#         ax.set_title("Random Histogram")
#         ax.set_xlabel("Value")
#         ax.set_ylabel("Frequency")

#         # Display the histogram on the page
#         st.pyplot(fig)
#     else:
#         st.write("Please enter a valid URL.")
