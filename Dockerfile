# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8501 (default port for Streamlit)
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "server.py", "--server.port=8501", "--server.address=0.0.0.0"]

