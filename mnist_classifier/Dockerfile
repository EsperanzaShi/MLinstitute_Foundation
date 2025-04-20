FROM python:3.10-slim
WORKDIR /app
COPY mnist_classifier/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY mnist_classifier/ ./
# Copy the trained model checkpoint into the container
COPY mnist_best.pth ./
EXPOSE 8501
CMD ["streamlit", "run", "interactive_front_end.py", "--server.port=8501", "--server.address=0.0.0.0"]