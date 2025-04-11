FROM paddlecloud/paddleocr:2.6-cpu-latest

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    flask \
    requests \
    opencv-python \
    paddleocr 

# Copy application code
COPY . /app/

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "app.py"]