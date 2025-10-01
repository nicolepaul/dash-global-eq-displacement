# Use lightweight Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . .

# Expose port
EXPOSE 8050

# Run app with Gunicorn (better for production)
CMD ["gunicorn", "app:server", "-b", "0.0.0.0:8050"]