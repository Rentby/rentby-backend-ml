# Use the official Python image from the Docker Hub
FROM python:3.11

# Set environment variables
ENV PIP_DEFAULT_TIMEOUT=100
ENV VIRTUAL_ENV=/home/appuser/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Create a non-root user and set permissions
RUN useradd -ms /bin/bash appuser

# Set working directory
WORKDIR /home/appuser/app

# Switch to root to set up permissions
USER root

# Ensure the app directory has the correct permissions
RUN mkdir -p /home/appuser/app && chown -R appuser:appuser /home/appuser/app

# Switch to the non-root user
USER appuser

# Copy the requirements file into the container
COPY --chown=appuser:appuser requirements.txt .

# Install virtualenv and create a virtual environment
RUN python -m venv $VIRTUAL_ENV

# Install the dependencies with retry mechanism
RUN pip install --no-cache-dir -r requirements.txt || \
    (sleep 5 && pip install --no-cache-dir -r requirements.txt) || \
    (sleep 10 && pip install --no-cache-dir -r requirements.txt)

# Copy the rest of the application code
COPY --chown=appuser:appuser . .

# Expose the port for the application
EXPOSE 8080

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
