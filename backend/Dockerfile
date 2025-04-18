# Use a Python 3.9 base image
FROM python:3.9

# Create a user to run the app (you can also skip if you prefer to use the root user)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY --chown=user ./requirements.txt requirements.txt

# Install the dependencies listed in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application code into the container
COPY --chown=user . /app

# Set the environment variable to specify the port for Flask (default Flask port is 5000, but Hugging Face uses 7860)
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=7860

# Expose the port
EXPOSE 7860

# Define the command to run your Flask app (Flask will automatically look for app.py and use it)
CMD ["flask", "run", "--host=0.0.0.0", "--port=7860"]
