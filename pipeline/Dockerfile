FROM python:3.11-slim

# Working Directory
WORKDIR /app

# Copy source code to working directory
COPY . /app/

# Install packages from requirements.txt

RUN pip install --no-cache-dir --upgrade pip &&\
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt


EXPOSE 5051
ENV PORT=5051

CMD ["python", "application_flask.py"]