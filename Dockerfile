FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY auto-evaluator.py auto-evaluator.py
COPY text_utils.py text_utils.py
COPY img img
RUN pip3 install -r requirements.txt

# EXPOSE 8501


ENTRYPOINT ["streamlit", "run", "auto-evaluator.py"]