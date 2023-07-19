FROM python:3.9

WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY *.py /app
COPY *.csv /app

EXPOSE 8000

ENV UVICORN_PORT=8000
ENV UVICORN_HOST=0.0.0.0

CMD ["uvicorn","sweet_v4:service"]