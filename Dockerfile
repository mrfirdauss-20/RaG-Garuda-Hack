FROM python:3.7-slim

COPY . .

WORKDIR /

RUN cat /requirements.txt

RUN pip install --no-cache-dir --upgrade -r /requirements.txt

ENV API_KEY=${API_KEY}
ENV TRANSFORMERS_CACHE=/transformers_cache

RUN mkdir -p  /transformers_cache && chmod -R 777  /transformers_cache

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
