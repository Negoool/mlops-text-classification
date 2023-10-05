From python:3.10.11-slim

WORKDIR /app

COPY  . .

RUN python -m pip install .

CMD ["uvicorn", "deploy.api:app", "--host",  "0.0.0.0"]