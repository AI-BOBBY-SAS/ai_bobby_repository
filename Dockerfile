FROM python:3.12

WORKDIR /app

COPY uv.lock .
RUN pip install uv

COPY . .

RUN uv sync

EXPOSE 8080

CMD ["uv", "run", "streamlit", "run", "src/app/main.py", "--server.port", "8080"]
