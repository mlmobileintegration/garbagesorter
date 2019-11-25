FROM denismakogon/opencv3-slim:edge

COPY requirements.txt .

RUN pip install --no-cache-dir --no-cache-dir --upgrade -r requirements.txt

COPY app app/

RUN python app/server.py

EXPOSE 8080

CMD ["python", "app/server.py", "serve"]
