FROM denismakogon/opencv3-slim:edge

COPY requirements.txt .

RUN pip install --no-cache --no-cache-dir -r requirements.txt

COPY app app/

RUN python app/server.py

EXPOSE 8080

CMD ["python", "app/server.py", "serve"]
