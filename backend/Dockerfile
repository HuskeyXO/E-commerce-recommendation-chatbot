FROM tiangolo/uvicorn-gunicorn:python3.9
WORKDIR /backend
COPY . /backend
RUN pip install --no-cache-dir -r requirements.txt && pip list
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
