uvicorn api:app --reload

curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/home/olga/Downloads/brain-tumor-detection-master/image.jpg'
