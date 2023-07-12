## SWEET Service

### Set up
- `python -m venv venv`
- `source venv/bin/activate`
- `pip install -r requiments.txt`

### Test

`python -m pytest`

### Run locally 

`uvicorn sweetr:service`

### Build and run in Docker

`docker build -t sweet-service . && docker run -p 8000:8000 sweet-service`

Access service at [http://localhost:8000](http://localhost:8000) 