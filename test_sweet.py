from fastapi.testclient import TestClient
from sweet import service

client = TestClient(service)


def test_get_countries():
    response = client.get('/v1/countries')
    assert response.status_code == 200
    assert response.json()

def test_get_country_by_name():
    response = client.get('/v1/countries/fake')
    assert response.status_code == 404

    response = client.get('/v1/countries/Singapore')
    assert response.status_code == 200
    assert response.json()


# add more tests...