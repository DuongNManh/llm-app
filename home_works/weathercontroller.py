from pydantic import BaseModel
import random
# import requests # sure kèo error vì ko install requests


class Weather(BaseModel):
    temperature: float
    humidity: float
    description: str


async def get_weather(city: str) -> Weather:
    if not city:
        raise ValueError("City name must be provided.")
    temp = random.uniform(15.0, 30.0)
    humidity = random.uniform(40.0, 80.0)
    description = random.choice(["Sunny", "Cloudy", "Rainy", "Windy"])
    return Weather(temperature=temp, humidity=humidity, description=description)


# map method to the http endpoint

# fastapi_app = FastAPI()  # sure kèo error vì ko install fastapi


# @fastapi_app.get("/weather/{city}", response_model=Weather)
# async def weather_endpoint(city: str):
#     return await get_weather(city)
