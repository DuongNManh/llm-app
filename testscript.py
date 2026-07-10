# # /// script
# # requires-python = ">=3.13"
# # dependencies = ["httpx", "pandas", "pydantic", "matplotlib"]
# # ///

# import pandas as pd

# import httpx

# from pydantic import BaseModel

# import matplotlib.pyplot as plt

# url = "https://api.open-meteo.com/v1/forecast"
# params = {
# 	"latitude": 16.04,
# 	"longitude": 13.41,
# 	"hourly": "temperature_2m",
# }

# class HourlyUnits(BaseModel):
#     time: str
#     temperature_2m: str

# class Hourly(BaseModel):
#     time: list[str]
#     temperature_2m: list[float]

# class Response(BaseModel):
#     latitude: float
#     longitude: float
#     elevation: float
#     utc_offset_seconds: int
#     generationtime_ms: float
#     timezone: str
#     timezone_abbreviation: str
#     hourly_units: HourlyUnits
#     hourly: Hourly

# responses = httpx.get(url, params=params).json()

# print("Response keys:", responses.keys())
# response = Response(**responses)
# print(f"Coordinates: {response.latitude}°N {response.longitude}°E")
# print(f"Elevation: {response.elevation} m asl")
# print(f"Timezone difference to GMT+0: {response.utc_offset_seconds}s")

# hourly = response.hourly
# hourly_temperature_2m = hourly.temperature_2m

# hourly_data = {
#     "date": pd.to_datetime(hourly.time),
#     "temperature_2m": hourly_temperature_2m,
# }

# hourly_dataframe = pd.DataFrame(data = hourly_data)
# print("\nHourly data\n", hourly_dataframe)

# fig =plt.figure(figsize=(12, 6))
# plt.plot(hourly_dataframe["date"], hourly_dataframe["temperature_2m"], marker='o', linestyle='-', color='b')
# plt.title("Hourly Temperature Forecast")
# plt.xlabel("Date and Time")
# plt.ylabel("Temperature (°C)")
# plt.legend(["Temperature 2m"])
# plt.grid()
# plt.xticks(rotation=45)
# plt.yticks(range(int(min(hourly_temperature_2m)) - 1, int(max(hourly_temperature_2m)) + 2, 1))

# # show the figure
# plt.tight_layout()
# plt.show()
