
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
import os

# Initialize FastMCP server
mcp = FastMCP("weather")

# Constants - OpenWeatherMap API
OPENWEATHER_API_KEY = ""  # Replace with your API key
OPENWEATHER_API_BASE = "https://api.openweathermap.org/data/2.5"



async def make_weather_request(url: str) -> dict[str, Any] | None:
    """Make a request to the OpenWeatherMap API with proper error handling."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API request error: {str(e)}")
            return None

@mcp.tool()
async def get_weather(city: str, country: str = "IN") -> str:
    """Get current weather for a city in India.

    Args:
        city: Name of the city (e.g. Bangalore, Mumbai)
        country: Country code (default is IN for India)
    """
    url = f"{OPENWEATHER_API_BASE}/weather?q={city},{country}&units=metric&appid={OPENWEATHER_API_KEY}"
    data = await make_weather_request(url)

    if not data:
        return f"Unable to fetch weather data for {city}, {country}."

    if data.get("cod") != 200:
        return f"Error: {data.get('message', 'Unknown error')}"

    weather_main = data["weather"][0]["main"]
    weather_desc = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    wind_speed = data["wind"]["speed"]

    return f"""
Current Weather for {city}, {country}:
Condition: {weather_main} ({weather_desc})
Temperature: {temp}°C (Feels like: {feels_like}°C)
Humidity: {humidity}%
Wind Speed: {wind_speed} m/s
    """

@mcp.tool()
async def get_forecast(city: str, country: str = "IN") -> str:
    """Get 5-day weather forecast for a city in India.

    Args:
        city: Name of the city (e.g. Bangalore, Mumbai)
        country: Country code (default is IN for India)
    """
    url = f"{OPENWEATHER_API_BASE}/forecast?q={city},{country}&units=metric&appid={OPENWEATHER_API_KEY}"
    data = await make_weather_request(url)

    if not data:
        return f"Unable to fetch forecast data for {city}, {country}."

    if data.get("cod") != "200":
        return f"Error: {data.get('message', 'Unknown error')}"

    # Format the forecast into readable text
    city_name = data["city"]["name"]
    forecasts = []
    
    # Get one forecast per day (at noon)
    days_added = set()
    for item in data["list"]:
        date = item["dt_txt"].split(" ")[0]
        time = item["dt_txt"].split(" ")[1]
        
        # Only add one entry per day (around noon)
        if date in days_added or not ("11:00:00" < time < "15:00:00"):
            continue
            
        days_added.add(date)
        
        weather = item["weather"][0]["main"]
        description = item["weather"][0]["description"]
        temp = item["main"]["temp"]
        feels_like = item["main"]["feels_like"]
        humidity = item["main"]["humidity"]
        
        forecast = f"""
Date: {date}
Condition: {weather} ({description})
Temperature: {temp}°C (Feels like: {feels_like}°C)
Humidity: {humidity}%
"""
        forecasts.append(forecast)
        
        # Limit to 5 days
        if len(forecasts) >= 5:
            break

    if not forecasts:
        return f"No forecast data available for {city}, {country}."

    return f"5-Day Forecast for {city_name}:\n" + "\n---\n".join(forecasts)

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')