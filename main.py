
import os
import requests

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

# 1. Initialize the Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)


# 2. Create the weather tool
@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city.
    Use this tool when the user asks about weather conditions.
    Input should be a city name, such as Mumbai or Bengaluru.
    """
    try:
        # Find the city's latitude and longitude
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json",
            },
            timeout=15,
        )
        geo_response.raise_for_status()
        locations = geo_response.json().get("results", [])

        if not locations:
            return {"error": f"City '{city}' not found."}

        location = locations[0]
        latitude = location["latitude"]
        longitude = location["longitude"]

        # Fetch current weather
        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "apparent_temperature,"
                    "precipitation,"
                    "weather_code,"
                    "wind_speed_10m"
                ),
                "timezone": "auto",
            },
            timeout=15,
        )
        weather_response.raise_for_status()

        weather = weather_response.json()["current"]

        # Convert weather codes into descriptions
        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            80: "Rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail",
        }

        return {
            "city": location["name"],
            "country": location.get("country", ""),
            "time": weather.get("time"),
            "temperature_c": weather.get("temperature_2m"),
            "feels_like_c": weather.get("apparent_temperature"),
            "humidity_percent": weather.get("relative_humidity_2m"),
            "wind_speed_kmh": weather.get("wind_speed_10m"),
            "precipitation_mm": weather.get("precipitation"),
            "condition": weather_codes.get(
                weather.get("weather_code"),
                "Unknown weather condition",
            ),
        }

    except requests.RequestException as exc:
        return {"error": f"Weather service request failed: {exc}"}
    except (KeyError, ValueError) as exc:
        return {"error": f"Could not process weather data: {exc}"}


# 3. Create the AI agent and register its tool
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt=(
        "You are a helpful weather assistant. "
        "When asked about weather, call the get_weather tool. "
        "Answer using the returned data only. "
        "Never invent weather conditions. "
        "If the tool returns an error, explain it clearly. "
        "Mention temperatures in Celsius and wind speed in km/h."
    ),
)


# 4. Run a conversation loop
def main():
    if not os.getenv("GROQ_API_KEY"):
        print("Please set GROQ_API_KEY in your .env file.")
        return

    print("🌦️ Weather AI Agent")
    print("Ask about the weather, or type 'exit' to quit.")

    while True:
        query = input("\nYou: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Agent: Goodbye!")
            break

        if not query:
            continue

        try:
            result = agent.invoke({
                "messages": [
                    {"role": "user", "content": query}
                ]
            })

            # Get the final assistant response
            for message in reversed(result["messages"]):
                if message.type == "ai" and message.content:
                    print(f"\nAgent: {message.content}")
                    break

        except Exception as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    main()
