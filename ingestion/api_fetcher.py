import urllib.request
import json
from typing import Dict, Any, Optional

class WeatherAPIFetcher:
    """
    Fetches real-time and forecast weather data (Temperature, Humidity, Direct Radiation)
    using Open-Meteo Solar API (No API key required, 100% free) or OpenWeatherMap.
    """
    def __init__(self, lat: float = 28.6139, lon: float = 77.2090):
        # Default: Delhi coordinates (can be customized)
        self.lat = lat
        self.lon = lon

    def fetch_current_and_forecast(self) -> Dict[str, Any]:
        """
        Queries Open-Meteo Solar API:
        Returns: ambient_temp, relative_humidity, direct_normal_irradiance, cloud_cover
        """
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={self.lat}&longitude={self.lon}&"
            f"current=temperature_2m,relative_humidity_2m,direct_normal_irradiance,cloud_cover&"
            f"hourly=temperature_2m,direct_normal_irradiance&forecast_days=1"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MAEMS-Energy-System/1.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                current = data.get("current", {})
                return {
                    "status": "online",
                    "source": "open-meteo",
                    "ambient_temp": current.get("temperature_2m", 25.0),
                    "humidity": current.get("relative_humidity_2m", 50.0),
                    "irradiance_wm2": current.get("direct_normal_irradiance", 0.0),
                    "cloud_cover_pct": current.get("cloud_cover", 0.0)
                }
        except Exception as e:
            # Fallback safe defaults if offline
            return {
                "status": "offline_fallback",
                "error": str(e),
                "ambient_temp": 28.0,
                "humidity": 55.0,
                "irradiance_wm2": 450.0,
                "cloud_cover_pct": 20.0
            }

if __name__ == "__main__":
    fetcher = WeatherAPIFetcher()
    print("Testing Live Weather API Fetch:")
    print(fetcher.fetch_current_and_forecast())
