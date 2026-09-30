"""
Open-Meteo weather fetcher with forecast parsing, retry/backoff, and safe coercion.

This replaces the original which discarded the hourly payload it requested.
Now it returns both current conditions AND the next-hour forecast.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class WeatherAPIFetcher:
    """
    Fetches real-time and forecast weather data using Open-Meteo Solar API
    (no API key required, free tier 10,000 calls/day).
    """

    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    DEFAULT_LAT = 28.6139  # Delhi
    DEFAULT_LON = 77.2090
    USER_AGENT = "MAEMS-Energy-System/1.1"

    def __init__(
        self,
        lat: float = DEFAULT_LAT,
        lon: float = DEFAULT_LON,
        max_retries: int = 3,
        base_delay: float = 1.0,
        timeout: float = 8.0,
    ):
        self.lat = lat
        self.lon = lon
        self.max_retries = max(1, int(max_retries))
        self.base_delay = float(base_delay)
        self.timeout = float(timeout)

    def _build_url(self) -> str:
        return (
            f"{self.BASE_URL}?"
            f"latitude={self.lat}&longitude={self.lon}&"
            f"current=temperature_2m,relative_humidity_2m,direct_normal_irradiance,cloud_cover&"
            f"hourly=temperature_2m,relative_humidity_2m,direct_normal_irradiance,cloud_cover&"
            f"forecast_days=2&timezone=UTC"
        )

    def _fetch_json(self, url: str) -> Optional[Dict[str, Any]]:
        req = urllib.request.Request(url, headers={"User-Agent": self.USER_AGENT})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def fetch_current_and_forecast(self) -> Dict[str, Any]:
        """
        Returns a dict with:
          - status: "online" | "offline_fallback"
          - source: "open-meteo"
          - ambient_temp, humidity, irradiance_wm2, cloud_cover_pct (current)
          - forecast: list of next 4 hourly points (temp, humidity, irradiance, cloud)
          - next_hour_irradiance: float (irradiance at t+1h) or None
        """
        url = self._build_url()
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                data = self._fetch_json(url)
                current = data.get("current", {})
                hourly = data.get("hourly", {})

                # Build forecast list (next 4 hours)
                forecast: List[Dict[str, Any]] = []
                for i in range(min(4, len(hourly.get("time", [])))):
                    forecast.append({
                        "time_utc": hourly["time"][i],
                        "ambient_temp": hourly["temperature_2m"][i],
                        "humidity": hourly["relative_humidity_2m"][i],
                        "irradiance_wm2": hourly["direct_normal_irradiance"][i],
                        "cloud_cover_pct": hourly["cloud_cover"][i],
                    })

                next_hour_irradiance = None
                if len(forecast) > 1:
                    next_hour_irradiance = float(forecast[1]["irradiance_wm2"])

                return {
                    "status": "online",
                    "source": "open-meteo",
                    "ambient_temp": float(current.get("temperature_2m", 25.0)),
                    "humidity": float(current.get("relative_humidity_2m", 50.0)),
                    "irradiance_wm2": float(current.get("direct_normal_irradiance", 0.0)),
                    "cloud_cover_pct": float(current.get("cloud_cover", 0.0)),
                    "forecast": forecast,
                    "next_hour_irradiance": next_hour_irradiance,
                }

            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.base_delay * attempt)

        # Fallback after all retries exhausted
        return {
            "status": "offline_fallback",
            "error": str(last_exc) if last_exc else "unknown",
            "ambient_temp": 28.0,
            "humidity": 55.0,
            "irradiance_wm2": 450.0,
            "cloud_cover_pct": 20.0,
            "forecast": [],
            "next_hour_irradiance": None,
        }


if __name__ == "__main__":
    fetcher = WeatherAPIFetcher()
    print("Testing Live Weather API Fetch:")
    result = fetcher.fetch_current_and_forecast()
    print(f"Status: {result['status']}")
    print(f"Current: T={result['ambient_temp']:.1f}C, RH={result['humidity']:.0f}%, "
          f"DNI={result['irradiance_wm2']:.0f}W/m2, Cloud={result['cloud_cover_pct']:.0f}%")
    if result.get("next_hour_irradiance") is not None:
        print(f"Next hour DNI: {result['next_hour_irradiance']:.0f}W/m2")