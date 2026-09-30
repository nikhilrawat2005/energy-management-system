$ErrorActionPreference = "Continue"
Set-Location "c:\Users\NIKHIL RAWAT\OneDrive\Desktop\Nikhil PROJECT\energy system"
python -m ingestion.live_streamer *>> "logs\streamer.log"
