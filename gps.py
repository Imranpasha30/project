
import subprocess
import json

# Function to get the current location using PowerShell
def get_current_location():
    try:
        # PowerShell script to request location permissions and get the location
        ps_script = '''
            Add-Type -AssemblyName System.Device
            $loc = New-Object System.Device.Location.GeoCoordinateWatcher
            $loc.ReportInterval = 0
            $loc.Start()
            # Wait for the GeoCoordinateWatcher status to change to 'Ready' or 'Denied'
            while ($loc.Status -ne 'Ready' -and $loc.Permission -ne 'Denied') {
                Start-Sleep -Seconds 1
            }
            # Check if the permission is granted
            if ($loc.Permission -eq 'Granted') {
                # If granted, get the location
                $position = $loc.Position.Location
                $loc.Stop()
                [math]::Round($position.Latitude, 15) | Out-File -FilePath "latitude.txt"
                [math]::Round($position.Longitude, 15) | Out-File -FilePath "longitude.txt"
            } else {
                # If denied, output a message
                Write-Output 'Location permission is denied.'
            }
        '''
        # Running the PowerShell script
        process = subprocess.run(["powershell", "-Command", ps_script], capture_output=True)
        # Decoding the output to JSON
        latitude = open("latitude.txt", "r").read()
        longitude = open("longitude.txt", "r").read()
        location_data = {
            "Latitude": latitude,
            "Longitude": longitude
        }
        # Returning the location data
        return location_data
    except subprocess.SubprocessError as e:
        return f"An error occurred while trying to retrieve the location: {e}"
    except json.JSONDecodeError as e:
        return "Failed to decode the location data from PowerShell output."

# Call the function and print the result
location = get_current_location()
# Check if location data was successfully retrieved
if isinstance(location, dict) and 'Latitude' in location and 'Longitude' in location:
    print(f"Latitude: {location['Latitude']}")
    print(f"Longitude: {location['Longitude']}")
else:
    print("Failed to retrieve location data.")
