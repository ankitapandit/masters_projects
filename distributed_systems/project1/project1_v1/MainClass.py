import googlemaps
from pymongo import MongoClient
from flask import Flask, render_template,request
import json
import polyline
import urllib.request

app = Flask(__name__)
db_client=MongoClient()
dbName=db_client.Maps
weather_data = list()
city_latlng = list()
city_name = list()
# Add your API keys
gmaps = googlemaps.Client(key='Insert your key')
openWeatherMapKey = "Insert your key"
# Flask Key
app.config['SECRET_KEY'] = 'cnf1miou0vdt9d1qe8sx164zejp3fnay'


# Gets list of coordinates from the polyline
def findCoordinates(origin,destination):
    directions_result = gmaps.directions(origin,destination,mode="driving")
    for waypoints in directions_result:
            polyline_input=waypoints['overview_polyline']
    coordinates=polyline.decode(polyline_input['points'])
    findWeather(coordinates)
    temp_values=list()
    max_temp=list()
    min_temp = list()
    humidity = list()
    for weather_main in weather_data:
        temp_values.append(weather_main['main']['temp'])
        max_temp.append(weather_main['main']['temp_max'])
        min_temp.append(weather_main['main']['temp_min'])
        humidity.append(weather_main['main']['humidity'])
    final_data=dict()
    latlng = dbName.Latlng
    return temp_values,city_latlng,city_name,max_temp,min_temp,humidity


# Get weather for the coordinates using OWM API
def findWeather(coordinates):
    weather_key = "Insert your key"
    for coordinate in coordinates:
        if (coordinates.index(coordinate)):
            latitude = coordinate[0]
            longitude = coordinate[1]
            reverse_geocode_result = gmaps.reverse_geocode((latitude, longitude))
            flag = False
            for result in reverse_geocode_result:
                if 'locality' in result['types']:
                    city_latlng.append(str(latitude) + ',' + str(longitude))
                    city_name.append(result['formatted_address'])
                    weather_result = urllib.request.urlopen(
                        "http://api.openweathermap.org/data/2.5/weather?lat=" + str(latitude) + "&lon=" + str(
                            longitude) + "&appid=" + weather_key + "&units=metric")
                    response = weather_result.read().decode('utf-8')
                    json_data = json.loads(response)
                    weather_data.append(json_data)


# Basic flask call to start web app
@app.route("/",methods=['GET', 'POST'])
def basic() :
    if request.method=='POST':
        origin=request.form['origin']
        destination=request.form['destination']
        temperature_to_html,coordinates_to_html,city,max_temp,min_temp,humidity=findCoordinates(origin,destination)
        return render_template("view.html",origin=origin,destination=destination,coordinates=coordinates_to_html,temperature=temperature_to_html,city=city,max=max_temp,min=min_temp,humidity=humidity)
    else:
        return render_template("index.html")


# Call main function
if __name__ == "__main__":
    app.run()
