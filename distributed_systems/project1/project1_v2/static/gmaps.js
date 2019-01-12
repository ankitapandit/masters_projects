var map;
var geoJSON;
var request;
var myLatLng;
var gettingData = false;
var openWeatherMapKey = "fdc0d2bf3fbb538ce1ab83556f675500"

function initMap() {
	var markerArray = [];
	var directionsService = new google.maps.DirectionsService;
	var directionsDisplay = new google.maps.DirectionsRenderer({map:map});
	var stepDisplay=new google.maps.InfoWindow
	for (var i = 0; i < markerArray.length; i++) {
		markerArray[i].setMap(null);
	}
	var geocoder = new google.maps.Geocoder();
	var address = document.getElementById("from").value;
	geocoder.geocode( { 'address': address}, function(results, status) {
		if (status == google.maps.GeocoderStatus.OK) {
			var latitude = results[0].geometry.location.lat();
			var longitude = results[0].geometry.location.lng();
			myLatLng = {lat: latitude, lng: longitude};
			var map = new google.maps.Map(document.getElementById('map'), {
				zoom: 4,
				center: myLatLng
			});
			directionsDisplay.setMap(map);
			directionsService.route({
				origin: document.getElementById('from').value,
				destination: document.getElementById('to').value,
				optimizeWaypoints: true,
				provideRouteAlternatives: true,
				unitSystem: google.maps.UnitSystem.IMPERIAL,
				travelMode: 'DRIVING'
			}, function(response, status) {
				if (status === 'OK') {
					directionsDisplay.setDirections(response);
					showSteps(response, markerArray, stepDisplay, map);
				} else {
					window.alert('Directions request failed due to ' + status);
				}
			});
		}
	});
}
function showSteps(directionResult, markerArray, stepDisplay, map) {
	var myRoute = directionResult.routes[0].legs[0];
	for (var i = 0; i < myRoute.steps.length; i++) {
		var marker = markerArray[i] = markerArray[i] || new google.maps.Marker;
		marker.setMap(map);
		marker.setPosition(myRoute.steps[i].start_location);					
		attachInstructionText(
			stepDisplay, marker, myRoute.steps[i].instructions, map);
	}
}
function attachInstructionText(stepDisplay, marker, text, map) {
	google.maps.event.addListener(marker, 'mouseover', function() {
		var lat = marker.getPosition().lat();
		var lng = marker.getPosition().lng();
		initialize(lat,lng,map,marker,text);
		displayRoute(marker,text);
	});
	google.maps.event.addListener(marker,'mouseout', function() {
		infowindow.close();
	});
}
function initialize(lat,long,map,marker,text) {
	var myLatLng1 = {lat: lat, lng: long};
	var mapOptions = {
		zoom: 4,
		center: myLatLng1
	};
	getCoords(lat,long);
	infowindow.open(map,marker);
}
var checkIfDataRequested = function() {
	while (gettingData === true) {
		request.abort();
		gettingData = false;
	}
};

var getCoords = function(lat,long) {
	getWeather(lat,long);
};

var getWeather = function(lat,long) {
	var requestString = "http://api.openweathermap.org/data/2.5/weather?lat=" + lat + "&lon=" + long + "&appid=" + openWeatherMapKey+"&units=metric";
	request = new XMLHttpRequest();
	request.open("get", requestString, false);
	request.send();
	var returned=JSON.parse(request.response);
	var temp=returned.main.temp;
	var temp_min=returned.main.temp_min;
	var temp_max=returned.main.temp_max;
	var weatherCondition=returned.weather[0].main;
	var humidity=returned.main.humidity;
	var weatherIcon=returned.weather[0].icon;
	var iconUrl="http://openweathermap.org/img/w/" + weatherIcon + ".png";
	infowindow.setContent("Temperature: " + temp + "&nbsp;Minimum: " + temp_min + "&nbsp;Maximum: " + temp_max + "<br>Weather condition: " + weatherCondition + "<br>Humidity: " + humidity + "<br>Weather Mood: <img src=" + iconUrl + ">");
};

function displayRoute(marker,text) {
	google.maps.event.addListener(marker, 'click', function() {
		var summaryPanel = document.getElementById('directions');
		summaryPanel.innerHTML = '';
		summaryPanel.innerHTML += '<b>Route: ' + text + '</b><br>';
	});
}

var infowindow = new google.maps.InfoWindow();