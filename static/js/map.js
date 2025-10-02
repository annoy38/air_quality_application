const map = L.map("map").setView([23.8103,90.4125], 11);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",{maxZoom:19}).addTo(map);
L.marker([23.8103,90.4125]).addTo(map).bindPopup("Dhaka AQI: 92");
