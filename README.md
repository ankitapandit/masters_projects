# ms_fall18_projects
Masters Fall'18 Projects

This repository contains all projects done as part of Master program in Computer Science.
Currently the repository contains projects for 2 courses –
•	Information Retrieval – CSE 535
•	Distributed Systems – CSE 586


Information Retrieval – 
Project – Tweet Analyzer (Python project)
Objective: Complete Search & Analytics Solution based on dissecting twitter data
Approach: 
1.	Data collected from Twitter using developer account.
2.	Data was indexed on Solr.
3.	GUI provides an elaborate form for user to select specific search parameters. Result generated involved multiple analysis – sentiment analysis, distribution of data based on city and topic, wordcloud and trending hashtags along with the display of filtered tweet data.
Directory structure for tweet_analyzer:
•	Static directory contains css, fonts, img, js files.
•	Templates directory contains HTML files.
•	Util directory contains assets.py which is used for bundling of assets (static files) to the templates and pre-process these files.
•	Init python file is provided.
•	Main python file contains entire code involved to achieve the objective. You might require changing IP address as per your requirement.


Distributed Systems – 
Project 1 – MyWayPoints (Python Project)
Objective: Get Waypoints and weather along the route for selected source and destination. It is done in 2 parts – one with database and another without it.
Approach:
1.	Use GoogleMap API and OpenWeatherMap API.
2.	For version 2, fetch and save in the database.
Directory structure for mywaypoints:
•	Static directory containing img, js, css files.
•	Templates directory contains HTML files.
•	GenerateSecretKey python file generates random secret key used by Flask app.
•	MainClass python file contains code to achieve the objective.
•	README files are provided as well.

Project 2 – Publisher-Subscriber Model implementation (Python Project)
Objective: Emulate Publisher-Subscriber Distributed System using Docker containers.
1.	Implement centralized pub-sub application.
2.	Implement distributed pub-sub application using nodes managing subscribers and events.
3.	Deploy docker node and setup intercommunication between Docker images.
Approach:
1.	Version 1 implements a basic interface which allows user to paste code and the code is compiled in Docker with the result displayed to user on the webpage.
2.	Version 2 implements basic pub-sub application executed in Docker container and perform client request.
3.	Version 3 implements pub-sub application in one docker communicating with another docker containing database and perform client request.
Directory structure for pub-sub model (all 3 versions):
•	README files are provided.
•	Web folder containing – 
o	Static directory that contain css, img files.
o	Templates directory containing HTML files.
o	app python file containing the main logic of the objective.
o	Dockerfile contains the DSL allowing automation in creating image.
o	Requirements files ensures that pip install of the required packages is done by Python.
o	Docker-compose file is a YAML file defining services, networks and volumes.
o	Shell scripting files provide an ease for quicker execution of certain commands.
