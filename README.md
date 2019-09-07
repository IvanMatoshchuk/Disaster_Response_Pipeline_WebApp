# Disaster Response Pipeline Project

Udacity Data Science Nanodegree Term 2

1. [Project Description](#Description)
2. [Instructions](#Instructions)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Description <a name="Instructions"></a>

This projects consists of the model that classifies disaster messages and a Wep App.
Web App includes 3 Graphs of the data analysis and a text classifier.

### Instructions: <a name="Instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. In the terminal, use this command to get the link for vieweing the app:
env | grep WORK

The link wil be:
http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN replacing WORKSPACEID and WORKSPACEDOMAIN with your values.

3. Run the following command in the app's directory to run your web app.
    `python run.py`



### File Descriptions <a name="files"></a>

```process_data.py``` is used as the pipeline for processing the data and preparing in for the further usage. 
```train_classifier.py``` is used to create a model needed for the given classification problem.
`run.py`, `go.html`, `master.html` are used to run the Web App.
Markdown cells were used to assist in walking through the thought process for individual steps.  



### Results<a name="results"></a>

3 Graphs and a text classifier can be observed in the constructed Web App.
Example:


![alt text](https://raw.githubusercontent.com/IvanMatoshchuk/Disaster_Response_Pipeline_WebApp/master/WebApp.PNG)


### Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Udacity for the data. Feel free to use the code here as you would like! 

