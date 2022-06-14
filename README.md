# **Add descriptive title for project**

## Project Overview

### Topic
This project will develop a machine learning model to predict which of two sample preparation and purification methods, also known as solid phase extraction (SPE) methods, is optimal to use for a chemical compound based on properties of that compound's structure. 

### Reason for Selection
The team at an automated chemistry platform that works to automate the process of making chemical compounds to be used in research and development for medicinal **(and other?)** purposes is seeking a machine learning model that can be used to select the best SPE method to test for purification of each chemical compound in a large library of compounds. Without a machine learning model that can effectively predict the optimal SPE method to use, the team must make a best guess of which method to test based on a subset of properties of each compound’s structure. This process can be time consuming and expensive, especially if the wrong SPE method ends up being selected and the purification testing must be repeated using the other method. Development of a machine learning model has the potential to improve the time and cost efficiency of the automated chemistry platform’s process. 

### Data Source
This project utilizes datasets provided by the data team at the automated chemistry platform. The first dataset lists compounds tested by the platform over the past two years **(is this timeframe correct? should we give a specific date range?)** and includes compound properties such as molecular weight, topological polar surface area (TPSA), quantitative estimate of drug-likeness (QED), among many others that may be relevant to predicting the appropriate SPE method to use for compound purification. The second dataset includes the status of testing for each compound and the SPE method used for each compound that has completed the purification stage. Each compound is identified by a unique structure ID, and proprietary information about the actual structure of the compound has been excluded from the datasets.

### Questions to Answer
The questions that will be answered through this project are:
- Which properties of a compound’s structure are relevant to include as features in a machine learning model to predict the optimal SPE method for compound purification?
- Can a machine learning model be developed that has sufficiently high accuracy, precision, and sensitivity for predicting optimal SPE method for compound purification?
- Which machine learning model will perform best for predicting optimal SPE method for compound purification?

## Database

## Machine Learning Model

## Team Communication Protocol
The project team will use the following protocol for communicating about the project: 
- A dedicated Slack channel that includes only team members will be used as the primary method for communication about the project.
- Slack messages directed at a specific team member or members will include a mention for the team member(s).
- All team members will have Slack notifications for the channel turned on so they will be alerted to new project messages. 
- All team members will be expected to respond to Slack messages within 24 hours. If no response is received within 24 hours, communication will be re-sent via text message. 
- Team meetings will be conducted via Google Meet on an as needed basis. The date and time of the next meeting will be determined at the end of each meeting. 
