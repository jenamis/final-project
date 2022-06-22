## Data & Database

### Data Source

This project utilizes datasets provided by the data team at the automated chemistry platform. The first dataset lists compounds tested by the platform over the past two years and includes compound properties such as molecular weight, topological polar surface area (TPSA), quantitative estimate of drug-likeness (QED), among many others that may be relevant to predicting the appropriate SPE method to use for compound purification. These are calculated using RDkit from input SMILES string. The file for these calculations can be found at [chemCalculate.py](database/chemCalculate.py). 

The second dataset includes the status of testing for each compound and the SPE method used for each compound that has completed the purification stage. Each compound is identified by a unique structure ID, and proprietary information about the actual structure of the compound has been excluded from the datasets.

### Data ETL Process

1. Raw data is extracted from chemistry platform database as CSV files. These are accessible from AWS S3 buckets:
   - https://purifai.s3.us-west-1.amazonaws.com/data/outcomes.csv
   - https://purifai.s3.us-west-1.amazonaws.com/data/structures.csv
2. Data is transformed and cleaned using pandas in this file [clean_dataset.ipynb](database/clean_dataset.ipynb) and generates new CSV files. These were uploaded to and are accessible from AWS S3 buckets:
   - https://purifai.s3.us-west-1.amazonaws.com/clean-data/cleaned-outcomes.csv
   - https://purifai.s3.us-west-1.amazonaws.com/clean-data/cleaned-structures.csv
3. Data is loaded into the AWS database (*purifai.ceoinb9nwfxg.us-west-1.rds.amazonaws.com*) using PySpark using this file [purifAI_database.ipynb](database/purifAI_database.ipynb).
4. Data for SPE analysis is extracted as a merged table (`spe_analysis_df`) using SQLAlchemy and pandas. This dataframe for analysis is obtained using the code from [spe_analysis_data.ipynb](database/spe_analysis_data.ipynb).