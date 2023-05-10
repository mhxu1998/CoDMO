CoDMO
===
This is the source code for ***Cooperative Dual Medical Ontology Representation Learning for Clinical Assisted Decision-Making***.

Requirements
----
This project was run in a conda virtual environment on Ubuntu 16.04 with CUDA 10.1. 
+ Pytorch==1.4.0
+ Python==3.7.16

Data preparation
----
1. You will first need to request access for **MIMIC-III**, a publicly avaiable electronic health records collected from ICU patients over 11 years. Place the following MIMIC-III CSV files in the *data* directory:
    + ADMISSIONS.csv
    + DIAGNOSES_ICD.csv
    + PROCEDURES_ICD.csv
2. The **Clinical Classifications Software for ICD-9-CM/PCS (CCS)** is applied as the diagnosis and procedure medical ontology. Download the following CSV files from [here] and place them in the *data* directory:
    + ccs_multi_dx_tool_2015.csv
    + ccs_multi_pr_tool_2015.csv
    + ccs_single_dx_tool_2015.csv
    + ccs_single_pr_tool_2015.csv
 
[here]:https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp#download

