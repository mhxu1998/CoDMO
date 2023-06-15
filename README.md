CoDMO
===
This is the source code for ***[Cooperative Dual Medical Ontology Representation Learning for Clinical Assisted Decision-Making]***.

[Cooperative Dual Medical Ontology Representation Learning for Clinical Assisted Decision-Making]:https://www.sciencedirect.com/science/article/pii/S0010482523006030

Requirements
----
This project was run in a conda virtual environment on Ubuntu 16.04 with CUDA 10.1. 
+ Pytorch==1.7.1
+ Python==3.7.9
+ theano==1.0.5

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

Model training
----
1. Process raw MIMIC data.
```
python process_mimic.py --out_path processed/
```
2. The original code is transformed using CCS to generate the final input and label information. And alse get the hierarchy of the ontology.
```
python build_trees.py --out_path processed/
```
3. Pretrain the code embedding by Glove. The initialization of the basic embeddings follows the same procedure as in [GRAM].
```
cd preTrain
python glove.py
```
4. Train the model.
```
python main.py --device <gpu_index> --embed_file_diag preTrain/diag/diag.npz --embed_file_proc preTrain/proc/proc.npz
```
[GRAM]:https://github.com/mp2893/gram

Cite
----
If you find the paper or the implementation helpful, please cite the following paper:
```
@article{xu2023cooperative,
  title={Cooperative dual medical ontology representation learning for clinical assisted decision-making},
  author={Xu, Muhao and Zhu, Zhenfeng and Li, Youru and Zheng, Shuai and Li, Linfeng and Wu, Haiyan and Zhao, Yao},
  journal={Computers in Biology and Medicine},
  volume={163},
  pages={107138},
  year={2023},
  publisher={Elsevier}
}
```

