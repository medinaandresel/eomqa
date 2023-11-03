# KGReasoning using Q2B and O2B
**Data**

- test_ind accounts for test inductive case, test_ded for deductive case and test_ind_ded for inductive+deductive case.

For S either Q2B (also known as [Query2Box](https://github.com/hyren/query2box)) or O2B (our ontology-aware version of Q2B):

- train_plain is for S_plain;
- train_gen is for S_gen; 
- train_spe is for S_spe;
- train_onto is S_onto.

In order to reproduce all experiments please combine each train with each test data.


**Run Q2B and O2B**

1. To run Q2B please check ```example.sh```

2. To run O2B please add boolean paramenters: ```--tbox``` and ```--mean_gen```

3. To run the rewriting over the pre-trained Q2B, please use: ```--checkpoint_path  <path_to_trained_model>``` and ```--query_rew```  (see ``example.sh``)

