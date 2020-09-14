# Black-Box-ECG-attack
The code in this repository is for the paper "Hard-Label Black-Box Adversarial Attack on Deep Electrocardiogram Classifier".
#### File Descriptions
attack.py: main document for running attack<br/>
boundary_attack.py: function definitions used in attack.py, saving necessary numpy files of progress of attack<br/>
utils.py: additional function definitions, specifically the smoothing function for perturbations<br/>

There are comments within the ECG_Adversarial_Attacks.pynb about how to run both the white box and black box attacks. The code cells should be run in order which they are presented. The necessary files are all cited, including the dataset, white-box attack, and required versions of TensorFlow and other packages. 
 
Further details regarding the white-box attack and its respective arguments can be found in the repository below:
https://github.com/codespace123/ECGadv


Exmaple of Black Box Boundary Attack for Pairs of ECG Data (Original & Target)
```
%cd
%cd /content/drive/My Drive/ECG_v1

# i and j represent the 4 types of ECG classifications 
# k represents how many pairs the user decides to create for each combination of original and target ECG pairs
for i in range(4):
  for j in range(4):
    if(i == j):
      continue
    for k in range(10):
      title = "ECG_orig{}_target{}".format(i,j) +"/" + "ECG_Pair{}".format(k)
      !python attack.py --input_dir ./PairedDataPhysio/$title/ --store_dir ./results_Physio/$title/ --net ./ResNet_30s_34lay_16conv.hdf5 --max_steps steps --s --win len --candidate
```
Optional Arguments
* --input_dir: path for folder containing both original and target ECG numpy files
* --store_dir: path where resulting numpy files from attack will be saved to 
* --net: path for victim ECG classification model
* --max_steps: maximum number of iterations allowed for boundary attack, default is 10000
  * our attack set *steps* to 8000
*  --s: including this argument applies hanning filter to perturbations
  * --win: length of hanning filter applied to pertubations (only used if --s is used)
    * our attack set *len* to 41
* --candidate: including this argument creates a starting point initialized closer to the original

Additional Possible Arguments Not Included
* --max_queries: indicates user wants a limited number of queries, default is infinite
  * our attack used the default, allowing for as many queries as necessary
  
