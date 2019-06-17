# RF_SDD

Learns a Random Forest (RF) classifier from data using WEKA, and then converts it into a Sentential Decision Diagram (SDD).

### Input Format

The input is specified using "config.json". The fields that should be configured are:

- "dataset": the name of the dataset
- "tree_count": the maximum number of trees for the RF (passed to WEKA)
- "tree_depth": the maximum depth for each tree in the RF (passed to WEKA)
- "threshold_ratio": the ratio for merging similar thresholds; a ratio of 1.0 does not do any merging

The training and testing file of the dataset should be located at ```"data/%s/%s.train.csv"``` and ```"data/%s/%s.test.csv"``` after replacing each occurrence of ```%s``` with ```config["dataset"]```.

For example if ```config["dataset"] = "magic"``` then the training and testing file should be located at ```"data/magic/magic.train.csv"``` and ```"data/magic/magic.test.csv"```

### Output Format

The output of the program will be 4 files:

- SDD file: The SDD representation of the decision function of the learned RF classifier
- vtree file: The vtree accompanying the SDD
- variable description file: A description of the variables of the SDD (includes variable order, metadata)

After replacing each occurrence of ```%s``` with ```config["dataset"]```,
the SDD file will be written at ```config["sdd_filename"]```,
the vtree file will be written at ```config["vtree_filename"]```, and
the variable description file will be written at ```config["constraint_filename_output"]```.

### Running RF_SDD

```
./run
```

### Further Questions

Contact us at 
```
Andy Shih: andyshih@cs.ucla.edu
Arthur Choi: aychoi@cs.ucla.edu
Adnan Darwiche: darwiche@cs.ucla.edu
```
