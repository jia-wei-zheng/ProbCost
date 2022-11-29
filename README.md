# Alignment-based conformance checking over probabilistic event

This repo is the official implementation for the paper Alignment-based Conformance Checking Over Probabilistic Events. For further information related to PM4Py, please refer to https://pm4py.fit.fraunhofer.de/documentation. 

# Prerequisites

+ Python >= 3.7
+ Pm4py ==2.2.0

## First example

The *example.py* file shows a first starting example. 

We provide two simple process model with 3 activities in `pnml` file format, one of which is a branching process model:

![branch](./img/branch.png)

another is a linear process model:

![branch](./img/linear.png)

The probabilistic matrix of uncertain events in provided in `events.csv` file, where you can change the probabilities of events and play around results. 

You can also start playing the algorithm with the following code:

```python
import sys 
sys.path.append("./utils")
import pandas as pd
from pm4py.objects.petri.importer import importer as pnml_importer

net1, im1, fm1 = pnml_importer.apply("./linear mdoel.pnml") # Process model

df = pd.read_csv("./events.csv") # Probabilistic matrix of categorical distribution
df = df.fillna(0)

model_dic = {}
model_dic["net"] = net1
model_dic["im"] = im1
model_dic["fm"] = fm1


import probabilistic_alignments # import probabilistic alignment algorithm

result =  probabilistic_alignments.apply(df, model_dic, 0.4)
print(result)
```

