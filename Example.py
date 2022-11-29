# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:22:01 2022

@author: Jiawei Zheng, Petros Papapanagiotou, Jacques D. Fleuriot
"""


from pm4py.objects.petri.petrinet import PetriNet, Marking
from pm4py.objects.petri.synchronous_product import construct, construct_cost_aware
from pm4py.objects.petri import utils
import sys 
sys.path.append("./utils") 
import log_net
import pandas as pd
import numpy as np

from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.objects.petri import align_utils


from pm4py.objects.petri.importer import importer as pnml_importer

# import petri net of the process model
net1, im1, fm1 = pnml_importer.apply("./branching model.pnml")

df = pd.read_csv("./event.csv")
df = df.fillna(0)

# Visualize process model
gviz_model = pn_visualizer.apply(net1, im1, fm1)
pn_visualizer.view(gviz_model)

# Visualize weighted trace model
log_net.visualize_log_net(df)

# alignment based on probabilistic events
import probabilistic_alignments

model_dic = {}
model_dic["net"] = net1
model_dic["im"] = im1
model_dic["fm"] = fm1

result =  probabilistic_alignments.apply(df,model_dic,0.4)
print(result)
align_utils.pretty_print_alignments(result)

# standard alignment by using argmax transform to deterministic events

event_df = df.copy()
event_df["concept:name"] = event_df.idxmax(axis=1)

event_df = event_df.loc[:,["concept:name"]]
event_df["time:timestamp"] = np.arange(len(event_df))
event_df["case:concept:name"] = 1

from pm4py.objects.conversion.log import converter as log_converter

event_log = log_converter.apply(event_df)

from pm4py.algo.conformance.alignments import algorithm as alignments
from pm4py.algo.conformance.alignments import variants

replayedtrace_flow1 = alignments.apply(event_log, model_dic["net"], model_dic["im"], model_dic["fm"], variant=variants.state_equation_a_star)
align_utils.pretty_print_alignments(replayedtrace_flow1[0])
print(replayedtrace_flow1)







