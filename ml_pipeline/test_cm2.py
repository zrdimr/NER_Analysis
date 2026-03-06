import numpy as np

Accuracy = 0.7664
Precision = 0.7778
Recall = 0.7636
# dreaddit,False,bert,transformer

Pr_calc = (1 - Accuracy) / (1 - 2 * Recall + Recall / Precision)
TP = Pr_calc * Recall
FN = Pr_calc - TP
FP = (TP / Precision) - TP
TN = 1 - Pr_calc - FP

print("Pr_calc:", Pr_calc)
print("TP:", TP)
print("FP:", FP)
print("FN:", FN)
print("TN:", TN)
print("Acc Check:", TP + TN)
print("Prec Check:", TP / (TP + FP))
print("Rec Check:", TP / (TP + FN))
