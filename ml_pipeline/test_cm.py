import numpy as np

Accuracy = 0.7581
Precision = 0.6812
Recall = 0.8545
Pos_train = 258
Neg_train = 242

Pr = Pos_train / (Pos_train + Neg_train)
Nr = Neg_train / (Pos_train + Neg_train)

TP_f = Recall * Pr
FN_f = Pr - TP_f

FP_f = (TP_f / Precision) - TP_f
TN_f = Nr - FP_f

Acc_calculated = TP_f + TN_f
print("Acc calculated:", Acc_calculated, "vs", Accuracy)
print("TP:", TP_f)
print("FP:", FP_f)
print("FN:", FN_f)
print("TN:", TN_f)

