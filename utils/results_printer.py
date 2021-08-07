

def write_results(modelname,lr, bs, do_str,do_prob,ep,l2, acc,filename="history.txt"):
    with open(filename,"a") as f:
        f.write(modelname)
        f.write(f"\n\tLearning Rate : {lr}\n")
        f.write(f"\tBatch size : {bs}\n")
        f.write(f"\tDrop Out : {do_str} (p = {do_prob})\n")
        f.write(f"\tEpochs : {ep}\n")
        f.write(f"\tL2 : {l2}\n")
        f.write(f"\tBest achieved accuracy : {acc}\n\n")
