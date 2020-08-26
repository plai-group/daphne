import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
from math import sqrt
from sys import argv

from plot_helpers import plot_ensemble, aistats_columnwidth, set_size, setup_matplotlib
setup_matplotlib()


# TODO setup your loader here

loader = ExperimentLoader(
#    mongo_uri=LOCAL_URI,
#    db_name=DATABASE_NAME
)

def build_query(model, loss_choice, experiment=None):
    return {"$and": [
        {"experiment.name": experiment} if experiment else {},
        {"config.gmodel_name": model},
        {"config.flow_connectivity": "faithful",
         "config.loss_choice": loss_choice},
    ]}

colors = {"forward": "blue",
          "backward": "green", 
          "sym": "red"}

names = {"forward": "forward KL",
         "reverse": "reverse KL",
         "backward": "reverse KL",
         "sym": "\\textbf{symmetric KL}"}

def plot_loss_comparison(model, experiment):
    for to_compare in ["reverse_kl", "forward_kl"]:
        fig = plt.figure(figsize=set_size(aistats_columnwidth))
        for loss_choice in colors.keys():
            query = build_query(model, loss_choice, experiment)
            print(model, experiment, loss_choice, to_compare, [exp.metrics[to_compare] for exp in loader.find(query)
                               if len(exp.metrics[to_compare]) == 2000])
            traces = np.array([exp.metrics[to_compare] for exp in loader.find(query)
                               if len(exp.metrics[to_compare]) == 2000])
            #print(loss_choice, traces.shape)
            plot_ensemble(np.arange(2000), traces,
                          label=names[loss_choice],
                          color=colors[loss_choice])

        plt.grid()
        plt.xlim(0, 2000)
        if to_compare != "forward_kl":
            plt.yscale("log")
        plt.ylabel("expected shifted \\textit{" + to_compare.replace('_kl', ' KL') + "}")
        plt.xlabel("training step")
        plt.legend(loc="upper right", fancybox=True)
        plt.tight_layout()
        #plt.show()
        plt.savefig(to_compare + "_log_comparison.pdf", bbox_inches="tight")
        plt.close()


    # plot symmetric KL
    fig = plt.figure(figsize=set_size(aistats_columnwidth))
    to_compare = "moving_sym_kl"
    for loss_choice in colors.keys():
        query = build_query(model, loss_choice, experiment)
        traces = np.array([exp.metrics["reverse_kl"] for exp in loader.find(query)
                           if len(exp.metrics["reverse_kl"]) == 2000]) * 0.5
        traces += np.array([exp.metrics["forward_kl"] for exp in loader.find(query)
                            if len(exp.metrics["forward_kl"]) == 2000]) * 0.5
        print(loss_choice, traces.shape)
        plot_ensemble(np.arange(2000), traces,
                      label=names[loss_choice], color=colors[loss_choice])

    plt.grid()
    plt.xlim(0, 2000)
    plt.yscale("log")
    #plt.ylim(bottom=0.01)
    plt.ylabel("expected " + to_compare.replace("_kl", " KL").replace("moving_sym", "\\textit{symmetric}"))
    plt.xlabel("training step")
    plt.legend(loc="upper right")
    plt.tight_layout()
    #plt.show()
    plt.savefig(to_compare + "_log_comparison.pdf", bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    # paper plot comparisons:
    plot_loss_comparison(argv[1], argv[2])


#plot_loss_comparison("state_space", "kl_convergence_test4")
