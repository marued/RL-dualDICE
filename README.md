# RL-dualDICE

This repositery is a case study of these papers:
```
@article{nachum2019dualdice,
    title={DualDICE: Behavior-Agnostic Estimation of Discounted Stationary
    Distribution Corrections},
    author={Nachum, Ofir and Chow, Yinlam and Dai, Bo and Li, Lihong},
    journal={NeurIPS},
    year={2019}
}

@article{liu2018breaking,
    title={Breaking the curse of horizon: Infinite-horizon off-policy estimation},
    author={Liu, Qiang and Li, Lihong and Tang, Ziyang and Zhou, Dengyong},
    journal={arXiv preprint arXiv:1810.12429},
    year={2018}
}
```
Their relevent github can be found at: https://github.com/google-research/google-research/tree/master/dual_dice
and 
https://github.com/zt95/infinite-horizon-off-policy-estimation

# Goal

Reproducing expiriments from DualDICE on the Taxi environment. 

## Running experiments

The project has a visual code settings file; you can simply open the project is VS Code and execute the different relevent python files

Our main experiment starts from the file called run_graphs_compare_both.py and we isolated each project within their own subfolders.
We integrated multiprocessing to run experiments faster and made some changes to the 2 environments to make them work in this context.

```
python3 run_graphs_compare_both.py
```

### More information
Here is a small video explaining the goal of the project: https://youtu.be/no-JKqfD0zw
It's possible to refer our small analisys in the fallowing document: https://github.com/marued/RL-dualDICE/blob/master/COMP767_Project.pdf
