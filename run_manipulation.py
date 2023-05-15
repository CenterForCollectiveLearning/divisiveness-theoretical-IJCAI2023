from comchoice.aggregate import divisiveness, win_rate, copeland, borda
from helpers import using_preflib

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns
import math

# SELET DATASET TO USE
parser = argparse.ArgumentParser()

parser.add_argument("-a", "--alternatives", default=10,
                    type=int, required=False)
parser.add_argument("-k", "--kind", default="last",
                    type=str, required=False)
parser.add_argument("-i", "--iterations", default=30, type=int, required=False)
parser.add_argument("-t", "--type", default="UM10",
                    type=str, required=False)
parser.add_argument("-m", "--method", default="copeland",
                    type=str, required=False)
parser.add_argument("-s", "--starting", default=10,
                    type=int, required=False)

parser.add_argument("-st", "--step", default=10,
                    type=int, required=False)

args = parser.parse_args()

_type = args.type
kind = args.kind
method_name = args.method
n_alternatives = args.alternatives
n_iterations = args.iterations
step = args.step

# starting_issue = args.starting


def get_manipulation(
    agents_initial_size=100,
    n_alternatives=10,
    max_agents=1000,
    method=copeland,
    starting_issue=10,
    step=10,
    _type="UM10"
):
    """Heuristic for manipulation

    Parameters
    ----------
    agents_initial_size : int, optional
        Initial number of agents, by default 100
    n_alternatives : int, optional
        Number of issues, by default 10
    max_agents : int, optional
        Exit condition for the heuristic. It represent the maximum number of agents to be added, by default 1000
    method : str, optional
        Voting method function, by default copeland
    starting_issue : int, optional
        Issue to manipulate its ranking, by default 10
    step : int, optional
        New agents added before calculate divisiveness, by default 10
    _type : str, optional
        Agents' generation method, by default "UM10"

    Returns
    -------
    pd.DataFrame
        A DataFrame with the manipulation
    """
    i = 1
    method_kws = dict() if method_name == "copeland" else dict(score="weighted")

    output_divisiveness = []

    data = using_preflib(
        number_proposals=n_alternatives,
        number_individuals=agents_initial_size,
        type_=_type
    )

    alternatives = data[0]
    data = [">".join(map(str, x)) for x in data]

    df = pd.DataFrame(data, columns=["ballot"])
    df["voter"] = range(df.shape[0])

    voter = agents_initial_size * 1

    df_dv = divisiveness(
        df,
        dtype="ballot",
        convert_pairwise_kws=dict(),
        method=method,
        method_kws=method_kws
    )

    df_dv["size"] = agents_initial_size
    df_dv["iteration"] = 0

    alternative_id_manipulate = starting_issue - 1

    alternative_id = df_dv["alternative"].unique()[alternative_id_manipulate]
    df_dv["custom_id"] = range(1, df_dv.shape[0] + 1)
    df_dv["alternative_of_interest"] = df_dv["alternative"] == alternative_id

    rank = method(df)
    custom_id = df_dv[["custom_id", "alternative"]]

    output_divisiveness.append(df_dv)

    rank = rank[rank["alternative"].astype(str) != str(alternative_id)]
    alternatives_rmv = list(rank["alternative"])
    # alternatives_rmv = list(alternatives_rmv)

    while i <= max_agents:

        # random.shuffle(alternatives_rmv)

        # If i is odd
        if i % 2 == 1:
            ballot = ">".join(map(str, [alternative_id] + alternatives_rmv))
            df2 = pd.DataFrame({"ballot": [ballot], "voter": [voter]})
            df = pd.concat([df, df2], ignore_index=True)
        else:
            ballot = ">".join(map(str, alternatives_rmv + [alternative_id]))
            df2 = pd.DataFrame({"ballot": [ballot], "voter": [voter]})
            df = pd.concat([df, df2], ignore_index=True)

        # df["alternative"] = df["alternative"].astype(str)

        if i % step == 0 and i > 1:

            df_dv = divisiveness(
                df,
                dtype="ballot",
                convert_pairwise_kws=dict(),
                method=method,
                method_kws=method_kws
            )
            df_dv["size"] = voter
            df_dv["alternative_of_interest"] = df_dv["alternative"] == alternative_id
            df_dv = pd.merge(df_dv, custom_id, on="alternative")
            df_dv["iteration"] = i

            output_divisiveness.append(df_dv)

        voter += 1
        i += 1

    df_manipulation = pd.concat(output_divisiveness)
    df_manipulation["starting_id"] = starting_issue
    df_manipulation["type"] = _type

    return df_manipulation


output = []

for starting_issue in [2, int(math.ceil(n_alternatives/2)), n_alternatives]:
    _ = copeland if method_name == "copeland" else borda
    max_agents = 101
    if method_name == "UM50":
        max_agents = 151
    for iteration in range(n_iterations):
        try:
            dd = get_manipulation(
                _type=_type,
                kind=kind,
                max_agents=max_agents,
                n_alternatives=n_alternatives,
                starting_issue=starting_issue,
                step=step,
                method=_
            )

            # dd.to_csv(
            #     f"data_output/{method_name}_{_type}_{n_alternatives}_alternatives_{iteration}_it_50_voters_{starting_issue}.csv",
            #     index=False
            # )

            output.append(dd)
        except:
            pass

df = pd.concat(output, ignore_index=True)
df["alternative"] = df["alternative"].astype(str)
df["n_alternatives"] = n_alternatives

df.to_csv(
    f"data/{method_name}_{_type}_{n_alternatives}_alternatives_{n_iterations}_iterations_step_{step}.csv",
    index=False
)
