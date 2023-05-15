import pandas as pd
from itertools import combinations
import preflibtools
import math

MAX_INT = 1_147_483  # _647


def data_transform(all_rankings):
    # Impartial culture
    ids = 0
    output = []
    for each_ranking in all_rankings:
        df_tmp = pd.DataFrame(list(combinations(each_ranking, 2)),
                              columns=["option_a", "option_b"])
        df_tmp["uuid"] = ids
        output.append(df_tmp)
        ids = ids + 1
    return output


def using_preflib(number_proposals, number_individuals, type_="IC"):
    from preflibtools.instances import OrdinalInstance
    instance = OrdinalInstance()
#     5 voters and 10 alternatives
    # instance.populate_mallows_mix(5, 10, 3)
    # instance.populate_urn(5, 10, 76)
    # instance.populate_IC(5, 10)
    # instance.populate_IC_anon(5, 10)
    # sys.maxsize
    from preflibtools.properties import borda_scores, has_condorcet
    if type_ == "IC":
        instance.populate_IC(number_individuals, number_proposals)
    elif type_ == "UM10":
        instance.populate_urn(
            number_individuals, number_proposals, (math.factorial(number_proposals)/9))
    elif type_ == "UM50":
        instance.populate_urn(
            number_individuals, number_proposals, (math.factorial(number_proposals)))
    return list([[y[0] for y in x] for x in instance.full_profile()])


def standard_deviation_rankings(all_rankings):
    #     np.array([[10,1],[5,2],[1,3]])[:,1])
    aux1 = pd.DataFrame(all_rankings).melt().groupby(
        "value")["variable"].std().reset_index()
    aux1.columns = ["id", "value"]
    aux1["rank"] = aux1["value"].rank(method="average", ascending=False)
    return aux1
