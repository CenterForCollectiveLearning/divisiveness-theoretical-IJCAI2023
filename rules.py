import pandas as pd
import numpy as np
import tqdm
from trueskill import Rating, rate_1vs1
import math
from itertools import combinations, permutations
import scipy, numpy, math
from sklearn.utils import shuffle


def calculate_number_of_rows_for_complete_information(number_proposals, number_preferences):
    return scipy.math.factorial(number_proposals)/(scipy.math.factorial(number_preferences)*scipy.math.factorial(number_proposals - number_preferences))


def imperfect_population(n_individuals, all_rankings):
    # Impartial culture with repetition
    ids = 0
    output = []
    for times in range(n_individuals):
        df_tmp = pd.DataFrame(list(combinations(all_rankings[times % len(all_rankings)], 2)), \
                              columns=["option_a", "option_b"])
        df_tmp["uuid"] = ids
        output.append(df_tmp)
        ids = ids + 1
        all_rankings = shuffle(all_rankings)
    return output
        
def perfect_population(n_times, all_rankings):
    # Impartial culture
    ids = 0
    output = []
    for times in range(n_times):
        for each_ranking in all_rankings:
            df_tmp = pd.DataFrame(list(combinations(each_ranking, 2)), \
                                  columns=["option_a", "option_b"])
            df_tmp["uuid"] = ids
            output.append(df_tmp)
            ids = ids + 1
    return output

def data_transform_with_pairwise(individuals, 
                                 all_rankings, 
                                 number_individuals, 
                                 number_proposals, 
                                 number_rows=10, 
                                 number_preferences=2):
    
    proposals = list(range(0, number_proposals))
    for individual_id in range(number_individuals):
        options = shuffle([[x,y] for x, y in combinations(proposals, number_preferences)])
        options = shuffle(options)
        option = options[0]
        ranking = all_rankings[individual_id]
        selected_proposal = option[0] if ranking.index(option[0]) < ranking.index(option[1]) else option[1]
        individuals.append([individual_id,\
                            option[0],\
                            option[1],\
                            selected_proposal])  
        options = options[1:]
        if (options == None) or (len(options) == 0):
            break
    return individuals

def data_transform(all_rankings):
        # Impartial culture
    ids = 0
    output = []
    for each_ranking in all_rankings:
        df_tmp = pd.DataFrame(list(combinations(each_ranking, 2)), \
                              columns=["option_a", "option_b"])
        df_tmp["uuid"] = ids
        output.append(df_tmp)
        ids = ids + 1
    return output
    

def using_preflib(number_proposals, number_individuals,type_='IC'):
    from preflibtools.instances import OrdinalInstance
    instance = OrdinalInstance()
#     5 voters and 10 alternatives
    # instance.populate_mallows_mix(5, 10, 3)
    # instance.populate_urn(5, 10, 76)
    # instance.populate_IC(5, 10)
    # instance.populate_IC_anon(5, 10)
    from preflibtools.properties import borda_scores, has_condorcet
    if type_ == 'IC':
        instance.populate_IC(number_individuals, number_proposals)
    elif type_ == 'UM10':
        instance.populate_urn(number_individuals, number_proposals, (math.factorial(number_proposals)/9))
    elif type_ == 'UM50':
        instance.populate_urn(number_individuals, number_proposals, (math.factorial(number_proposals)))
    return list([[y[0] for y in x] for x in instance.full_profile()])

def standard_deviation_rankings(all_rankings):
#     np.array([[10,1],[5,2],[1,3]])[:,1])
    aux1 = pd.DataFrame(all_rankings).melt().groupby('value')['variable'].std().reset_index()
    aux1.columns = ['id','value']
    aux1['rank'] = aux1['value'].rank(method='average', ascending=False)
    return aux1

def win_rate(df):
    dd = df.groupby(["option_source", "option_target"]).agg(
        {"uuid": "count"}).reset_index()
    m = dd.pivot(index="option_source", columns="option_target",
                 values="uuid").fillna(0)
    ids = set(df["option_source"]) | set(df["option_target"])
    m = m.reindex(ids)
    m = m.reindex(ids, axis=1)
    m = m.fillna(0)

    r = m + m.T
    values = m.sum() / r.sum()

    return pd.DataFrame(values).reset_index().rename(columns={"option_target": "id", 0: "value"})


def copeland(df):
    m = matrix_pairs(df) > 0.5
    m = m.astype(float)
    np.fill_diagonal(m.values, np.nan)

    return pd.DataFrame([(a, b) for a, b in list(zip(list(m), np.nanmean(m, axis=0)))], columns=["id", "value"])


def matrix_pairs(df):
    dd = df.groupby(["option_source", "option_target"]).agg(
        {"uuid": "count"}).reset_index()
    m = dd.pivot(index="option_source", columns="option_target",
                 values="uuid").fillna(0)
    ids = set(df["option_source"]) | set(df["option_target"])
    m = m.reindex(ids)
    m = m.reindex(ids, axis=1)
    m = m.fillna(0)

    r = m + m.T
    win_rate = r / r.sum()
    return m / r


def trueskill(data):
    df = data[["option_a", "option_b", "selected"]].copy()
    df = df.dropna()
    for col in df.columns:
        df[col] = df[col].astype("int")

    # Get IDs and initialize values
    all_ids = sorted(list(dict.fromkeys(
        [i for i in df["option_a"].unique()] + [i for i in df["option_b"].unique()])))
    mu = len(all_ids) / 2
    sigma = len(all_ids) / 6

    # Ratings initialization
    ratings = {_id: Rating(mu, sigma) for _id in all_ids}
    score = {_id: {"won": 0, "lost": 0} for _id in all_ids}

    df = df[["option_a", "option_b", "selected"]].rename(
        columns={"option_a": "left", "option_b": "right", "selected": "win"}
    )

    games = [
        (left, right, win)
        for (left, right, win) in zip(df["left"], df["right"], df["win"])
    ]

    for (left, right, win) in games:
        if win == left:
            ratings[left], ratings[right] = rate_1vs1(
                ratings[left], ratings[right])
            score[left]["won"] += 1
            score[right]["lost"] += 1
        elif win == right:
            ratings[right], ratings[left] = rate_1vs1(
                ratings[right], ratings[left])
            score[right]["won"] += 1
            score[left]["lost"] += 1
        elif win == 0:
            ratings[left], ratings[right] = rate_1vs1(
                ratings[left], ratings[right], drawn=True)

    df = pd.DataFrame({"id": all_ids})
    df["mu"] = [ratings[_id].mu for _id in all_ids]
    df["sigma"] = [ratings[_id].sigma for _id in all_ids]
    df["value"] = df["mu"] - 3 * df["sigma"]
    # df["won"] = [score[_id]["won"] for _id in all_ids]
    # df["lost"] = [score[_id]["lost"] for _id in all_ids]
    # df["score"] = df["won"] - df["lost"]

    df = df.sort_values(by=["value"], ascending=False)
    df["rank"] = list(range(1, len(all_ids) + 1))

    # df = df[["id", "skill", "rank"]].sort_values(by="id")
    # print(df) # FOR TESTING

    return df



def divisiveness_copeland(df):
    return divisiveness(df, method=copeland, voter="uuid", full=False, progress=True)


def get_condorcet_efficiency(data):
    N = data.shape[1]

    # Generates expected matrix
    expected_matrix = np.zeros((N, N))
    expected_matrix[np.triu_indices(N)] = 1
    np.fill_diagonal(expected_matrix, np.nan)

    # Generates matrix with the winner
    winner_matrix = data > data.T
    winner_matrix = winner_matrix.astype(float)
    np.fill_diagonal(winner_matrix.values, np.nan)

    # Generates Condorcet Efficiency matrix
    condorcet_eff = winner_matrix == expected_matrix
    condorcet_eff = condorcet_eff.astype(float)
    np.fill_diagonal(condorcet_eff.values, np.nan)

    eff = np.sum(condorcet_eff).sum() / (N * (N - 1))
    print("Condorcet Efficiency:", eff)
    return eff


def matrix_pairs(df):
    dd = df.groupby(["option_source", "option_target"]).agg(
        {"uuid": "count"}).reset_index()
    m = dd.pivot(index="option_source", columns="option_target",
                 values="uuid").fillna(0)
    ids = set(df["option_source"]) | set(df["option_target"])
    m = m.reindex(ids)
    m = m.reindex(ids, axis=1)
    m = m.fillna(0)

    r = m + m.T
    win_rate = r / r.sum()
    return m / r

    return pd.DataFrame(win_rate).reset_index().rename(columns={"option_target": "id", 0: "value"})


def divisiveness_elo(df):
    return divisiveness(df, method=elo, voter="uuid", full=False, progress=True)


def divisiveness(df, method=win_rate, voter="uuid", full=False, progress=True):
    """
    Calculates divisiveness measure
    """
    selected = "selected"
    option_a = "option_a"
    option_b = "option_b"
    candidate = "id"
    df = df[(df[option_a] == df[selected]) | (
        df[option_b] == df[selected])].copy()

    dd = df.groupby(["card_id", selected, voter]).agg({"id": "count"})
    _data = df.copy().set_index(voter)

    _ = method

    def _f(idx, df_select):
        card_id = idx[0]
        s = idx[1]
        users = [item[2] for item in df_select.index.to_numpy()]

        data_temp = _data.loc[users]

        r_tmp = _(data_temp.reset_index()).dropna()
        r_tmp["card_id"] = card_id
        r_tmp[selected] = s

        del data_temp, users

        return r_tmp

    tmp_list = []

    _data_tmp = dd.groupby(level=[0, 1])

    _iter = tqdm.tqdm(_data_tmp, position=0,
                      leave=True) if progress else _data_tmp

    for idx, df_select in _iter:
        tmp_list.append(_f(idx, df_select))

    tmp = pd.concat(tmp_list, ignore_index=True)

    tmp[[f"{option_a}_sorted", f"{option_b}_sorted"]
        ] = tmp["card_id"].str.split("_", expand=True)
    tmp["group"] = tmp[f"{option_a}_sorted"].astype(
        str) == tmp[selected].astype(str)
    tmp["group"] = tmp["group"].replace({True: "A", False: "B"})

    tmp_a = tmp[tmp["group"] == "A"]
    tmp_b = tmp[tmp["group"] == "B"]

    tmp_dv = pd.merge(tmp_a, tmp_b, on=[
                      "card_id", candidate, f"{option_a}_sorted", f"{option_b}_sorted"])
    tmp_dv = tmp_dv[[candidate, "card_id", "value_x",
                     "value_y", f"{selected}_x", f"{selected}_y"]]
    tmp_dv["value"] = abs(tmp_dv["value_x"] - tmp_dv["value_y"])

    tmp_frag_a = tmp_dv[[candidate, f"{selected}_x", "value"]].rename(
        columns={f"{selected}_x": "selected"})
    tmp_frag_b = tmp_dv[[candidate, f"{selected}_y", "value"]].rename(
        columns={f"{selected}_y": "selected"})
    tmp_frag_c = pd.concat([tmp_frag_a, tmp_frag_b])
    tmp_frag_c = tmp_frag_c[tmp_frag_c[candidate]
                            == tmp_frag_c["selected"]]
    tmp_frag_c = tmp_frag_c.groupby(candidate).agg(
        {"value": "mean"}).reset_index()

    return tmp_frag_c


def balance_data(df, random_state=0):
    vmin = df.groupby("card_id").agg(
        {"id": "count"}).sort_values("id")["id"].min()

    output = []
    for i, df_tmp in df.groupby("card_id"):
        output.append(df_tmp.sample(vmin, random_state=random_state))

    tmp = pd.concat(output).reset_index(drop=True)
    return tmp


def bootstrap(df, method, iterations=1, aggregate=False, rank=True, rank_column="rank"):
    output = []
    for random_state in range(iterations):
        tmp = balance_data(df, random_state)
        tmp = method(tmp)
        tmp["random_state"] = random_state
        output.append(tmp)

    tmp = pd.concat(output).reset_index(drop=True)

    if aggregate:
        tmp = tmp.groupby("id").agg(
            {"value": ["mean", "std", "count"]}).reset_index()
        tmp.columns = ["id", "value", "std", "n"]
        v = 1.96 * tmp["std"] / tmp["n"].apply(np.sqrt)
        tmp["uci"] = tmp["value"] + v
        tmp["lci"] = tmp["value"] - v

    if rank:
        tmp = tmp.sort_values("value", ascending=False)
        tmp[rank_column] = range(1, tmp.shape[0] + 1)

    tmp = tmp.reset_index(drop=True)

    return tmp


def flip_answers(flip_df, number_flips):
    flip_df['aux'] = flip_df['option_target'].copy()
    getids = np.random.choice(len(flip_df), number_flips, replace=False)
    flip_df.loc[flip_df.index.isin(getids), 'option_target'] = flip_df[flip_df.index.isin(getids)]['option_source'].values
    flip_df.loc[flip_df.index.isin(getids), 'option_source'] = flip_df[flip_df.index.isin(getids)]['aux'].values
    flip_df.loc[flip_df.index.isin(getids), 'option_selected'] = flip_df[flip_df.index.isin(getids)]['option_selected'].values*(-1)
    flip_df.loc[flip_df.index.isin(getids), 'selected'] = flip_df[flip_df.index.isin(getids)]['option_target'].values 
    return flip_df
    

def bootstrap3(df, method, iterations=1, aggregate=False, rank=True, rank_column="rank", manipulation=False, perc_flips=0.1):
    
    output = []
    for random_state in range(iterations):
        if iterations > 1:
            tmp = df.sample(int(df.shape[0] / 2), random_state=random_state, replace=False)
        else:
            tmp = df.copy()
        if manipulation:
            tmp = flip_answers(tmp, number_flips=(int(perc_flips*len(tmp))))
        tmp = method(tmp)
        tmp["random_state"] = random_state
        output.append(tmp)

    tmp = pd.concat(output).reset_index(drop=True)

    if aggregate:
        tmp = tmp.groupby("id").agg(
            {"value": ["mean", "std", "count"]}).reset_index()
        tmp.columns = ["id", "value", "std", "n"]
        v = 1.96 * tmp["std"] / tmp["n"].apply(np.sqrt)
        tmp["uci"] = tmp["value"] + v
        tmp["lci"] = tmp["value"] - v

    if rank:
        tmp = tmp.sort_values("value", ascending=False)
        tmp[rank_column] = range(1, tmp.shape[0] + 1)

    tmp = tmp.reset_index(drop=True)

    return tmp


def bootstrap4(df, method, iterations=1, aggregate=False, rank=True, rank_column="rank"):
    output = []
    for random_state in range(iterations):
        tmp = df.sample(frac=1).copy()
        tmp = method(tmp)
        tmp["random_state"] = random_state
        output.append(tmp)

    tmp = pd.concat(output).reset_index(drop=True)

    if aggregate:
        tmp = tmp.groupby("id").agg(
            {"value": ["mean", "std", "count"]}).reset_index()
        tmp.columns = ["id", "value", "std", "n"]
        v = 1.96 * tmp["std"] / tmp["n"].apply(np.sqrt)
        tmp["uci"] = tmp["value"] + v
        tmp["lci"] = tmp["value"] - v

    if rank:
        tmp = tmp.sort_values("value", ascending=False)
        tmp[rank_column] = tmp['value'].rank(method='average', ascending=False)
#         tmp[rank_column] = range(1, tmp.shape[0] + 1)

    tmp = tmp.reset_index(drop=True)

    return tmp

def bootstrap2(df, method, iterations=1, aggregate=False, rank=True, rank_column="rank"):
    output = []
    for random_state in range(iterations):
        if iterations > 1:
            tmp = df.sample(int(df.shape[0] / 2), random_state=random_state)
        else:
            tmp = df.copy()
        tmp = method(tmp)
        tmp["random_state"] = random_state
        output.append(tmp)

    tmp = pd.concat(output).reset_index(drop=True)

    if aggregate:
        tmp = tmp.groupby("id").agg(
            {"value": ["mean", "std", "count"]}).reset_index()
        tmp.columns = ["id", "value", "std", "n"]
        v = 1.96 * tmp["std"] / tmp["n"].apply(np.sqrt)
        tmp["uci"] = tmp["value"] + v
        tmp["lci"] = tmp["value"] - v

    if rank:
        tmp = tmp.sort_values("value", ascending=False)
        tmp[rank_column] = range(1, tmp.shape[0] + 1)

    tmp = tmp.reset_index(drop=True)

    return tmp


def elo(df, DEFAULT_RATING=400, K=10):
    ELO_RATING = {i: DEFAULT_RATING for i in set(
        df["option_a_sorted"]) | set(df["option_b_sorted"])}

    for option_a, option_b, selected in list(zip(df["option_a_sorted"], df["option_b_sorted"], df["option_selected"])):
        r_a = ELO_RATING[option_a]
        r_b = ELO_RATING[option_b]

        q_a = 10 ** (r_a / 400)
        q_b = 10 ** (r_b / 400)

        e_a = q_a / (q_a + q_b)
        e_b = q_b / (q_a + q_b)

        if selected == 0:
            s_a = 0.5
            s_b = 0.5
        else:
            is_a_selected = selected == 1
            s_a = 1 if is_a_selected else 0
            s_b = 1 - s_a

        ELO_RATING[option_a] = r_a + K * (s_a - e_a)
        ELO_RATING[option_b] = r_b + K * (s_b - e_b)

    df_elo = pd.DataFrame(ELO_RATING.items(), columns=[
                          "id", "value"]).sort_values("value", ascending=False)
    df_elo["ranking"] = [i + 1 for i in range(df_elo.shape[0])]

    return df_elo


def ahp(df):
    dd = df.groupby(["option_source", "option_target"]).agg(
        {"uuid": "count"}).reset_index()

    a = dd.pivot(index="option_target", columns="option_source", values="uuid")
    b = np.divide(a, a.T)
    np.fill_diagonal(b.values, 1)
    c = b / b.sum()
    weight = [1/c.shape[0] for i in range(c.shape[0])]

    df_ahp = pd.DataFrame(np.sum(np.multiply(c, weight), axis=1)).reset_index(
    ).rename(columns={"option_target": "id", 0: "value"})
    df_ahp = df_ahp.sort_values("value", ascending=False)
    df_ahp["ranking"] = [i + 1 for i in range(df_ahp.shape[0])]

    return df_ahp
