# %%
# Comparison
import numpy as np
import plotly.express as px
import pandas as pd


# %%
def median_grade(x, n_mods=5):
    """
    Inputs:
    x: array of grades
    n_mods: number of modalities
    Output:
    alpha: median graden
    p: proportion of grades above alpha
    q: proportion of grades below alpha
    y: proportion of grades for each modality
    """
    n_votes = len(x)
    y = np.bincount(x, minlength=n_mods + 1)[1:] / n_votes
    z = np.cumsum(y)
    for alpha, quantile in enumerate(z):
        if quantile >= 0.5:
            break
    p = 1 - z[alpha]
    q = z[max(alpha - 1, 0)]
    return alpha + 1, p, q, y


def usual_judgement(alpha, p, q):
    return alpha + 0.5 * (p - q) / (1 - p - q)


def majority_judgement(alpha, p, q):
    if p > q:
        return alpha + p
    else:
        return alpha - q


def score_computation(df_raw, n_mods=5, tie="Majority Judgement"):
    n_cand = len(df_raw.columns)
    proportions = np.zeros((n_cand, n_mods))
    scores = np.zeros(n_cand)

    for i, name in enumerate(df_raw.columns):
        alpha, p, q, y = median_grade(df_raw[name].dropna(), n_mods=n_mods)
        proportions[i, :] = y
        # print(alpha, p, q)
        if tie == "Usual Judgement":
            scores[i] = usual_judgement(alpha, p, q)
        elif tie == "Majority Judgement":
            scores[i] = majority_judgement(alpha, p, q)
    df = pd.DataFrame(proportions)
    df.columns = np.arange(n_mods) + 1
    df.index = df_raw.columns
    df["scores"] = scores
    return df


def plot_vote_synthesis(df, n_mods, tie):
    discr = np.linspace(0.1, 0.9, n_mods)
    colors = px.colors.sample_colorscale("RdBu", discr)

    df_plot = df.copy()
    df_plot = df_plot.sort_values(by="scores", ascending=True)
    # remove column scores in df_plot
    df_plot = df_plot.drop(columns=["scores"])
    columns = [str(i + 1) for i in range(n_mods)]
    # display same graph horizontally:
    fig = px.bar(
        df_plot, orientation="h", barmode="relative", color_discrete_sequence=colors
    )
    fig.update_layout(
        xaxis_title="Proportion of grades",
        yaxis_title="Candidates",
        legend_title="Notes",
        title="Comparison of candidates: <br>ordered by scores" + f" ({tie})",
        font=dict(size=12),
    )

    fig.show()
    print(df)


# Random example
# n_mods = 3
# n_cand = 4
# n_votes = 100
# x = np.random.randint(1, n_mods + 1, n_votes * n_cand).reshape(n_cand, n_votes)
# x[2:, 0:25] = 1
# cand_names = [f"Candidate {i+1}" for i in range(n_cand)]


# # wikipedia almost example
# n_cand = 4
# n_mods = 3
# n_votes = 10

# x = np.array(
#     [
#         [1, 2, 2, 2, 2, [f"Candidate {i+1}" for i in range(n_cand)]2, 2, 3, 3, 3],
#         [1, 2, 2, 2, 2, 2, 2, 2, 2, 3],
#         [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
#         [1, 1, 1, 1, 2, 3, 3, 3, 3, 3],
#     ]
# )
# cand_names = [f"Candidate {i+1}" for i in range(n_cand)]
#

# # lime survey example
#  import in a dataframe with integer values
# %%
n_mods = 5
tie = "Majority Judgement"

df_raw = pd.read_csv("results-survey-location.csv", dtype="Int64")
df_raw.set_index("Response ID", inplace=True)
df_raw.columns = ["St-Eloi", "St-Priest", "MTD"]
df = score_computation(df_raw, n_mods=5, tie=tie)
plot_vote_synthesis(df, n_mods, tie)
# %%
df_raw = pd.read_csv("results-survey-time.csv", dtype="Int64")
df_raw.set_index("Response ID", inplace=True)
df_raw.columns = [
    "Wednesday : 09:30 AM - 10:30 AM",
    "Wednesday : 10:00 AM - 11:00 AM",
    "Wednesday : 10:30 AM - 11:30 AM",
    "Wednesday : 11:00 AM - 12:00 AM",
    "Wednesday : 01:00 PM - 02:00 PM",
    "Wednesday : 01:30 PM - 02:30 PM",
    "Wednesday : 02:00 PM - 03:00 PM",
    "Wednesday : 02:30 PM - 03:30 PM",
    "Wednesday : 03:00 PM - 04:00 PM",
    "Wednesday : 03:30 PM - 04:30 PM",
    "Thursday : 09:30 AM - 10:30 AM",
    "Thursday : 10:00 AM - 11:00 AM",
    "Thursday : 10:30 AM - 11:30 AM",
    "Thursday : 11:00 AM - 12:00 AM",
    "Thursday : 01:00 PM - 02:00 PM",
    "Thursday : 01:30 PM - 02:30 PM",
    "Thursday : 02:00 PM - 03:00 PM",
    "Thursday : 02:30 PM - 03:30 PM",
    "Thursday : 03:00 PM - 04:00 PM",
    "Thursday : 03:30 PM - 04:30 PM",
]
df = score_computation(df_raw, n_mods=5, tie=tie)
plot_vote_synthesis(df, n_mods, tie)
# %%
