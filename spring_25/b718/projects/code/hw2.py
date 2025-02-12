"""Python code for HW 2."""

import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import pandas as pd  # noqa
import statsmodels.formula.api as smf  # type: ignore

# Load in the data
imt = pd.read_excel("data/imtdata.xlsx")

# ----------
# Question 1
# ----------

# Part A
# ------

# Post model
# Model to specify: imt1 = \beta_0 + \beta_1 trt + \epsilon

# First prepare the features, noting that 'trt' needs to be converted to
# a catgorical variable
imt["trt"] = imt["trt"].astype("category")

# Build the post model
post_model = smf.ols("imt1 ~ trt", data=imt).fit()

# Change model
# Model to specify: imt1 - imt0 = \beta_0 + \beta_1 trt + \epsilon

# Let's build the features and targets for this model. Note that
# the features for the post and the change are identical
imt["y_change"] = imt["imt1"] - imt["imt0"]


# Build the change model
change_model = smf.ols("y_change ~ trt", data=imt).fit()

# ANCOVA model
# Model to specify: imt1 = \beta_0 + \beta_1 trt + \beta_2 imt0 + \epsion

# No new features need to be added to this model, so our existing DF is
# in good shape

# Build the ANCOVA model
ancova_model = smf.ols("imt1 ~ trt + imt0", data=imt).fit()

# Coefficient table

# Post
post_df = pd.DataFrame(post_model.params)
post_df["std.err"] = post_model.bse
post_df["lower"] = post_df.iloc[:, 0] - 1.96 * post_df["std.err"]
post_df["upper"] = post_df.iloc[:, 0] + 1.96 * post_df["std.err"]
post_df["95.pct.interval"] = post_df.apply(
    lambda row: f"[{round(row['lower'], 3)}, {round(row['upper'], 3)}]", axis=1
)
post_df.drop(columns=["lower", "upper"], inplace=True)
post_df["pvalues"] = post_model.pvalues
post_df["model"] = "Post"
post_df.reset_index(inplace=True)
post_df.columns = pd.Index(
    [
        "Term",
        "Estimate",
        "Standard Error",
        "95% Confidence",
        "p-value",
        "Model",
    ]
)

# Change
change_df = pd.DataFrame(change_model.params)
change_df["std.err"] = change_model.bse
change_df["lower"] = change_df.iloc[:, 0] - 1.96 * change_df["std.err"]
change_df["upper"] = change_df.iloc[:, 0] + 1.96 * change_df["std.err"]
change_df["95.pct.interval"] = change_df.apply(
    lambda row: f"[{round(row['lower'], 3)}, {round(row['upper'], 3)}]", axis=1
)
change_df.drop(columns=["lower", "upper"], inplace=True)
change_df["pvalues"] = change_model.pvalues
change_df["model"] = "Change"
change_df.reset_index(inplace=True)
change_df.columns = pd.Index(
    [
        "Term",
        "Estimate",
        "Standard Error",
        "95% Confidence",
        "p-value",
        "Model",
    ]
)

# ANCOVA
ancova_df = pd.DataFrame(ancova_model.params)
ancova_df["std.err"] = ancova_model.bse
ancova_df["lower"] = ancova_df.iloc[:, 0] - 1.96 * ancova_df["std.err"]
ancova_df["upper"] = ancova_df.iloc[:, 0] + 1.96 * ancova_df["std.err"]
ancova_df["95.pct.interval"] = ancova_df.apply(
    lambda row: f"[{round(row['lower'], 3)}, {round(row['upper'], 3)}]", axis=1
)
ancova_df.drop(columns=["lower", "upper"], inplace=True)
ancova_df["pvalues"] = ancova_model.pvalues
ancova_df["model"] = "ANCOVA"
ancova_df.reset_index(inplace=True)
ancova_df.columns = pd.Index(
    [
        "Term",
        "Estimate",
        "Standard Error",
        "95% Confidence",
        "p-value",
        "Model",
    ]
)

# Full table
coefficient_table = pd.concat([post_df, change_df, ancova_df])


# Part B
# ------

# Post model
# Model to specify: imt1 = \beta_0 + \beta_1 male + \epsilon

# First prepare the features, noting that 'male' needs to be converted to
# a catgorical variable
imt["male"] = imt["male"].astype("category")

# Build the post model
post_model = smf.ols("imt1 ~ male", data=imt).fit()

# Change model
# Model to specify: imt1 - imt0 = \beta_0 + \beta_1 male + \epsilon

# Let's build the features and targets for this model. Note that
# the features for the post and the change are identical
imt["y_change"] = imt["imt1"] - imt["imt0"]


# Build the change model
change_model = smf.ols("y_change ~ male", data=imt).fit()

# ANCOVA model
# Model to specify: imt1 = \beta_0 + \beta_1 male + \beta_2 imt0 + \epsion

# No new features need to be added to this model, so our existing DF is
# in good shape

# Build the ANCOVA model
ancova_model = smf.ols("imt1 ~ male + imt0", data=imt).fit()

# Coefficient table

# Post
post_df = pd.DataFrame(post_model.params)
post_df["std.err"] = post_model.bse
post_df["lower"] = post_df.iloc[:, 0] - 1.96 * post_df["std.err"]
post_df["upper"] = post_df.iloc[:, 0] + 1.96 * post_df["std.err"]
post_df["95.pct.interval"] = post_df.apply(
    lambda row: f"[{round(row['lower'], 3)}, {round(row['upper'], 3)}]", axis=1
)
post_df.drop(columns=["lower", "upper"], inplace=True)
post_df["pvalues"] = post_model.pvalues
post_df["model"] = "Post"
post_df.reset_index(inplace=True)
post_df.columns = pd.Index(
    [
        "Term",
        "Estimate",
        "Standard Error",
        "95% Confidence",
        "p-value",
        "Model",
    ]
)

# Change
change_df = pd.DataFrame(change_model.params)
change_df["std.err"] = change_model.bse
change_df["lower"] = change_df.iloc[:, 0] - 1.96 * change_df["std.err"]
change_df["upper"] = change_df.iloc[:, 0] + 1.96 * change_df["std.err"]
change_df["95.pct.interval"] = change_df.apply(
    lambda row: f"[{round(row['lower'], 3)}, {round(row['upper'], 3)}]", axis=1
)
change_df.drop(columns=["lower", "upper"], inplace=True)
change_df["pvalues"] = change_model.pvalues
change_df["model"] = "Change"
change_df.reset_index(inplace=True)
change_df.columns = pd.Index(
    [
        "Term",
        "Estimate",
        "Standard Error",
        "95% Confidence",
        "p-value",
        "Model",
    ]
)

# ANCOVA
ancova_df = pd.DataFrame(ancova_model.params)
ancova_df["std.err"] = ancova_model.bse
ancova_df["lower"] = ancova_df.iloc[:, 0] - 1.96 * ancova_df["std.err"]
ancova_df["upper"] = ancova_df.iloc[:, 0] + 1.96 * ancova_df["std.err"]
ancova_df["95.pct.interval"] = ancova_df.apply(
    lambda row: f"[{round(row['lower'], 3)}, {round(row['upper'], 3)}]", axis=1
)
ancova_df.drop(columns=["lower", "upper"], inplace=True)
ancova_df["pvalues"] = ancova_model.pvalues
ancova_df["model"] = "ANCOVA"
ancova_df.reset_index(inplace=True)
ancova_df.columns = pd.Index(
    [
        "Term",
        "Estimate",
        "Standard Error",
        "95% Confidence",
        "p-value",
        "Model",
    ]
)

# Full table
coefficient_table = pd.concat([post_df, change_df, ancova_df])
