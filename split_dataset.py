"""
==========================================================
split_dataset.py
Author : Md. Jakaria Masud

Project : NeuroOrder

Split Dataset (Leakage-Free)

80% Train
10% Validation
10% Test

Game-level split is used to prevent data leakage.
==========================================================
"""

import pandas as pd
from sklearn.model_selection import train_test_split


# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

INPUT_FILE = "self_play_data.csv"


# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

data = pd.read_csv(INPUT_FILE)

print("Total Samples :", len(data))


# ----------------------------------------------------
# Get Unique Game IDs
# ----------------------------------------------------

game_ids = data["game_id"].unique()

print("Total Games :", len(game_ids))


# ----------------------------------------------------
# 80% Train
# 20% Temporary
# ----------------------------------------------------

train_games, temp_games = train_test_split(

    game_ids,

    test_size=0.20,

    random_state=42,

    shuffle=True

)


# ----------------------------------------------------
# Split Remaining 20%
#
# 10% Validation
# 10% Test
# ----------------------------------------------------

validation_games, test_games = train_test_split(

    temp_games,

    test_size=0.50,

    random_state=42,

    shuffle=True

)


# ----------------------------------------------------
# Select Rows by Game ID
# ----------------------------------------------------

train_data = data[

    data["game_id"].isin(train_games)

]

validation_data = data[

    data["game_id"].isin(validation_games)

]

test_data = data[

    data["game_id"].isin(test_games)

]


# ----------------------------------------------------
# Remove game_id
#
# It is only needed for splitting.
# The model should not use it as a feature.
# ----------------------------------------------------

train_data = train_data.drop(columns=["game_id"])

validation_data = validation_data.drop(columns=["game_id"])

test_data = test_data.drop(columns=["game_id"])


# ----------------------------------------------------
# Save Files
# ----------------------------------------------------

train_data.to_csv(

    "train.csv",

    index=False

)

validation_data.to_csv(

    "validation.csv",

    index=False

)

test_data.to_csv(

    "test.csv",

    index=False

)


# ----------------------------------------------------
# Information
# ----------------------------------------------------

print("\nDataset Split Completed Successfully\n")

print("Train Samples      :", len(train_data))

print("Validation Samples :", len(validation_data))

print("Test Samples       :", len(test_data))

print("\nGenerated Files")

print("---------------------------")

print("train.csv")

print("validation.csv")

print("test.csv")