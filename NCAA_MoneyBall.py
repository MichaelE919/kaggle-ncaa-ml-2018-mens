"""
Two-part script.

Uses Pandas to first create a prediction dataset then Scikit-Learn to create
several machine learning models.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from xgboost.sklearn import XGBClassifier

print('Loading data...')

df_tourney = pd.read_csv('DataFiles/NCAATourneyCompactResults.csv')
df_season = pd.read_csv('DataFiles/RegularSeasonDetailedResults.csv')
df_teams = pd.read_csv('DataFiles/Teams.csv')
df_kenpom = pd.read_csv('DataFiles/KenPom.csv')
df_seeds = pd.read_csv('DataFiles/NCAATourneySeeds.csv')
df_rankings = pd.read_csv('DataFiles/MasseyOrdinals_thruSeason2018_Day128.csv')
df_sample_sub = pd.read_csv('DataFiles/SampleSubmissionStage2.csv')

df_kenpom['TeamID'] = df_kenpom['TeamName'].apply(
    lambda x: df_teams[df_teams['TeamName'] == x].values[0][0])

print('Creating helper columns...')

wPos = df_season.apply(
    lambda row: 0.96 * (row.WFGA + row.WTO + 0.44 * row.WFTA - row.WOR),
    axis=1)
lPos = df_season.apply(
    lambda row: 0.96 * (row.LFGA + row.LTO + 0.44 * row.LFTA - row.LOR),
    axis=1)

df_season['POSS'] = (wPos + lPos) / 2

wPIE = df_season.apply(
    lambda row: row.WScore + row.WFGM + row.WFTM - row.WFGA - row.WFTA +
    row.WDR + 0.5 * row.WOR + row.WAst + row.WStl + 0.5 * row.WBlk - row.WPF -
    row.WTO,
    axis=1)
lPIE = df_season.apply(
    lambda row: row.LScore + row.LFGM + row.LFTM - row.LFGA - row.LFTA +
    row.LDR + 0.5 * row.LOR + row.LAst + row.LStl + 0.5 * row.LBlk - row.LPF -
    row.LTO,
    axis=1)

df_season['WPIE'] = wPIE / (wPIE + lPIE)
df_season['LPIE'] = lPIE / (wPIE + lPIE)

# Effective Field Goal Percentage =
# (Field Goals Made + 0.5*3P Field Goals Made) / Field Goal Attempts

print('Creating columns for the Four Factors...')
print('\tEffective Field Goal Percentage')

df_season['WeFGP'] = df_season.apply(
    lambda row: (row.WFGM + 0.5 * row.WFGM3) / row.WFGA, axis=1)
df_season['LeFGP'] = df_season.apply(
    lambda row: (row.LFGM + 0.5 * row.LFGM3) / row.LFGA, axis=1)

# Turnover Rate =
# Turnovers/(Field Goal Attempts + 0.44*Free Throw Attempts + Turnovers)
print('\tTurnover Rate')
df_season['WToR'] = df_season.apply(
    lambda row: row.WTO / (row.WFGA + 0.44 * row.WFTA + row.WTO), axis=1)
df_season['LToR'] = df_season.apply(
    lambda row: row.LTO / (row.LFGA + 0.44 * row.LFTA + row.LTO), axis=1)

# Offensive Rebounding Percentage =
# Offensive Rebounds / (Offensive Rebounds + Opponent’s Defensive Rebounds)
print('\tOffensive Rebounding Percentage')
df_season['WORP'] = df_season.apply(
    lambda row: row.WOR / (row.WOR + row.LDR), axis=1)
df_season['LORP'] = df_season.apply(
    lambda row: row.LOR / (row.LOR + row.WDR), axis=1)

# Free Throw Rate =
# Free Throws Attempted / Field Goals Attempted
print('\tFree Throw Rate')
df_season['WFTR'] = df_season.apply(lambda row: row.WFTA / row.WFGA, axis=1)
df_season['LFTR'] = df_season.apply(lambda row: row.LFTA / row.LFGA, axis=1)

# 4 Factors is weighted as follows
# 1. Shooting (40%)
# 2. Turnovers (25%)
# 3. Rebounding (20%)
# 4. Free Throws (15%)

print('Adding weights to the Four Factors...')
df_season['W4Factor'] = df_season.apply(
    lambda row: 0.4 * row.WeFGP + 0.25 * row.WToR + 0.2 * row.WORP + 0.15 *
    row.WFTR,
    axis=1)
df_season['L4Factor'] = df_season.apply(
    lambda row: 0.4 * row.LeFGP + 0.25 * row.LToR + 0.2 * row.LORP + 0.15 *
    row.LFTR,
    axis=1)


def getAdjO(Year, TeamID):
    """Obtain adjusted offensive efficiency."""
    try:
        AdjO = df_kenpom[(df_kenpom['TeamID'] == TeamID)
                         & (df_kenpom['Season'] == Year)].values[0][2]
    except IndexError:
        AdjO = df_kenpom[df_kenpom['TeamID'] == TeamID].mean().values[1].round(
            2)
    return AdjO


def getAdjD(Year, TeamID):
    """Obtain adjusted defensive efficiency."""
    try:
        AdjD = df_kenpom[(df_kenpom['TeamID'] == TeamID)
                         & (df_kenpom['Season'] == Year)].values[0][3]
    except IndexError:
        AdjD = df_kenpom[df_kenpom['TeamID'] == TeamID].mean().values[2].round(
            2)
    return AdjD


def getAdjEM(Year, TeamID):
    """Obtain adjusted efficiency margin."""
    try:
        AdjEM = df_kenpom[(df_kenpom['TeamID'] == TeamID)
                          & (df_kenpom['Season'] == Year)].values[0][4]
    except IndexError:
        AdjEM = df_kenpom[df_kenpom['TeamID'] == TeamID].mean().values[
            3].round(2)
    return AdjEM


print('Creating Adjusted Efficiency columns...')

# Adjusted Offensive Efficiency
print('\tOffensive')
df_season['WAdjO'] = df_season.apply(
    lambda row: getAdjO(row['Season'], row['WTeamID']), axis=1)
df_season['LAdjO'] = df_season.apply(
    lambda row: getAdjO(row['Season'], row['LTeamID']), axis=1)

# Adjusted Defensive Efficiency
print('\tDefensive')
df_season['WAdjD'] = df_season.apply(
    lambda row: getAdjD(row['Season'], row['WTeamID']), axis=1)
df_season['LAdjD'] = df_season.apply(
    lambda row: getAdjD(row['Season'], row['LTeamID']), axis=1)

# Adjusted Efficiency Margin
print('\tMargin')
df_season['WAdjEM'] = df_season.apply(
    lambda row: getAdjEM(row['Season'], row['WTeamID']), axis=1)
df_season['LAdjEM'] = df_season.apply(
    lambda row: getAdjEM(row['Season'], row['LTeamID']), axis=1)

print('Creating remaining columns:')

# Defensive Rebounding Percentage =
# Defensive Rebounds / (Defensive Rebounds + Opponents Offensive Rebounds)
print('\tDefensive Rebounding Percentage')
df_season['WDRP'] = df_season.apply(
    lambda row: row.WDR / (row.WDR + row.LOR), axis=1)
df_season['LDRP'] = df_season.apply(
    lambda row: row.LDR / (row.LDR + row.WOR), axis=1)

# Offensive Rebound to Turnover Margin
print('\tOffensive Rebound to Turnover Margin')
df_season['WORTM'] = df_season.apply(lambda row: row.WOR - row.WTO, axis=1)
df_season['LORTM'] = df_season.apply(lambda row: row.LOR - row.LTO, axis=1)

# Assist Ratio =
# Assists/ (Field Goal Attempts + Free Throw Attempts*0.44 + Assists +
# Turnovers)
print('\tAssist Ratio')
df_season['WAR'] = df_season.apply(
    lambda row: row.WAst / (row.WFGA + row.WFTA * 0.44 + row.WAst + row.WTO),
    axis=1)
df_season['LAR'] = df_season.apply(
    lambda row: row.LAst / (row.LFGA + row.LFTA * 0.44 + row.LAst + row.LTO),
    axis=1)

# Free Throw Percentage
print('\tFree Throw Percentage')
df_season['WFTP'] = df_season.apply(
    lambda row: 0.0 if row.WFTA == 0 else row.WFTM / row.WFTA, axis=1)
df_season['LFTP'] = df_season.apply(
    lambda row: 0.0 if row.LFTA == 0 else row.LFTM / row.LFTA, axis=1)

# Score Differential = Points scored - points allowed
print('\tScore Differential')
df_season['WPtsDf'] = df_season.apply(
    lambda row: row.WScore - row.LScore, axis=1)
df_season['LPtsDf'] = df_season.apply(
    lambda row: row.LScore - row.WScore, axis=1)

df_season.drop(
    labels=[
        'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst',
        'WTO', 'WStl', 'WBlk', 'WPF'
    ],
    axis=1,
    inplace=True)
df_season.drop(
    labels=[
        'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst',
        'LTO', 'LStl', 'LBlk', 'LPF'
    ],
    axis=1,
    inplace=True)

df_season_totals = pd.DataFrame()

print('Creating prediction dataset...')

# Calculate wins and losses to get winning percentage
df_season_totals['Wins'] = df_season['WTeamID'].groupby(
    [df_season['Season'], df_season['WTeamID']]).count()
df_season_totals['Losses'] = df_season['LTeamID'].groupby(
    [df_season['Season'], df_season['LTeamID']]).count()
df_season_totals['WinPCT'] = df_season_totals['Wins'] / (
    df_season_totals['Wins'] + df_season_totals['Losses'])

df_season_totals['WPIE'] = df_season['WPIE'].groupby(
    [df_season['Season'], df_season['WTeamID']]).mean()
df_season_totals['WeFGP'] = df_season['WeFGP'].groupby(
    [df_season['Season'], df_season['WTeamID']]).mean()
df_season_totals['WToR'] = df_season['WToR'].groupby(
    [df_season['Season'], df_season['WTeamID']]).mean()
df_season_totals['WORP'] = df_season['WORP'].groupby(
    [df_season['Season'], df_season['WTeamID']]).mean()
df_season_totals['WFTR'] = df_season['WFTR'].groupby(
    [df_season['Season'], df_season['WTeamID']]).mean()
df_season_totals['W4Factor'] = df_season['W4Factor'].groupby(
    [df_season['Season'], df_season['WTeamID']]).mean()
df_season_totals['WAdjO'] = df_season['WAdjO'].groupby(
    [df_season['Season'], df_season['WTeamID']]).mean()
df_season_totals['WAdjD'] = df_season['WAdjD'].groupby(
    [df_season['Season'], df_season['WTeamID']]).mean()
df_season_totals['WAdjEM'] = df_season['WAdjEM'].groupby(
    [df_season['Season'], df_season['WTeamID']]).mean()
df_season_totals['WDRP'] = df_season['WDRP'].groupby(
    [df_season['Season'], df_season['WTeamID']]).mean()
df_season_totals['WORTM'] = df_season['WORTM'].groupby(
    [df_season['Season'], df_season['WTeamID']]).mean()
df_season_totals['WAR'] = df_season['WAR'].groupby(
    [df_season['Season'], df_season['WTeamID']]).mean()
df_season_totals['WFTP'] = df_season['WFTP'].groupby(
    [df_season['Season'], df_season['WTeamID']]).mean()
df_season_totals['WPtsDf'] = df_season['WPtsDf'].groupby(
    [df_season['Season'], df_season['WTeamID']]).mean()

df_season_totals['LPIE'] = df_season['LPIE'].groupby(
    [df_season['Season'], df_season['LTeamID']]).mean()
df_season_totals['LeFGP'] = df_season['LeFGP'].groupby(
    [df_season['Season'], df_season['LTeamID']]).mean()
df_season_totals['LToR'] = df_season['LToR'].groupby(
    [df_season['Season'], df_season['LTeamID']]).mean()
df_season_totals['LORP'] = df_season['LORP'].groupby(
    [df_season['Season'], df_season['LTeamID']]).mean()
df_season_totals['LFTR'] = df_season['LFTR'].groupby(
    [df_season['Season'], df_season['LTeamID']]).mean()
df_season_totals['L4Factor'] = df_season['L4Factor'].groupby(
    [df_season['Season'], df_season['LTeamID']]).mean()
df_season_totals['LAdjO'] = df_season['LAdjO'].groupby(
    [df_season['Season'], df_season['LTeamID']]).mean()
df_season_totals['LAdjD'] = df_season['LAdjD'].groupby(
    [df_season['Season'], df_season['LTeamID']]).mean()
df_season_totals['LAdjEM'] = df_season['LAdjEM'].groupby(
    [df_season['Season'], df_season['LTeamID']]).mean()
df_season_totals['LDRP'] = df_season['LDRP'].groupby(
    [df_season['Season'], df_season['LTeamID']]).mean()
df_season_totals['LORTM'] = df_season['LORTM'].groupby(
    [df_season['Season'], df_season['LTeamID']]).mean()
df_season_totals['LAR'] = df_season['LAR'].groupby(
    [df_season['Season'], df_season['LTeamID']]).mean()
df_season_totals['LFTP'] = df_season['LFTP'].groupby(
    [df_season['Season'], df_season['LTeamID']]).mean()
df_season_totals['LPtsDf'] = df_season['LPtsDf'].groupby(
    [df_season['Season'], df_season['LTeamID']]).mean()

df_season_totals[
    'PIE'] = df_season_totals['WPIE'] * df_season_totals['WinPCT'] + \
    df_season_totals['LPIE'] * (
        1 - df_season_totals['WinPCT'])
df_season_totals[
    'eFGP'] = df_season_totals['WeFGP'] * df_season_totals['WinPCT'] + \
    df_season_totals['LeFGP'] * (
        1 - df_season_totals['WinPCT'])
df_season_totals[
    'ToR'] = df_season_totals['WToR'] * df_season_totals['WinPCT'] + \
    df_season_totals['LToR'] * (
        1 - df_season_totals['WinPCT'])
df_season_totals[
    'ORP'] = df_season_totals['WORP'] * df_season_totals['WinPCT'] + \
    df_season_totals['LORP'] * (
        1 - df_season_totals['WinPCT'])
df_season_totals[
    'FTR'] = df_season_totals['WFTR'] * df_season_totals['WinPCT'] + \
    df_season_totals['LFTR'] * (
        1 - df_season_totals['WinPCT'])
df_season_totals[
    '4Factor'] = df_season_totals['W4Factor'] * df_season_totals['WinPCT'] + \
    df_season_totals['L4Factor'] * (
        1 - df_season_totals['WinPCT'])
df_season_totals[
    'AdjO'] = df_season_totals['WAdjO'] * df_season_totals['WinPCT'] + \
    df_season_totals['LAdjO'] * (
        1 - df_season_totals['WinPCT'])
df_season_totals[
    'AdjD'] = df_season_totals['WAdjD'] * df_season_totals['WinPCT'] + \
    df_season_totals['LAdjD'] * (
        1 - df_season_totals['WinPCT'])
df_season_totals[
    'AdjEM'] = df_season_totals['WAdjEM'] * df_season_totals['WinPCT'] + \
    df_season_totals['LAdjEM'] * (
        1 - df_season_totals['WinPCT'])
df_season_totals[
    'DRP'] = df_season_totals['WDRP'] * df_season_totals['WinPCT'] + \
    df_season_totals['LDRP'] * (
        1 - df_season_totals['WinPCT'])
df_season_totals[
    'ORTM'] = df_season_totals['WORTM'] * df_season_totals['WinPCT'] + \
    df_season_totals['LORTM'] * (
        1 - df_season_totals['WinPCT'])
df_season_totals[
    'AR'] = df_season_totals['WAR'] * df_season_totals['WinPCT'] + \
    df_season_totals['LAR'] * (
        1 - df_season_totals['WinPCT'])
df_season_totals[
    'FTP'] = df_season_totals['WFTP'] * df_season_totals['WinPCT'] + \
    df_season_totals['LFTP'] * (
        1 - df_season_totals['WinPCT'])
df_season_totals[
    'PtsDf'] = df_season_totals['WPtsDf'] * df_season_totals['WinPCT'] + \
    df_season_totals['LPtsDf'] * (
        1 - df_season_totals['WinPCT'])

df_season_totals.reset_index(inplace=True)

df_season_totals[pd.isnull(df_season_totals['Losses'])]

df_season_totals.loc[4064, 'Losses'] = 0
df_season_totals.loc[4064, 'WinPCT'] = 1
df_season_totals.loc[4064, 'PIE'] = df_season_totals.loc[4064, 'WPIE']
df_season_totals.loc[4064, 'eFGP'] = df_season_totals.loc[4064, 'WeFGP']
df_season_totals.loc[4064, 'ToR'] = df_season_totals.loc[4064, 'WToR']
df_season_totals.loc[4064, 'ORP'] = df_season_totals.loc[4064, 'WORP']
df_season_totals.loc[4064, 'FTR'] = df_season_totals.loc[4064, 'WFTR']
df_season_totals.loc[4064, '4Factor'] = df_season_totals.loc[4064, 'W4Factor']
df_season_totals.loc[4064, 'AdjO'] = df_season_totals.loc[4064, 'WAdjO']
df_season_totals.loc[4064, 'AdjD'] = df_season_totals.loc[4064, 'WAdjD']
df_season_totals.loc[4064, 'AdjEM'] = df_season_totals.loc[4064, 'WAdjEM']
df_season_totals.loc[4064, 'DRP'] = df_season_totals.loc[4064, 'WDRP']
df_season_totals.loc[4064, 'ORTM'] = df_season_totals.loc[4064, 'WORTM']
df_season_totals.loc[4064, 'AR'] = df_season_totals.loc[4064, 'WAR']
df_season_totals.loc[4064, 'FTP'] = df_season_totals.loc[4064, 'WFTP']
df_season_totals.loc[4064, 'PtsDf'] = df_season_totals.loc[4064, 'WPtsDf']

df_season_totals.loc[4211, 'Losses'] = 0
df_season_totals.loc[4211, 'WinPCT'] = 1
df_season_totals.loc[4211, 'PIE'] = df_season_totals.loc[4211, 'WPIE']
df_season_totals.loc[4211, 'eFGP'] = df_season_totals.loc[4211, 'WeFGP']
df_season_totals.loc[4211, 'ToR'] = df_season_totals.loc[4211, 'WToR']
df_season_totals.loc[4211, 'ORP'] = df_season_totals.loc[4211, 'WORP']
df_season_totals.loc[4211, 'FTR'] = df_season_totals.loc[4211, 'WFTR']
df_season_totals.loc[4211, '4Factor'] = df_season_totals.loc[4211, 'W4Factor']
df_season_totals.loc[4211, 'AdjO'] = df_season_totals.loc[4211, 'WAdjO']
df_season_totals.loc[4211, 'AdjD'] = df_season_totals.loc[4211, 'WAdjD']
df_season_totals.loc[4211, 'AdjEM'] = df_season_totals.loc[4211, 'WAdjEM']
df_season_totals.loc[4211, 'DRP'] = df_season_totals.loc[4211, 'WDRP']
df_season_totals.loc[4211, 'ORTM'] = df_season_totals.loc[4211, 'WORTM']
df_season_totals.loc[4211, 'AR'] = df_season_totals.loc[4211, 'WAR']
df_season_totals.loc[4211, 'FTP'] = df_season_totals.loc[4211, 'WFTP']
df_season_totals.loc[4211, 'PtsDf'] = df_season_totals.loc[4211, 'WPtsDf']

df_season_totals.drop(
    labels=[
        'Wins', 'Losses', 'WPIE', 'WeFGP', 'WToR', 'WORP', 'WFTR', 'W4Factor',
        'WAdjO', 'WAdjD', 'WAdjEM', 'WDRP', 'WORTM', 'WAR', 'WFTP', 'WPtsDf',
        'LPIE', 'LeFGP', 'LToR', 'LORP', 'LFTR', 'L4Factor', 'LAdjO', 'LAdjD',
        'LAdjEM', 'LDRP', 'LORTM', 'LAR', 'LFTP', 'LPtsDf'
    ],
    axis=1,
    inplace=True)

columns = df_season_totals.columns.tolist()
columns.pop(2)
columns.append('WinPCT')
df_season_totals = df_season_totals[columns]
df_season_totals.rename(columns={'WTeamID': 'TeamID'}, inplace=True)

df_rpi = df_rankings[df_rankings['SystemName'] == 'RPI']
df_rpi_2018 = df_rpi[df_rpi['Season'] == 2018]
df_rpi_2018_final = df_rpi_2018[df_rpi_2018['RankingDayNum'] == 128]

df_rpi_prev_final = df_rpi[df_rpi['RankingDayNum'] == 133]

df_rpi_final = pd.concat([df_rpi_prev_final, df_rpi_2018_final])
df_rpi_final['Season'].value_counts()

df_rpi_final.drop(labels=['RankingDayNum', 'SystemName'], inplace=True, axis=1)

df_seeds['Seed_new'] = df_seeds['Seed'].apply(lambda x: int(x[1:3]))

df_seeds.drop(labels='Seed', axis=1, inplace=True)
df_seeds.rename(columns={'Seed_new': 'Seed'}, inplace=True)

# Use tourney seeds from 2003 on
df_seeds_final = df_seeds[df_seeds['Season'] > 2002]
df_seeds_final.head()

df_tourney_temp = pd.merge(
    left=df_seeds_final,
    right=df_rpi_final,
    how='left',
    on=['Season', 'TeamID'])
df_tourney_final = pd.merge(
    left=df_tourney_temp,
    right=df_season_totals,
    how='left',
    on=['Season', 'TeamID'])

df_tourney.drop(
    labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'],
    inplace=True,
    axis=1)
df_tourney = df_tourney[df_tourney['Season'] > 2002]
df_tourney.reset_index(drop=True, inplace=True)

df_win_teams = pd.merge(
    left=df_tourney,
    right=df_tourney_final,
    how='left',
    left_on=['Season', 'WTeamID'],
    right_on=['Season', 'TeamID'])
df_win_teams.drop(labels='TeamID', inplace=True, axis=1)

df_loss_teams = pd.merge(
    left=df_tourney,
    right=df_tourney_final,
    how='left',
    left_on=['Season', 'LTeamID'],
    right_on=['Season', 'TeamID'])
df_loss_teams.drop(labels='TeamID', inplace=True, axis=1)

df_win_diff = df_win_teams.iloc[:, 3:] - df_loss_teams.iloc[:, 3:]
df_win_diff['result'] = 1
df_win_diff = pd.merge(
    left=df_win_diff,
    right=df_tourney,
    left_index=True,
    right_index=True,
    how='inner')

df_loss_diff = df_loss_teams.iloc[:, 3:] - df_win_teams.iloc[:, 3:]
df_loss_diff['result'] = 0
df_loss_diff = pd.merge(
    left=df_loss_diff,
    right=df_tourney,
    left_index=True,
    right_index=True,
    how='inner')

prediction_dataset = pd.concat((df_win_diff, df_loss_diff), axis=0)
prediction_dataset.sort_values('Season', inplace=True)

print('Separating dataset into features, labels, and IDs...')

# For stage 1, label these as "labels", "features", and "ID's"
y = prediction_dataset['result']
X = prediction_dataset.loc[:, :'WinPCT'] # pylint: disable=E1127
train_IDs = prediction_dataset.loc[:, 'Season':] # pylint: disable=E1127

# Identify numerical key at bottom row of 2003-2013 set
# len(prediction_dataset[prediction_dataset['Season'] < 2014])

# Create 2014-2017 test set

# test_labels = prediction_dataset['result'][prediction_dataset['Season']>2013]
# test_features = prediction_dataset.iloc[1426:, :15]
# test_IDs = prediction_dataset.iloc[1426:, 16:]

# Create the 2003-2013 training set

# y = prediction_dataset['result'][prediction_dataset['Season'] < 2014]
# X = prediction_dataset.iloc[:1426, :15]
# train_IDs = prediction_dataset.iloc[:1426, 16:]

print('Creating training and test sets...')

# Split the 2003-2013 set even further
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, test_size=0.3, random_state=1, stratify=y)

print('Creating classifier objects, parameter grids, and pipelines...')

# Initiate classifiers
clf1 = LogisticRegression()
clf2 = KNeighborsClassifier()
clf3 = XGBClassifier()
clf4 = DecisionTreeClassifier()
clf5 = RandomForestClassifier()
clf6 = GradientBoostingClassifier()

# Configure Parameter grids
param_grid1 = [{'clf1__C': list(np.logspace(start=-5, stop=3, num=9))}]

param_grid2 = [{
    'clf2__n_neighbors': list(range(1, 20)),
    'clf2__p': [1, 2],
    'clf2__algorithm': ['ball_tree', 'kd_tree']
}]

param_grid3 = [{
    'learning_rate': [0.1, 0.3],
    'max_depth': [3, 6],
    'min_child_weight': list(range(1, 3))
}]

param_grid4 = [{
    'max_depth': list(range(3, 6)),
    'criterion': ['gini', 'entropy'],
    'min_samples_leaf': [20, 50]
}]

param_grid5 = [{
    'max_depth': list(range(1, 5)),
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 3]
}]

param_grid6 = [{
    'learning_rate': [0.01, 0.1],
    'loss': ['deviance', 'exponential'],
    'max_depth': list(range(3, 4))
}]

# Build the pipelines
pipe1 = Pipeline([('clf1', clf1)])

pipe2 = Pipeline([('clf2', clf2)])

pipe3 = Pipeline([('clf3', clf3)])

pipe4 = Pipeline([('clf4', clf4)])

pipe5 = Pipeline([('clf5', clf5)])

pipe6 = Pipeline([('clf6', clf6)])

print('Performing Grid Search Cross Validation...')

# Set up GridSearchCV objects, one for each algorithm
gridcvs = {}

inner_cv = StratifiedKFold(n_splits=10, shuffle=False, random_state=2)
outer_cv = StratifiedKFold(n_splits=10, shuffle=False, random_state=2)

for pgrid, est, name in zip(
    (param_grid1, param_grid2, param_grid3, param_grid4, param_grid5,
     param_grid6), (pipe1, pipe2, clf3, clf4, clf5, clf6),
    ('Logistic', 'KNN', 'XGBoost', 'DTree', 'Random Forest',
     'Gradient Boosting')):

    # First loop runs GridSearch and does Cross validation to find the best
    # parameters
    gcv = GridSearchCV(
        estimator=est,
        param_grid=pgrid,
        scoring='neg_log_loss',
        cv=outer_cv,
        verbose=0,
        refit=True,
        return_train_score=False)

    gcv.fit(X_train, y_train)

    gridcvs[name] = gcv

    print(name)
    print()
    print(gcv.best_estimator_)
    print()
    print('Best score on Grid Search Cross Validation is %.2f%%' %
          (gcv.best_score_))
    print()
    results = pd.DataFrame(gcv.cv_results_)

    # Inner loop runs Cross Val Score on tuned parameter model to determine
    # accuracy of fit

    # for name, gs_est in sorted(gridcvs.items()):
    nested_score = 0
    nested_score = cross_val_score(
        gcv, X=X_train, y=y_train, cv=inner_cv, scoring='neg_log_loss')

    print(
        'Name, Log Loss, Std Dev, based on Best Parameter Model using Cross',
        'Validation Scoring'
    )
    print('%s | %.2f %.2f' % (
        name,
        nested_score.mean(),
        nested_score.std() * 100,
    ))
    print()

    # Generate predictions and probabilities

    best_algo = gcv

    best_algo.fit(X_train, y_train)

    train_acc = accuracy_score(
        y_true=y_train, y_pred=best_algo.predict(X_train))
    test_acc = accuracy_score(y_true=y_test, y_pred=best_algo.predict(X_test))

    print('Training Accuracy: %.2f%%' % (100 * train_acc))
    print('Test Accuracy: %.2f%%' % (100 * test_acc))
    print()

    # prints classification report and confusion matrix

    if name != 'SVM':

        predictions = best_algo.predict(X_test)
        probability = best_algo.predict_proba(X_test)
        print(classification_report(y_test, predictions))
        print()
        print(confusion_matrix(y_test, predictions))
        print()

    else:
        print()

# Initialize classifier and fit data
clf = LogisticRegression(
    C=10.0,
    class_weight=None,
    dual=False,
    fit_intercept=True,
    intercept_scaling=1,
    max_iter=100,
    multi_class='ovr',
    n_jobs=1,
    penalty='l2',
    random_state=None,
    solver='liblinear',
    tol=0.0001,
    verbose=0, warm_start=False)

# clf = KNeighborsClassifier(
#     algorithm='ball_tree',
#     leaf_size=30,
#     metric='minkowski',
#     metric_params=None,
#     n_jobs=1,
#     n_neighbors=19, p=2, weights='uniform')

# clf = XGBClassifier(
#     base_score=0.5,
#     booster='gbtree',
#     colsample_bylevel=1,
#     colsample_bytree=1,
#     gamma=0,
#     learning_rate=0.1,
#     max_delta_step=0,
#     max_depth=3,
#     min_child_weight=1,
#     missing=None,
#     n_estimators=100,
#     n_jobs=1,
#     nthread=None,
#     objective='binary:logistic',
#     random_state=0,
#     reg_alpha=0,
#     reg_lambda=1,
#     scale_pos_weight=1,
#     seed=None,
#     silent=True,
#     subsample=1)

# clf = DecisionTreeClassifier(
#     class_weight=None,
#     criterion='gini',
#     max_depth=3,
#     max_features=None,
#     max_leaf_nodes=None,
#     min_impurity_decrease=0.0,
#     min_impurity_split=None,
#     min_samples_leaf=50,
#     min_samples_split=2,
#     min_weight_fraction_leaf=0.0,
#     presort=False,
#     random_state=None,
#     splitter='best')

# clf = RandomForestClassifier(
#     bootstrap=True,
#     class_weight=None,
#     criterion='entropy',
#     max_depth=4,
#     max_features='auto',
#     max_leaf_nodes=None,
#     min_impurity_decrease=0.0,
#     min_impurity_split=None,
#     min_samples_leaf=1,
#     min_samples_split=3,
#     min_weight_fraction_leaf=0.0,
#     n_estimators=10,
#     n_jobs=1,
#     oob_score=False,
#     random_state=None,
#     verbose=0,
#     warm_start=False)

# clf = GradientBoostingClassifier(
#     criterion='friedman_mse',
#     init=None,
#     learning_rate=0.1,
#     loss='deviance',
#     max_depth=3,
#     max_features=None,
#     max_leaf_nodes=None,
#     min_impurity_decrease=0.0,
#     min_impurity_split=None,
#     min_samples_leaf=1,
#     min_samples_split=2,
#     min_weight_fraction_leaf=0.0,
#     n_estimators=100,
#     presort='auto',
#     random_state=None,
#     subsample=1.0,
#     verbose=0,
#     warm_start=False)

print('Fitting data into classifier...')

clf.fit(X, y)

# Create data to input into the model
n_test_games = len(df_sample_sub)


def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))


print('Creating submission file...')

X_test = np.zeros(shape=(n_test_games, 1))
columns = df_tourney_final.columns.get_values()
model = []
data = []

for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)

    team1 = df_tourney_final[(df_tourney_final.TeamID == t1)
                             & (df_tourney_final.Season == year)].values
    team2 = df_tourney_final[(df_tourney_final.TeamID == t2)
                             & (df_tourney_final.Season == year)].values

    model = team1 - team2

    data.append(model)

Predictions = pd.DataFrame(np.array(data).reshape(2278, 19), columns=(columns))

Predictions.drop(labels=['Season', 'TeamID'], inplace=True, axis=1)

# Generate the predictions
# preds = clf.predict(Predictions)
preds = clf.predict_proba(Predictions)[:, 1]

df_sample_sub['Pred'] = preds

# Generate submission file
df_sample_sub.to_csv('Submissions/2018_predictions_lr.csv', index=False)
# df_sample_sub.to_csv('Submissions/2018_predictions_knn.csv', index=False)
# df_sample_sub.to_csv('Submissions/2018_predictions_xgb.csv', index=False)
# df_sample_sub.to_csv('Submissions/2018_predictions_dtree.csv', index=False)
# df_sample_sub.to_csv('Submissions/2018_predictions_rf.csv', index=False)
# df_sample_sub.to_csv('Submissions/2018_predictions_gb.csv', index=False)


def build_team_dict():
    """Return a team ID map to build a readable prediction sheet."""
    team_ids = pd.read_csv('DataFiles/Teams.csv')
    team_id_map = {}
    for _, row in team_ids.iterrows():
        team_id_map[row['TeamID']] = row['TeamName']
    return team_id_map


print('Creating predictions file...')

team_id_map = build_team_dict()
readable = []
less_readable = []  # A version that's easy to look up.
submission_data = df_sample_sub.values.tolist()
for pred in submission_data:
    parts = pred[0].split('_')
    less_readable.append(
        [team_id_map[int(parts[1])], team_id_map[int(parts[2])], pred[1]])
    # Order them properly.
    if pred[1] > 0.5:
        winning = int(parts[1])
        losing = int(parts[2])
        proba = pred[1]
    else:
        winning = int(parts[2])
        losing = int(parts[1])
        proba = 1 - pred[1]
    readable.append([
        '%s beats %s: %f' % (team_id_map[winning], team_id_map[losing], proba)
    ])

Finalpredictions = pd.DataFrame(readable)
Finalpredictions.to_csv('Predictions/LR_predictions.csv', index=False)
# Finalpredictions.to_csv('Predictions/KNN_predictions.csv', index=False)
# Finalpredictions.to_csv('Predictions/XGB_predictions.csv', index=False)
# Finalpredictions.to_csv('Predictions/DTREE_predictions.csv', index=False)
# Finalpredictions.to_csv('Predictions/RF_predictions.csv', index=False)
# Finalpredictions.to_csv('Predictions/GB_predictions.csv', index=False)
