import json
import re
import pandas as pd
from dateutil.parser import parse
from nltk import sent_tokenize, PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

import variables as var


def load_season_data(season, columns=var.columns_to_load):
    """
    Loads ready data with specified columns for the given season

        Parameters:
            season (int): season starting year \n
            columns (list of str) : list of columns to load

        Returns:
            df (DataFrame): Loaded Dataframe
    """
    df = pd.read_csv(f"{var.path_to_data}/{var.data_with_previews_name}_{season}.csv")[columns]
    df['Date'] = df['Date'].apply(lambda x: parse(x, dayfirst=False))
    return df


def load_full_data(columns=var.columns_to_load):
    """
    Loads ready data with specified columns for all seasons 2009/10-2021/22

        Parameters:
            columns (list of str) : list of columns to load

        Returns:
            df (pd.DataFrame): Loaded Dataframe
    """
    df = pd.read_csv(f"{var.path_to_data}/{var.data_with_previews_name}_all.csv")[columns]
    df['Date'] = df['Date'].apply(lambda x: parse(x, dayfirst=False))
    return df


def create_team_results(df):
    """
    Creates single team match results dataframe from match result dataframe with home and away teams data

        Parameters:
            df (Dataframe) : match results dataframe with columns \n
                                ['HomeTeam', 'FTHG', 'FTAG', 'HS', 'HST', 'HC', \n
                                'AwayTeam', 'FTAG', 'FTHG', 'AS', 'AST', 'AC', \n
                                'Date', 'Season', 'FTR']

        Returns:
            df_team_res (DataFrame): single team match results dataframe with columns \n
                               ['Team', 'Date', 'Season', 'Goals_Scored', 'Goals_Conceded', \n
                               'Shots', 'Shots_on_Target','Corners', 'Points']
    """
    team_result_columns = ['Team', 'Date', 'Season', 'Goals_Scored', 'Goals_Conceded', 'Shots', 'Shots_on_Target',
                           'Corners', 'Points']

    result_to_points_home = {'H': 3, 'D': 1, 'A': 0}
    df_home_teams = df[['HomeTeam', 'Date', 'Season', 'FTHG', 'FTAG', 'HS', 'HST', 'HC', 'FTR']].copy()
    columns_dict_home = dict(zip(df_home_teams.columns, team_result_columns))
    df_home_teams['FTR'] = df_home_teams['FTR'].apply(lambda x: result_to_points_home[x])
    df_home_teams.rename(columns=columns_dict_home, inplace=True)

    result_to_points_away = {'H': 0, 'D': 1, 'A': 3}
    df_away_teams = df[['AwayTeam', 'Date', 'Season', 'FTAG', 'FTHG', 'AS', 'AST', 'AC', 'FTR']].copy()
    columns_dict_away = dict(zip(df_away_teams.columns, team_result_columns))
    df_away_teams['FTR'] = df_away_teams['FTR'].apply(lambda x: result_to_points_away[x])
    df_away_teams.rename(columns=columns_dict_away, inplace=True)

    df_team_res = pd.concat([df_home_teams, df_away_teams], ignore_index=True)
    return df_team_res


def count_stats_up_to_match(df):
    """
    Counts "before" statistics: goals scored, goals conceded, points gained by team before the match

        Parameters:
            df (DataFrame): single team match results dataframe with columns \n
                               ['Team', 'Date', 'Season', 'Goals_Scored', 'Goals_Conceded', \n
                               'Shots', 'Shots_on_Target','Corners', 'Points']

        Returns:
            df (DataFrame): single team match results dataframe with 3 additional columns \n
                               ['Goals_Scored_Before', 'Goals_Conceded_Before', 'Points_Before'] \n
    """
    df = df.sort_values('Date')
    df['Goals_Scored_Before'] = df.groupby(['Team', 'Season'])['Goals_Scored'].cumsum() - df['Goals_Scored']
    df['Goals_Conceded_Before'] = df.groupby(['Team', 'Season'])['Goals_Conceded'].cumsum() - df['Goals_Conceded']
    df['Points_Before'] = df.groupby(['Team', 'Season'])['Points'].cumsum() - df['Points']
    return df


def count_stat_prev_k(df, k, columns):
    """
    Counts "previous k" statistics for given single match statistics specified in 'columns'

        Parameters:
            df (DataFrame): single team match results dataframe with columns \n
                               ['Team', 'Date', 'Season', 'Goals_Scored', 'Goals_Conceded', \n
                               'Shots', 'Shots_on_Target','Corners', 'Points'] \n
            k (int): number of previous matches to count
            columns (list of str): columns in df for counting statistics

        Returns:
            df (DataFrame): single team match results dataframe with len(columns) additional fields named like
                            {column_name}_prev_{k}
    """
    df = df.sort_values('Date')
    df[[f"{column_name}_prev_{k}" for column_name in columns]] = \
        df.groupby(['Team', 'Season'], as_index=False)[columns] \
          .rolling(window=k, min_periods=k, closed='left').sum()[columns]
    return df


def count_aggregations(df, k):
    """
    Creates dataframe with aggregated statistics from match results dataframe

        Parameters:
            df (Dataframe) : match results dataframe with columns \n
                                ['HomeTeam', 'FTHG', 'FTAG', 'HS', 'HST', 'HC', \n
                                'AwayTeam', 'FTAG', 'FTHG', 'AS', 'AST', 'AC', \n
                                'Date', 'Season', 'FTR']
            k (int): number of matches to count in "k previous" statistics

        Returns:
            df (DataFrame): match results dataframe with aggregated stats - additional columns: \n
                               ['HTP', 'HGD', 'HGSKP', 'HGCKP', 'HSKP', 'HSTKP', 'HCKP', 'HSt' \n
                               'ATP', 'AGD', 'AGSKP', 'AGCKP', 'ASKP', 'ASTKP', 'ACKP', 'ASt' \n
                               'TPDiff', 'GDDiff', 'GSKPDiff', 'GCKPDiff', 'SKPDiff', 'CKPDiff', 'StDiff']
    """

    # Single team match results dataframe
    df_teams_res = create_team_results(df)

    # Total Goals scored, goals conceded and points gained before match
    df_teams_res = count_stats_up_to_match(df_teams_res)

    # Goal Difference
    df_teams_res['Goal_Difference_Before'] = df_teams_res['Goals_Scored_Before'] - df_teams_res['Goals_Conceded_Before']

    # Past k matches
    df_teams_res = count_stat_prev_k(df_teams_res, k, ['Goals_Scored', 'Goals_Conceded', 'Shots', 'Shots_on_Target',
                                                       'Corners', 'Points'])

    df_teams_res.drop(
        columns=['Season', 'Goals_Scored', 'Goals_Conceded', 'Shots', 'Shots_on_Target', 'Corners', 'Points',
                 'Goals_Scored_Before', 'Goals_Conceded_Before'], inplace=True)

    # Join to initial match dataframe
    df = pd.merge(df, df_teams_res, left_on=['HomeTeam', 'Date'], right_on=['Team', 'Date'], how='inner').drop(
        columns=['Team'])
    columns_home_dict = dict(zip(df_teams_res.columns[2:], var.columns_features_home))
    df.rename(columns=columns_home_dict, inplace=True)

    df = pd.merge(df, df_teams_res, left_on=['AwayTeam', 'Date'], right_on=['Team', 'Date'], how='inner').drop(
        columns=['Team'])
    columns_away_dict = dict(zip(df_teams_res.columns[2:], var.columns_features_away))
    df.rename(columns=columns_away_dict, inplace=True)

    # Counting Differential stats
    for feature_name in var.features_abbreviations:
        df[f"{feature_name}Diff"] = df[f"H{feature_name}"] - df[f"A{feature_name}"]

    return df


def clear_sentence(sentence, stop_words):
    """
    Leaves only latin characters in the given sentence, deletes 'stop_words' and transforms to lower.

        Parameters:
            sentence (str): sentence to clear \n
            stop_words (set of str) : list of stop-words in lower register

        Returns:
            cleared_sentence (str): cleared sentence
    """
    regex = re.compile("[A-z]+")
    sentence = sentence.lower()
    sentence = regex.findall(sentence)
    sentence = [w for w in sentence if w not in stop_words]
    return ' '.join(sentence)


def is_sentence_about_team(sentence, team_keywords):
    """
    Checks whether sentence includes any of the given keywords

        Parameters:
            sentence (str): sentence to check \n
            team_keywords (list of str) : keywords to look for

        Returns:
            is_about (bool): result of the checking
    """
    for keyword in team_keywords:
        if keyword in sentence:
            return True
    return False


def divide_sentences(sentences, home_team, away_team, teams_keywords):
    """
    Divides sentences into 2 groups depending on the team they are about.
    If the sentence is about neither or both teams - the sentence is skipped.

        Parameters:
            sentences (list of str): sentences to filter \n
            home_team (str) : home team name \n
            away_team (str) : away team name \n
            teams_keywords (dict {str: list of str}) : dictionary of teams' keywords

        Returns:
            home_sentences (list of str) : list of sentences about home team
            away_sentences (list of str) : list of sentences about away team
    """
    home_sentences = []
    away_sentences = []
    for sentence in sentences:
        is_about_home = is_sentence_about_team(sentence, teams_keywords[home_team])
        is_about_away = is_sentence_about_team(sentence, teams_keywords[away_team])
        if is_about_home ^ is_about_away:
            if is_about_home:
                home_sentences.append(sentence)
            else:
                away_sentences.append(sentence)
    return home_sentences, away_sentences


def stem_words(sentence, stemmer):
    """
    Stems words in the given sentence using the provided stemmer.

        Parameters:
            sentence (str): sentence to stem \n
            stemmer () : stemmer with  stemmer.stem() method

        Returns:
            words (str): sentence with stemmed words
    """
    if sentence is None:
        return None
    words = ' '.join([stemmer.stem(word) for word in sentence.split()])
    return words


def stem_df(df, columns=var.columns_features_pr):
    """
    Stems words with nltk.PortStemmer() in the given columns of a dataframe.

        Parameters:
            df (DataFrame): dataframe \n
            columns (list of str) : columns to stem words in

        Returns:
            df_copy (DataFrame): copy of the given df with stemmed words
    """
    stemmer = PorterStemmer()
    df_copy = df.copy()
    df_copy[columns] = df_copy.apply(lambda x: [stem_words(x[col], stemmer) for col in columns],
                                                axis=1, result_type='expand')
    return df_copy


def lemmatize_words(sentence, lemmatizer):
    """
    Lemmatizes words in the given sentence using the provided lemmatizer.

        Parameters:
            sentence (str): sentence to lemmatize \n
            lemmatizer () : lemmatizer with  lemmatizer.lemmatize() method

        Returns:
            words (str): sentence with lemmatized words
    """
    if sentence is None:
        return None
    words = ' '.join([lemmatizer.lemmatize(word) for word in sentence.split()])
    return words


def lemmatize_df(df,columns=var.columns_features_pr):
    """
    Lemmatizes words with nltk.WordNetLemmatizer() in the given columns of a dataframe.

        Parameters:
            df (Dataframe): dataframe \n
            columns (list of str) : columns to lemmatize words in

        Returns:
            df_copy (Dataframe): copy of the given df with lemmatized words
    """
    lemmatizer = WordNetLemmatizer()
    df_copy = df.copy()
    df_copy[columns] = df_copy.apply(lambda x: [lemmatize_words(x[col], lemmatizer) for col in columns],
                                                axis=1, result_type='expand')
    return df_copy


def preview_parsing(text, home_team, away_team, stop_words, teams_keywords):
    """
    Parses given preview text: tokenizes, clears, divides between teams

        Parameters:
            text (str): preview to parse \n
            home_team (str) : home team name \n
            away_team (str) : away team name \n
            stop_words (set of str) : list of stop-words in lower register \n
            teams_keywords (dict {str: list of str}) : dictionary of teams' keywords

        Returns:
            [home_sentences, away_sentences] ([str, str]): cleared and joined sentences about home and away teams
    """
    if pd.isna(text):
        return None, None
    sentences = sent_tokenize(text)[:-1]
    sentences = [clear_sentence(sentence, stop_words) for sentence in sentences]
    home_sentences, away_sentences = divide_sentences(sentences, home_team, away_team, teams_keywords)
    return [' '.join(home_sentences), ' '.join(away_sentences)]


def previews_to_words(df, columns=var.columns_features_pr):
    """
    Parses given dataframe columns with preview text: tokenizes, clears, divides between teams

        Parameters:
            df (DataFrame): match results dataframe with columns: \n
                      ['HomeTeam', 'AwayTeam', 'text'] \n
            columns ([str, str]) : new column names \n

        Returns:
            df (DataFrame): match results dataframe with parsed previews in 2 new columns specified by 'columns'
    """
    eng_stopwords = set(stopwords.words("english"))
    teams_keywords = load_keywords_from_file(f"{var.path_to_data}/{var.teams_keywords_file}")
    df[columns] = \
        df.apply(lambda x: preview_parsing(x.text, x.HomeTeam, x.AwayTeam, eng_stopwords, teams_keywords),
                 axis=1, result_type='expand')
    return df


def get_ready_data(k):
    """
    Loads and aggregates ready data for all seasons 2009/10-2021/22.
    Parses previews.

        Parameters:
            k (int): number of previous matches to count

        Returns:
            df (pd.DataFrame): Loaded Dataframe
    """
    df = load_full_data()
    df_agg = count_aggregations(df, k)
    df_words = previews_to_words(df_agg)
    return df_words


def create_count_vectorizer(df, columns=var.columns_features_pr):
    """
    Creates nltk.CountVectorizer() for texts from specified columns of df

        Parameters:
            df (DataFrame) : dataframe
            columns (list of str) : columns with texts to include in vectorizer

        Returns:
            vec (nltk.CountVectorizer()) : created vectorizer
    """
    vec = CountVectorizer(ngram_range=(1, 1))
    vec.fit(pd.concat([df[col] for col in columns]).dropna())
    return vec


def load_keywords_from_file(file_name):
    """
    Loads teams' keywords from specified json file

        Parameters:
            file_name (str) : json file name

        Returns:
            teams_dict (dict {str : list of str}) : teams' keywords dictionary
    """
    with open(file_name, "r") as outfile:
        teams_dict = json.load(outfile)
    return teams_dict


def add_keyword(teams_dict, team, keyword):
    """
    Adds a keyword to teams' keywords dictionary

        Parameters:
             teams_dict (dict {str : list of str}) : teams' keywords dictionary \n
             team (str) : team name \n
             keyword (str)

        Returns:
            teams_dict (dict {str : list of str}) : update teams' keywords dictionary
    """
    teams_dict[team].append(keyword)
    return teams_dict


def save_keywords_to_file(teams_dict, file_name):
    """
    Saves teams' keywords to specified json file

        Parameters:
            teams_dict (dict {str : list of str}) : teams' keywords dictionary \n
            file_name (str) : json file name

        Returns:
    """
    with open(file_name, "w") as outfile:
        json.dump(teams_dict, outfile)
    print(f"Successfully saved to {file_name}")
