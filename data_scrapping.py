import re
import pandas as pd
import datetime as dt
from dateutil.parser import parse
from bs4 import BeautifulSoup
import requests

import variables as var


def teams_from_title(title):
    """
    Parses team names from preview page title

        Parameters:
            title (str): page title

        Returns:
            home_team (str) : home team name \n
            away_team (str) : away team name \n
    """
    title = title.split(':')
    title[:] = filter(lambda string: ' v ' in string, title)
    title = title[0].replace('\t', ' ').replace('squad sheets', '').strip()
    # убираем время вида xx 1.30am или xx 4pm
    if title[-2:] in ['am', 'pm'] and title[-3].isdigit():
        title = title[:title.rfind(' ')].strip()
    # убираем день недели на конце
    if title[title.rfind(' ') + 1:].lower() in var.week_days:
        title = title[:title.rfind(' ')].strip()
    team_names = title.split(' v ')
    return team_names[0].strip().lower(), team_names[1].strip().lower()


def parse_match_page(url):
    """
    Parses single match page

        Parameters:
            url (str): page url

        Returns:
            data (list of str) : [text, preview_date, home_team, away_team, url]
    """
    r = requests.get(url)
    page = r.content.decode("utf-8")
    soup = BeautifulSoup(page, 'html.parser')

    # Проверяем, что статья про матч АПЛ по топику
    topics_block = soup.find_all(class_='dcr-1nx1rmt')
    if len(topics_block) < 1:
        topics_block = soup.find_all(class_='submeta')
    if len(topics_block) < 1:
        raise Exception('Topic block not found')
    topic = topics_block[0].get_text()
    if 'FA' in topic:
        return ['FA'] # FA Cup
    elif 'World Cup' in topic:
        return ['WC'] # World Cup
    elif 'Champions League' in topic:
        return ['CL'] # Champions League
    elif 'uefa super cup' in topic.lower():
        return ['USC']
    elif 'Carabao Cup' in topic:
        return ['CC'] # Carabao Cup
    elif 'Cup' in topic:
        return ['OC'] # Other Cup

    # Ищем текст превью, как самый длинный параграф в теле новости
    text_blocks = soup.find(class_=re.compile(".*article-body.*"))
    preview_text = ''
    for text_block in text_blocks:
        text = text_block.get_text()
        if len(text) > len(preview_text):
            preview_text = text
    if preview_text == '':
        raise Exception('Empty Preview')

    # Дата превью из ссылки страницы
    # Проще достать дату из ссылки, чем искать на странице
    preview_date = ' '.join(url.split('/')[4:7])
    # Если распарсилась неверная дата, получим исключение и сразу отловим это
    parse(preview_date)

    # Названия команд из заголовка - по ':' убираем лишнее, команды делим по ' v '
    title = soup.find('h1').get_text()
    try:
        team_names = teams_from_title(title)
    except Exception as e:
        raise Exception('Parsing Team Names ' + str(e))
    if team_names[0] == '' or team_names[1] == '':
        raise Exception('Empty Team Names')

    return [preview_text, preview_date, team_names[0].strip().lower(), team_names[1].strip().lower(), url]


def get_page_link(page_num):
    """
    Forms a url to news on page 'page_number'
    """
    return f"https://www.theguardian.com/football/series/match-previews?page={page_num}"


def get_page_num(url):
    """
    Gets page number from url
    """
    return int(url.split('=')[-1])


def parse_search_page(url):
    """
    Parses a search page. On each page there are 20 posts - parses each of them via 'parse_match_page'

        Parameters:
            url (str): search page url

        Returns:
            data (list of list) : list of data about parsed matches - see 'parse_match_page' returns
    """
    page_num = get_page_num(url)
    print(f'Parsing page {page_num}')
    previews_list = []
    r = requests.get(url)
    page = r.content.decode("utf-8")
    soup = BeautifulSoup(page, 'html.parser')

    # Достаем все сслыки на новости со страницы - их должно быть 20, кроме последней
    match_pages = soup.find_all(class_='u-faux-block-link__overlay js-headline-text')
    if len(match_pages) != 20 and page_num < var.guardian_max_page:
        print(f'Couldn\'t get links on page {page_num} --- Trying Again ---')
        return parse_search_page(url)

    parsed_cnt = 0
    other_tournaments = 0
    for match in match_pages:
        match_url = match['href']
        try:
            parsed_match = parse_match_page(match_url)
            if len(parsed_match) == 1:
                if parsed_match[0] in ['FA', 'WC', 'CL', 'USC', 'CC', 'OC']:
                    other_tournaments += 1
            else:
                # Достали матч АПЛ
                previews_list.append(parsed_match)
                parsed_cnt += 1
        except Exception as e:
            print(f"Error on page {page_num} parsing {match_url}")
            print(f"Error log: {str(e)}")
    print(f"Previews parsed: {parsed_cnt} (+ {other_tournaments} other games) on page number {page_num}\n")
    return previews_list


def parse_pages(start_page, end_page):
    """
    Parses pages by number: from start_page to end_page inclusively via 'parse_search_page'

        Parameters:
            start_page (int) \n
            end_page (int) \n

        Returns:
            data (list of list) : list of data about parsed matches - see 'parse_match_page' returns
    """
    end_page = min(end_page, var.guardian_max_page)
    previews = []
    for page_num in range(start_page, end_page + 1):
        page_link = get_page_link(page_num)
        page_result = parse_search_page(page_link)
        previews.extend(page_result)
    return previews


def save_preview_data(data, file_name):
    """
    Saves parsed data to file

        Parameters:
            data (list of list): list of match data - see 'parse_match_page' returns \n
            file_name (str)
    """
    df = pd.DataFrame(data, columns=['text', 'preview_date', 'home_team', 'away_team', 'url'])
    df.to_csv(f"{var.path_to_data}/{file_name}", index=False)


def download_previews(file_name):
    """
    Loads parsed data from file

        Parameters:
            file_name (str)

        Returns:
            data (list of list): list of match data - see 'parse_match_page' returns
    """
    df = pd.read_csv(f"{var.path_to_data}/{file_name}")
    df['preview_date'] = df.preview_date.apply(lambda date: parse(date))
    df = df.drop_duplicates()
    return df


def download_season_data(season_start_year):
    """
    Loads season data from file

        Parameters:
            season_start_year (str) : season to load (2009 -> 2009/10)

        Returns:
            df (DataFrame): season matches data
    """
    df = pd.read_csv(f"{var.path_to_data}/{var.raw_data_name}{season_start_year}.csv")
    df = df.dropna(how='all')
    df['Date'] = df['Date'].apply(lambda date: parse(date, dayfirst=True))
    df['HomeTeam'] = df['HomeTeam'].apply(lambda name: name.lower())
    df['AwayTeam'] = df['AwayTeam'].apply(lambda name: name.lower())
    return df


def find_different_names(df_data, df_previews):
    """
    Finds different names that should be changed

        Parameters:
            df_data (pd.DataFrame) : results dataframe with ['HomeTeam', 'AwayTeam'] columns \n
            df_previews (pd.DataFrame) : previews dataframe with ['home_team', 'away_team'] columns

        Returns:
            in_data (list of str) : names found in df_data and not found in df_previews \n
            in_previes (list of str) : names found in df_previews and not found in df_data \n
    """
    names_previews = set(df_previews['home_team'].unique().tolist() + df_previews['away_team'].unique().tolist())
    names_data = set(df_data['HomeTeam'].unique().tolist() + df_data['AwayTeam'].unique().tolist())
    return sorted(names_data.difference(names_previews)), sorted(names_previews.difference(names_data))


def join_data_previews(df_season_data, df_previews):
    """
    Joins season data with previews on pairs [home_team, away_team].
    Logs different names, not matched previews, duplicated previews, left matches.

        Parameters:
            df_season_data (pd.DataFrame) : season match results dataframe \n
            df_previews (pd.DataFrame) : previews dataframe (all seasons)

        Returns:
            df_merged (list of str) : dataframe with season matches, results and previews
    """
    min_date = df_season_data['Date'].min() - dt.timedelta(days=7)
    max_date = df_season_data['Date'].max() + dt.timedelta(days=2)
    df_previews_filtered = df_previews[(min_date < df_previews['preview_date']) &
                                       (df_previews['preview_date'] < max_date)].copy()
    df_previews_filtered['home_team'] = df_previews_filtered['home_team'].apply(lambda name: var.name_dict.get(name, name))
    df_previews_filtered['away_team'] = df_previews_filtered['away_team'].apply(lambda name: var.name_dict.get(name, name))

    # Даты первого и последнего матча сезона, первого и последнего превью
    print('Season  period:', df_season_data['Date'].min(), df_season_data['Date'].max())
    print('Preview period: ', df_previews_filtered['preview_date'].min(), df_previews_filtered['preview_date'].max())

    # Количество превью в рамках сезона и количество матчей в данных
    print(f'Previews: {len(df_previews_filtered)}\n'
          f'Matches: {len(df_season_data)}\n')

    # Количество дубликатов в превью (home_team + away_team)
    print('\nDuplicated matches:\n')
    df_duplicates = df_previews_filtered[df_previews_filtered.duplicated(['home_team', 'away_team'], keep=False)]\
        .drop('text', axis=1)
    print(df_duplicates.sort_values(['home_team', 'away_team', 'preview_date', 'url']))

    # Разные названия команд
    print('\nTeam names difference:', find_different_names(df_season_data, df_previews_filtered))
    print('Teams in database:\n', df_season_data['HomeTeam'].unique().tolist())

    # Количество превью без матчей
    df_skipped_previews = pd.merge(df_season_data, df_previews_filtered,
                                   left_on=['HomeTeam', 'AwayTeam'],
                                   right_on=['home_team', 'away_team'],
                                   how='right')
    df_skipped_previews = \
        df_skipped_previews[df_skipped_previews['HomeTeam'].isna()][['home_team', 'away_team', 'preview_date', 'url']]
    print('\nPreviews not matched\n', df_skipped_previews)

    # Готовая таблица
    df_merged = pd.merge(df_season_data, df_previews_filtered,
                         left_on=['HomeTeam', 'AwayTeam'],
                         right_on=['home_team', 'away_team'],
                         how='left')

    # Дубликаты после джоина
    print('\nDuplicated matches after join\n:')
    df_duplicates_joined = df_merged[df_merged.duplicated(['home_team', 'away_team'], keep=False)].drop('text', axis=1)
    print(df_duplicates_joined.sort_values(['home_team', 'away_team', 'preview_date'])[['home_team', 'away_team', 'preview_date', 'Date', 'url']].dropna())

    # Количество матчей без превью до фильтрации
    print(f'\nMatches without reviews before distant filter: {len(df_merged[df_merged.home_team.isna()])}\n')
    print(df_merged[['HomeTeam', 'home_team']].info())

    # Матчи, где между превью и матчем более 5 дней
    df_distant_matches = df_merged[(~df_merged.HomeTeam.isna()) & (~df_merged.home_team.isna())]
    df_distant_matches = \
        df_distant_matches[((df_distant_matches['Date'] - df_distant_matches['preview_date']) > dt.timedelta(days=5)) |
                                            (df_distant_matches['Date'] < df_distant_matches['preview_date'])]
    print(f'\nDistant matches\n')
    print(df_distant_matches[['HomeTeam', 'AwayTeam', 'Date', 'preview_date']])

    # Занулим превью на всех distant матчах
    distant_matches = df_distant_matches[['HomeTeam', 'AwayTeam']].values.tolist()
    mask = df_merged.apply(lambda x: not ([x['HomeTeam'], x['AwayTeam']] in distant_matches), axis=1)
    df_merged.loc[~mask, ['home_team', 'away_team', 'preview_date', 'text']] = pd.NA

    # Количество матчей без превью
    print(f'\nMatches without reviews: {len(df_merged[df_merged.home_team.isna()])}\n')
    print(df_merged[['HomeTeam', 'home_team']].info())

    return df_merged


def swap_teams(df, home_team, away_team, preview_date):
    """
    Swaps home_team and away_team in match on preview_date
    """
    df.loc[(df.home_team == home_team) &
           (df.away_team == away_team) &
           (df.preview_date == parse(preview_date, dayfirst=False)),
           ['home_team', "away_team"]] = [away_team, home_team]
    return df


def delete_preview(df, home_team, away_team, preview_date):
    """
    Deletes preview [home_team, away_team, preview_date]
    """
    df = df[~((df.home_team == home_team) &
              (df.away_team == away_team) &
              (df.preview_date == parse(preview_date, dayfirst=False)))]
    return df
