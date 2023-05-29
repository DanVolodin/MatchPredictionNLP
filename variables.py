path_to_data = 'Data'
path_to_logs = 'Logs'
data_with_previews_name = 'EPL_Data_Previews'
teams_keywords_file = 'team_keywords.json'
columns_to_load = ['HomeTeam', 'AwayTeam', 'Date', 'Season', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC',
                   'FTR', 'text']

columns_features_home = ['HTP', 'HGD', 'HGSKP', 'HGCKP', 'HSKP', 'HSTKP', 'HCKP', 'HSt']
columns_features_away = ['ATP', 'AGD', 'AGSKP', 'AGCKP', 'ASKP', 'ASTKP', 'ACKP', 'ASt']
columns_features_diff = ['TPDiff', 'GDDiff', 'GSKPDiff', 'GCKPDiff', 'SKPDiff', 'CKPDiff', 'StDiff']

features_abbreviations = ['TP', 'GD', 'GSKP', 'GCKP', 'SKP', 'STKP', 'CKP', 'St']

columns_features_q = ['HTP', 'HGD', 'HGSKP', 'HGCKP', 'HSKP', 'HSTKP', 'HCKP', 'HSt',
                      'ATP', 'AGD', 'AGSKP', 'AGCKP', 'ASKP', 'ASTKP', 'ACKP', 'ASt',
                      'TPDiff', 'GDDiff', 'GSKPDiff', 'GCKPDiff', 'SKPDiff', 'CKPDiff', 'StDiff']
columns_features_pr = ['words_home', 'words_away']
column_result = 'FTR'

# Data Scrapping
guardian_max_page = 219  # changes gradually
raw_data_name = 'EPL_Data_'
ready_data_name = 'EPL_Data_Previews'
season_info_file = 'Season_Info.csv'
scrapping_log_file = 'scrapping_log.txt'
joining_log_file = 'joining_log.txt'
previews_all_file = 'previews_all.csv'

week_days = ['monday', 'mon', 'tuesday', 'tue', 'wednesday', 'wed', 'thursday',
             'thu', 'friday', 'fri', 'saturday', 'sat', 'sunday', 'sun']

# parsed_name -> data name
name_dict = {'manchester city' : 'man city',
             'manchester united' : 'man united',
             'southamptom' : 'southampton',
             'leeds united' : 'leeds',
             'west bromwich albion' : 'west brom',
             'west bromwich' : 'west brom',
             'leicester city' : 'leicester',
             'brighton & hove albion' : 'brighton',
             'newcastle united' : 'newcastle',
             'norwich city' : 'norwich',
             'tottenham hotspur' : 'tottenham',
             'tottenham hostpur' : 'tottenham',
             'tottenham hotpsur' : 'tottenham',
             'spurs' : 'tottenham',
             'west ham united' : 'west ham',
             'wolverhampton wanderers' : 'wolves',
             'cardiff city' : 'cardiff',
             'huddersfield town' : 'huddersfield',
             'newcastle utd' : 'newcastle',
             'stoke city' : 'stoke',
             'swansea city' : 'swansea',
             'hull city' : 'hull',
             'queens park rangers' : 'qpr',
             'squad sheets queens park rangers' : 'qpr',
             'wigan athletic' : 'wigan',
             'blackburn rovers' : 'blackburn',
             'bolton wanderers' : 'bolton',
             'birmingham city' : 'birmingham',
             'birmngham city' : 'birmingham',
             'brimingham city' : 'birmingham'
             }

matches_to_swap = [['west ham', 'everton', '2021-10-16'],
                   ['leeds', 'manchester united', '2020-12-19'],
                   ['manchester united', 'liverpool', '2021-01-16'],
                   ['tottenham', 'manchester united', '2020-10-03'],
                   ['fulham', 'aston villa', '2011-02-04'],
                   ['wolverhampton wanderers', 'wigan athletic', '2010-10-01']
                   ]

matches_to_delete = [['tottenham hotspur', 'millwall', '2017-03-11'],
                     ['manchester city', 'wigan athletic', '2014-03-07'],
                     ['sheffield united', 'charlton athletic', '2014-03-07'],
                     ['arsenal', 'everton', '2014-03-07'],
                     ['hull city', 'sunderland', '2014-03-07'],
                     ['liverpool', 'stoke city', '2012-03-16'],
                     ['tottenham hotspur', 'bolton wanderers', '2012-03-16'],
                     ['chelsea', 'leicester city', '2012-03-16'],
                     ['everton', 'sunderland', '2012-03-16'],
                     ['manchester city', 'reading', '2011-03-11'],
                     ['arsenal', 'birmingham city', '2011-02-25'],
                     ['birmingham city', 'bolton wanderers', '2011-03-11'],
                     ['manchester united', 'arsenal', '2011-03-11'],
                     ['stoke city', 'west ham united', '2011-03-11'],
                     ['west bromwich albion', 'wolverhampton wanderers', '2010-12-17']
                     ]
