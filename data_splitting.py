from scipy.sparse import hstack
import variables as var


def match_results_encoding(result_letter):
    """
    Encodes match result from {'H', 'D', 'A'} to {0, 1, 2}
    """
    result_dict = {'H': 0, 'D': 1, 'A': 2}
    return result_dict[result_letter]


def print_sample_sizes(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Prints sizes of train, validation and test samples and stratification of target variable

        Parameters:
            X_train (DataFrame) : train sample features \n
            X_val (DataFrame) : validation sample features \n
            X_test (DataFrame) : test sample features \n
            y_train (DataFrame) : train sample target \n
            y_val (DataFrame) : validation sample target \n
            y_test (DataFrame) : test sample target \n
    """
    print("Sample sizes:\n")
    train_size, valid_size, test_size = len(X_train), len(X_val), len(X_test)
    total_size = train_size + valid_size + test_size
    print(f"Total: {total_size}")
    print(f"Train sample: {train_size}, {train_size / total_size}")
    print(y_train.value_counts() / train_size)
    print(f"Validation sample: {valid_size}, {valid_size / total_size}")
    print(y_val.value_counts() / valid_size)
    print(f"Test sample: {test_size}, {test_size / total_size}")
    print(y_test.value_counts() / test_size)


def split_data_by_time(df, X_columns, y_column, train_q=0.7, valid_q=0.15):
    """
    Splits data in train, validation and test samples according to train_q and valid_q proportions.
    The data is spit within each season so that first train_q % matches go to train sample, next valid_q % -
    to validation sample, the other - to test sample.
    Saves chronological order within seasons.

        Parameters:
            df (DataFrame) : dataframe with 'Season' column to split \n
            X_columns (list of str) : feature columns to return \n
            y_column (str) : target column to return \n
            train_q (double) : proportion of train sample \n
            valid_q (double) : proportion of validation sample \n

        Returns:
            X_train (DataFrame) : train sample features \n
            X_val (DataFrame) : validation sample features \n
            X_test (DataFrame) : test sample features \n
            y_train (DataFrame) : train sample target \n
            y_val (DataFrame) : validation sample target \n
            y_test (DataFrame) : test sample target \n
    """
    df = df.sort_values(["Date"], ascending=True)
    df['In_Season_id'] = df.groupby(["Season"]).cumcount()
    df['Train_Quantile'] = df.groupby(["Season"])["In_Season_id"].transform(lambda group: group.quantile(train_q))
    df['Valid_Quantile'] = df.groupby(["Season"])["In_Season_id"].transform(lambda group: group.quantile(train_q + valid_q))
    df[y_column] = df[y_column].apply(lambda x: match_results_encoding(x))
    df_train = df[df.In_Season_id <= df.Train_Quantile]
    df_val = df[(df.Train_Quantile < df.In_Season_id) & (df.In_Season_id <= df.Valid_Quantile)]
    df_test = df[df.Valid_Quantile < df.In_Season_id]
    X_train, X_val, X_test, y_train, y_val, y_test = df_train[X_columns], df_val[X_columns], df_test[X_columns], \
        df_train[y_column], df_val[y_column], df_test[y_column]
    print_sample_sizes(X_train, X_val, X_test, y_train, y_val, y_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_count_vectorizer(X_words, vec, columns=var.columns_features_pr):
    """
    Vectorizes provided columns with given nltk.CountVectorizer()

        Parameters:
            X_words (DataFrame) : dataframe \n
            vec (nltk.CountVectorizer) : vectorizer \n
            columns (list of str) : text columns to vectorize

        Returns:
            X (scipy.sparse.spmatrix) : sparsed matrix - vectorization
    """
    X = hstack([vec.transform(X_words[col]) for col in columns])
    return X


def split_data_by_time_vectorize(df, X_columns, y_column, vec, train_q=0.7, valid_q=0.15):
    """
    See 'split_data_by_time'
    After splitting vectorizes provided X_columns with given nltk.CountVectorizer()

        Parameters:
            df (DataFrame) : dataframe with 'Season' column to split \n
            X_columns (list of str) : feature columns to vectorize (must be str) \n
            y_column (str) : target column to return \n
            vec (nltk.CountVectorizer) : vectorizer \n
            train_q (double) : proportion of train sample \n
            valid_q (double) : proportion of validation sample

        Returns:
            X_train (scipy.sparse.spmatrix) : train sample features \n
            X_val (scipy.sparse.spmatrix) : validation sample features \n
            X_test (scipy.sparse.spmatrix) : test sample features \n
            y_train (DataFrame) : train sample target \n
            y_val (DataFrame) : validation sample target \n
            y_test (DataFrame) : test sample target \n
    """
    X_train, X_val, X_test, y_train, y_val, y_test = \
        split_data_by_time(df, X_columns, y_column, train_q, valid_q)
    X_train, X_val, X_test = \
        apply_count_vectorizer(X_train, vec, X_columns), \
        apply_count_vectorizer(X_val, vec, X_columns), \
        apply_count_vectorizer(X_test, vec, X_columns)
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_valid_quantitative_dataset(df):
    """
    Drops objects with None in aggregated columns
    """
    return df.dropna(subset=['HSt', 'ASt'], how='any')


def get_valid_preview_dataset(df):
    """
    Drops objects with None in both parsed previews
    """
    return df.dropna(subset=['words_home', 'words_away'], how='all').fillna('')


def get_valid_dataset(df):
    """
    Returns valid quantitative and text datase
    """
    df = df.dropna(subset=['HSt', 'ASt'], how='any')
    df = df.dropna(subset=['words_home', 'words_away'], how='all').fillna('')
    return df
