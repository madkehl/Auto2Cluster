import sqlalchemy
import pandas as pd


def load_my_sql(password):
    """

    :param password: password for my home machine
    :return:
    """
    passw = password
    engine = sqlalchemy.create_engine('mysql+pymysql://madkehl:' + passw + '@127.0.0.1/prozhito_orig', encoding='utf-8')
    connection = engine.connect()
    return connection


def resultToList(connec, diarynum):
    """
    used in keyword_vecs
    :param connec: connection object
    :param diarynum: the diary number
    :return: list of texts for that diarynum
    """
    trial = connec.execute('SELECT text FROM notes WHERE id = ' + diarynum + ';')
    ResultSet = trial.fetchall()
    koshka_list = []
    for n in ResultSet:
        koshka_list.append(n[0])
    return koshka_list


def retrieve_notes(connec, id_num):
    """
    retrieve all info for given ID
    """
    trial = connec.execute('SELECT * FROM notes WHERE id = ' + str(id_num) + ';')
    result_set = trial.fetchall()
    return result_set


def result2bio(connec, diarynum):
    """
    returns bio of given diarynum
    :param connec: connection object from load sql
    :param diarynum: id num for person
    :return: list of the given id info
    """
    trans = connec.execute('SELECT person FROM diary WHERE id = ' + str(diarynum) + ';')
    trans = trans.fetchall()
    if len(trans) < 1:
        return [[None, None, None]]
    bio_info = connec.execute('SELECT firstName, lastName, info FROM persons WHERE id = ' + str(trans[0][0]) + ';')
    result_set = bio_info.fetchall()
    result_ls = []
    for n in result_set:
        result_ls.append([n[0], n[1], n[2]])
    return result_ls


def sample(df, num_samples):
    dice = range(max(df['new_entry_type']) + 1)
    for_tatyana = []
    id_col = []
    for i in dice:
        temp = df[df['new_entry_type'] == i]
        name = temp['new_name_type'].iloc[0]
        res = temp['entry_id'].sample(n=num_samples, random_state=7)
        res = res.reset_index(drop=True)
        for_tatyana.append(res)
        for x in range(num_samples):
            id_col.append(name)

    return for_tatyana, id_col


def full_path_to_entries(df, n, passw):
    connec = load_my_sql(passw)
    full_entries = []
    ids, entry_types = sample(df, n)
    for id_ in ids:
        for n in id_:
            full_entries.append(retrieve_notes(connec, n))

    entry_id = []
    diarynum = []
    text = []
    year = []

    for entry in full_entries:
        for n in entry:
            entry_id.append(n[0])
            diarynum.append(n[1])
            text.append(n[2])
            year.append(n[3].year)

    bios = []
    for num in diarynum:
        bios.append(result2bio(connec, num))

    first_name = []
    last_name = []
    info = []

    for bio in bios:
        for n in bio:
            first_name.append(n[0])
            last_name.append(n[1])
            info.append(n[2])

    full_entries_df = pd.DataFrame({

        'name_type': entry_types,
        'entry_id': entry_id,
        'diarynum': diarynum,
        'text': text,
        'year': year,
        'first_name': first_name,
        'last_name': last_name,
        'info': info

    })

    return full_entries_df


def print_answers(df):
    for n in df.iterrows():
        i = n[1]

        print(i['name_type'])
        print(i['entry_id'])
        print(i['diarynum'])
        print()
        print(i['year'])
        print(str(i['first_name']) + ' ' + str(i['last_name']))
        print()
        print(i['text'])
        print()
