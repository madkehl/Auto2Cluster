import pandas as pd
import os
os.chdir("../mysql")
from mysqlfunctions import load_my_sql, result2bio
os.chdir("../misc. data pulls")

punct = [',', '.', '—', ')', '(', '>', ':', "''", '``', '*', ';', '?', '!', '...', '[', ']', '«', '»', '?', '„', '“', '…']
connection = load_my_sql('')
all_diarynums = connection.execute('SELECT diary FROM notes;')


for_tatyana = list(set([i[0] for i in all_diarynums.fetchall()]))
                     
reflection_fn = []
reflection_sn = []
reflection_info = []


for i in for_tatyana:
    q = result2bio(connection, str(i))
    try:
        reflection_fn.append(q[0][0])
    except IndexError:
        reflection_fn.append('None')
    try:
        reflection_sn.append(q[0][1])
    except IndexError:
        reflection_sn.append('None')
    try:
        reflection_info.append(q[0][2])
    except IndexError:
        reflection_info.append('None')

key_dnev = pd.DataFrame({'firstName': reflection_fn, 'lastName': reflection_sn, 'info': reflection_info})

odfrtaBIO = pd.concat([pd.Series(for_tatyana,name = 'diarynum'), key_dnev], axis=1)
odfrtaBIO.to_csv('./010820bios.csv')
