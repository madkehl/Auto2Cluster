from mysqlfunctions import full_path_to_entries, print_answers
import os
import pandas as pd


def main():
    """
    takes inputs of password and the number of entries desired
    :return:
    """
    pass_w = input("TYPE PASSWORD MYSQL: ")
    n = input("SELECT NUMBER OF ENTRIES: ")
    data_dir = '../final/data'
    for subdir, dirs, files in os.walk(data_dir):
        for filename in files:
            filepath = subdir + os.sep + filename
            df = pd.read_csv(filepath)
            full_entries_df = full_path_to_entries(df, n, pass_w)
            print_answers(full_entries_df)


if __name__ == '__main__':
    main()
