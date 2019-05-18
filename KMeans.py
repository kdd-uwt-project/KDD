import sklearn
import csv


def profile_loader():
    reader = csv.reader(open("./data_set_phase1/profiles.csv", 'r'))
    for line in reader:
        print(line)
        break


if __name__ == '__main__':
    pass
