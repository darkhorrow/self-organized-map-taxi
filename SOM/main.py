
from question_1 import Question_1
from question_2 import Question_2
from question_3 import Question_3
from question_4 import Question_4

from data_cleaner import DataCleaner

def main():
    dc = DataCleaner('datos/taxis.csv')

    Question_1(dc, 10)
    Question_2(dc, 10)
    Question_3(dc, 20)
    Question_4(dc, 20)

if __name__ == '__main__':
    main()