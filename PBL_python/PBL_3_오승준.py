import os

class StudentScores :
    def __init__(self, data_path) :
        self.path = data_path

        self.data = {}

        try :
            with open(self.path, 'r', encoding='utf-8-sig') as f:
                for line in f :
                    # 기존 데이터 = 학생 이름, 점수 
                    # 공백을 제거하고 ',' 기준으로 분할하여 학생 이름을 Key 값, 점수를 Value 값으로 딕셔너리 데이터 생성
                    name, score = line.strip().split(',')
                    self.data[name] = int(score)

        # 파일 경로를 잘못 전달 받았을 때 발생하는 오류 예외처리
        except FileNotFoundError :
            print('파일이 존재하지 않습니다.')

    def calculate_average(self) :
        # 딕셔너리의 밸류값(점수) => 내장함수 sum()을 사용해 총합 계산 후, self.data의 길이로 나누어 평균 계산
        self.average = sum(self.data.values()) / len(self.data)
        return self.average
    
    def get_above_average(self) :

        # self.data 딕셔너리에서 평균 이상의 점수를 가진 학생 이름(key 값)만 리스트로 추출
        self.above_students = [k for k, v in self.data.items() if v >= self.average]

        return self.above_students
        
    def save_below_average(self) :
        # self.data 딕셔너리에서 self.above_students 리스트에 포함되지 않은 학생들을 딕셔너리로 생성
        self.below_data = {k : v for k, v in self.data.items() if k not in self.above_students}

        with open('data/below_average_korean.txt', 'w', encoding='utf-8-sig') as f:
            # self.below_data 딕셔너리의 값을 txt 파일에 작성
            for name, score in self.below_data.items() :
                f.write(f'{name}, {score}\n')
    
    def print_summary(self) :
        print(f'평균 점수 : {self.average}')
        print('='*20)
        print(f'평균 이상 학생 : {self.above_students}')

# StudentScores 인스턴스 생성
stu_data = StudentScores('data/scores_korean.txt')

stu_data.calculate_average()
stu_data.get_above_average()
stu_data.save_below_average()
stu_data.print_summary()