import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class StudentScoreAnalysis :
    def __init__(self) :
        # 학생 1~20 까지 20명의 학생 생성
        students = [f'학생 {i}' for i in range(1,21)]

        # 수학, 영어, 과학 3개의 과목 생성
        subjects = ['수학', '영어', '과학']

        # 50 ~ 100점 까지의 점수를 난수로 생성 (개수 = 학생의 숫자, 과목 개수)
        scores = np.random.randint(50, 101, size = (len(students), 3))
        
        # 난수로 생성한 점수를 데이터로 가지고 인덱스는 학생명, 컬럼명은 과목인 데이터 프레임 생성
        self.df = pd.DataFrame(scores, index = students, columns = subjects)

    def get_subjects_mean(self) :
        # 과목별 평균 점수 계산
        subjects_mean = self.df.mean()
        
        # 과목별 평균 점수 반환
        return subjects_mean

    def get_top5_students(self) :
        # 각 학생별 평균 점수 계산
        mean_scores = self.df.mean(axis = 'columns')
        
        # 평균 점수를 기준으로 내림차순 정렬
        sorted_mean_scores = mean_scores.sort_values(ascending = False)

        # 평균 점수 상위 5위의 학생들 추출
        top5_students =  sorted_mean_scores.head(5)
        
        # 상위 5위 이내의 학생들의 이름, 평균 점수 반환
        return top5_students

    def plot_data(self, subject_mean, top5_students) :
        # 한글 폰트 설정 (맑은 고딕)
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

        # 두 개의 그래프를 담을 피규어 생성
        plt.figure(figsize = (8, 6))

        # 왼쪽에 위치할 과목별 평균 점수 막대 그래프 생성 (두께 = 0.5)
        plt.subplot(1,2,1)
        plt.bar(subject_mean.index, subject_mean.values, width = 0.5)
        
        # 평균 점수 막대 그래프의 차트 제목, 축 제목 설정
        plt.title('과목별 평균')
        plt.xlabel('과목명')
        plt.ylabel('평균 점수')
        
        # 오른쪽에 위치할 상위 5위 학생 막 대 그래프 생성 (시각화를 위한 색상 = 초록색)
        plt.subplot(1,2,2)
        plt.bar(top5_students.index, top5_students.values, color = 'g')

        # 상위 5위 학생 막대 그래프의 차트 제목, 축 제목 설정
        plt.title('평균 상위 5위 이내 학생')
        plt.xlabel('학생명')
        plt.ylabel('평균 점수')

        # 그래프 출력
        plt.show()

# StudentScoreAnalysis 인스턴스 생성
stu_analysis = StudentScoreAnalysis()

# 과목별 평균 점수 계산 함수를 실행
subjects_mean = stu_analysis.get_subjects_mean()

# 상위 5위 이내 학생 계산 함수를 실행
top5_students = stu_analysis.get_top5_students()

# 앞서 추출한 데이터들을 바탕으로 그래프를 생성
stu_analysis.plot_data(subjects_mean, top5_students)