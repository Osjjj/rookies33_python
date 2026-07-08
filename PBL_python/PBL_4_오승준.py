import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class SaleAnalysis :
    def __init__(self) :
        # 2024년 1월 1일부터 12월 31일까지의 날짜 생성
        self.date_index = pd.date_range('2024-01-01', '2024-12-31')

        # 일별 매출 1000~10000 사이의 난수로 생성, 생성 개수 = 날짜 데이터의 개수
        self.day_sale = np.random.randint(1000, 10001, size=(len(self.date_index)))

        # 앞서 생성한 날짜와 일별 매출 데이터를 기반으로 데이터 프레임 생성 
        self.df = pd.DataFrame({
            'Day' : self.date_index,
            'Sale' : self.day_sale
        })

        # 날짜를 인덱스로 설정함
        self.df.set_index('Day', inplace = True)


    def monthly_analysis(self) : 
        # 매출의 총합을 월별로 계산
        monthly_sales = self.df.groupby(self.df.index.month).sum()
        
        # 계산한 월별 매출 총합을 반환
        return monthly_sales
    
    def plot_sales(self, monthly_data) :

        # 한글 폰트 설정 (맑은 고딕)
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

        # 변수로 받아온 월별 데이터를 통해 꺾은선 그래프 생성 (선 색상 = 빨강)
        plt.plot(monthly_data.index, monthly_data['Sale'], c = 'r')
        
        # 차트 제목, 축 제목 생성
        plt.title('2024년 월별 매출 분석')
        plt.xlabel('월')
        plt.ylabel('매출 총합')

        # x축의 눈금을 1 ~ 12월로 설정
        plt.xticks(range(1,13))

        # 가독성을 위한 그리드 설정
        plt.grid()
        
        # 그래프 출력
        plt.show()

# SaleAnalysis 인스턴스 생성        
sale = SaleAnalysis()

# 월별 매출 분석 함수를 실행
monthly_data = sale.monthly_analysis()

# 꺾은선 그래프 생성을 통한 시각화
sale.plot_sales(monthly_data)