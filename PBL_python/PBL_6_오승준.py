import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class CustomerSalesAnalysis :
    def __init__(self) :
        # 고객 1~5 까지 5명의 고객을 생성
        customers = [f'고객 {i}' for i in range(1,6)]
        
        # 5명의 고객 중에 랜덤으로 뽑아 100개의 고객 데이터를 생성
        custoners_data = np.random.choice(customers, size = 100)

        # 2024년 1월 1일부터 12월 31일까지의 날짜 생성
        date_range = pd.date_range('2024-01-01', '2024-12-31')

        # 생성한 날짜 중 랜덤으로 100개를 뽑아 구매날짜 데이터 생성
        purchase_dates = np.random.choice(date_range, size = 100)

        # 상품의 종류와 단가를 정의한 딕셔너리 생성
        price_map = {
            '셔츠' : 30000,
            '바지' : 52000,
            '신발' : 71000,
            '모자' : 20000,
            '가방' : 43000
        }

        # price_map dict의 키를 뽑아 리스트에 저장
        products = list(price_map.keys())

        # 상품명 리스트에서 랜덤으로 100개를 뽑아 상품 데이터 생성
        products_data = np.random.choice(products, size = 100)
        
        # 상품명에 대응하는 단가를 price_map에서 조회하여 단가 리스트 생성 
        price_data = [price_map[p] for p in products_data]

        # 각 구매 건에 대한 수량(1~5개)을 랜덤으로 생성
        quantity_data = np.random.randint(1,6, size = 100)

        # 앞서 생성한 데이터들을 사용해 데이터 프레임 생성
        self.df = pd.DataFrame({
            'purchase_date' : purchase_dates,
            'product' : products_data,
            'quantity' : quantity_data,
            'price' : price_data
        }, index = custoners_data)

        # quantity(수량) * price(단가)를 통해 sales(매출) 데이터를 생성해 df 데이터 프레임에 추가
        self.df['sales'] = self.df['quantity'] * self.df['price']
        

    def get_monthly_sales(self) :
        # 매출의 총합을 월별로 계산
        monthly_sales = self.df.groupby(self.df['purchase_date'].dt.month)['sales'].sum()
        
        # 계산한 월별 매출 총합을 반환
        return monthly_sales
    
    def get_customers_sales(self) :
        # df의 인덱스(구매자) 를 그룹화해서 매출의 총합을 계산 (구매자별 매출)
        customers_sales = self.df.groupby(self.df.index)['sales'].sum()
        
        # 구매자별 매출을 반환
        return customers_sales
    
    def plot_data(self, monthly_data, customers_sales_data) :

        # 한글 폰트 설정 (맑은 고딕)
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

        # 두 개의 그래프를 담을 피규어 생성
        plt.figure(figsize = (8, 6))

        # 왼쪽에 위치할 월별 매 막대 그래프 생성 (테두리 색 = 검정)
        plt.subplot(1,2,1)
        plt.bar(monthly_data.index, monthly_data.values, edgecolor = 'black')

        # 왼쪽 막대 그래프의 x축 눈금을 1 ~ 12로 설정
        plt.xticks(range(1, 13))
        
        # 왼쪽 막대 그래프의 차트 제목, 축 제목 설정
        plt.title('월별 매출 총합')
        plt.xlabel('월')
        plt.ylabel('매출 총합')

        # 오른쪽에 위치할 고객별 매출 원형 그래프 생성 (시작 각도 = 90도 를 통해 가독성 상승)
        plt.subplot(1,2,2)
        plt.pie(customers_sales_data.values, labels = customers_sales_data.index, autopct = '%1.1f%%', startangle = 90)

        # 오른족 원형 그래프의 차트 제목 설정
        plt.title('고객별 누적 매출')

        # 그래프 출력
        plt.show()

# CustomerSalesAnalysis 인스턴스 생성
sales_data = CustomerSalesAnalysis()

# 월별 매출 반환 메서드를 실행
monthly_data = sales_data.get_monthly_sales()

# 고객별 매출 반환 메서드를 실행
customers_data = sales_data.get_customers_sales()

# 앞서 반환 받은 데이터들을 바탕으로 그래프를 생성
sales_data.plot_data(monthly_data, customers_data)