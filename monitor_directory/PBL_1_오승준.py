import re
import os
import csv
from collections import Counter

ip_list = []    # 로그에서 추출한 ip 주소들을 저장할 List
top3_ip = []     # 빈도수 3위 이내의 ip 주소들을 저장할 List

log_dir = input('로그 파일의 경로를 입력해주세요.')

# 입력 받은 경로에 파일이 있는지 확인하고, 없으면 exit()
if not os.path.exists(log_dir) :
    print("해당 경로에 파일이 존재하지 않습니다.")
    exit()

# 파일을 읽기 모드로 열고, 한 줄씩 처리함
with open(log_dir, 'r') as f:
    for line in f:
        # Ip 정규 표현식 (예 : 192.168.0.1)
        # \d{1,3} = 1~3자리의 숫자
        # (\.\d{1,3}){3} = .숫자 표현을 3번 반복
        ip = re.search(r'\d{1,3}(\.\d{1,3}){3}', line)

        # ip가 false(None) 이 아니라면, ip 주소를 리스트에 추가
        if ip :
            ip_list.append(ip.group())

# Counter 함수를 통해 상위 3개 Ip 주소를 뽑아 리스트에 할당
top3_ip = Counter(ip_list).most_common(3)    
print(top3_ip)

# 추출 결과를 CSV 파일로 저장 (utf-8-sig 인코딩 방식)
with open('ip_analysis.txt', 'w', newline = '', encoding = 'utf-8-sig') as f:
    writer = csv.writer(f)

    # 헤더 작성
    writer.write(['Ip', 'Count'])
    # (Ip주소, 빈도) 데이터를 csv 파일에 한 줄씩 작성
    for row in top3_ip :
        writer.writerow(row)