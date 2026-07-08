import os
import re
import time

# 현재 시간을 이용하기 위해 datetime 라이브러리 사용
from datetime import datetime

# 모니터링할 폴더의 주소 정의
dir_path = './monitor_directory'

# 폴더 내 기존 파일들의 집합 정의
pre_file = set(os.listdir(dir_path))

# 기존 파일들의 이름을 txt파일에 작성
with open('0625 파일 모니터링.txt', 'w', encoding='utf-8-sig') as f:
    for file in pre_file:
        f.write(f'[초기 파일] | {file}\n')
    f.write('\n')

while True:
    # 현재 파일들의 집합 정의
    current_file = set(os.listdir(dir_path))
    # 현재 파일들의 집합과 기존 파일들의 집합 차집합 => 추가된 파일 집합
    new_file = current_file - pre_file

    # 현재 날짜와 시간 정보 정의
    now = datetime.now()
    # 현재 시간 정보 정의
    hour = now.strftime('%H:%M:%S')

    # 추가된 파일이 있는 경우 for 문 호출
    for file in new_file :
        # txt파일을 'a' 모드로 열어 기존 내용에 새로운 내용을 추가
        with open('0625 파일 모니터링.txt', 'a', encoding='utf-8-sig') as f:
            print(f"\n추가된 파일 : {file} | 생성 시간 : {hour}\n")

            # 파일 이름에서 search를 통해 주의 확장자 파일이 있는지 검사
            file_type = re.search(r'(\.py|\.js|\.class)', file)
            
            # 만약 주의 확장자 파일일 경우
            if file_type :
                # 경고문 출력
                print(f'⚠︎ 추가된 파일의 타입이 {file_type.group()} 입니다. ⚠︎\n')
                # 경고 표시와 함께 txt 파일에 파일명 기록
                f.write(f'⚠︎ [추가 파일] | {file} || 생성 시간 : {hour} ⚠︎\n')
            # 주의 확장자 파일이 아닐 경우 txt 파일에 경고 표시 없이 파일명 기록
            else : 
                f.write(f'[추가 파일] | {file} || 생성 시간 : {hour}\n')

        # 모니터링 대상 파일의 전체 경로를 생성
        file_path = os.path.join(dir_path, file)

        # 생성된 파일 경로를 읽기 모드로 열어 파일 내용을 탐지
        with open(file_path, 'r', encoding= 'utf-8-sig') as f:
            lines = f.readlines()

            # enumerate 내장 함수를 통해 각 줄의 인덱스와 내용을 받아옴
            for line_num, line in enumerate(lines) :
                # 주석 정규 표현식
                if re.search(r'^\s*(#|//|--|/\*)', line) :
                    print(f'[주석 탐지] | Line : {line_num} | {line}', end ='')

                # 이메일 정규 표현식
                elif re.search(r'\w+@\w+\.\w+', line) :
                    print(f'[이메일 주소 탐지] | Line : {line_num} | {line}', end ='')

                # SQL 구문 정규 표현식 (주요 구문만 대소문자 구분 없이 탐지)
                elif re.search(r'(?i)(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE)', line) :
                    print(f'[SQL 구문 탐지] | Line : {line_num} | {line}', end ='')
            print('')

    # 현재 파일 목록을 기존 파일 목록으로 초기화
    pre_file = current_file

    # while 문 전체를 1초마다 반복
    time.sleep(1)