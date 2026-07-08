# 모듈 : 변수들, 함수들, 클래스들 가지는 파일
print('mod1 실행 시작')
print('mod1 새행 -> __name__ :', __name__)

PI = 3.14

def add(n1, n2) :
    return n1+n2

def sub(n1, n2) :
    return n1 - n2

print('mod1 실행 끝')

if __name__=='__main__' :
    # 현재 모듈의 함수를 테스트
    print(PI)
    print(add(10,30))
    print(sub(20,10))