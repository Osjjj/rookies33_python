# 모듈 : 외부에 있는 모듈을 사용

# import mod1
# import mod1 as mo
# from mod1 import PI, add, sub
from mod1 import *
# import mod.mod2 as o
# from mod.mod2 import PI, add, sub
# from mod.mod2 import *

print('__name__ :', __name__)

print(PI)
print(add(10,2))
print(sub(10,2))


# print(mo.mod1.add(1,3))
# print(mod1.add(1,3))
# print(mod1.sub(3,1))

def start() :
    print('start')

if __name__=='__main__' :
    start()