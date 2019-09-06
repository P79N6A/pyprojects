hehe = 6
def f():
    global hehe
    hehe = 8
    print hehe


f()
print(hehe)