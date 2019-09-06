# -*- coding: utf-8 -*-
class F(object):
    """
    test：测试attribute/getattr方法类
    """
    def __init__(self):
        self.name = 'A'

    def hello(self):
        print('hello')

    def __getattribute__(self, item):
        """
        how to use：当访问对象的属性或者是方法的时候触发,即 ob.name/ obj.func() 时会调用
        question：为什么 print (获取属性，方法,item) -->  ('\xe8\x8e\xb7\xe5\x8f\x96\xe5\xb1\x9e\xe6\x80\xa7\xef\xbc\x8c\xe6\x96\xb9\xe6\xb3\x95', 'name')
                     而 print '获取属性，方法'+item -->  获取属性，方法name
        """
        print '获取属性，方法：'+item
        return object.__getattribute__(self, item)

    def __getattr__(self, item):
        """
        how to use：拦截运算(obj.xx)，对没有定义的属性名和实例,会用属性名作为字符串调用这个方法
        question：还是不太懂？？？
        """
        if item == 'age':
            print '拦截属性：' + item
            return 40
        else:
            raise AttributeError('没有这个属性')


a = F()
name = a.name
a.hello()
print a.age