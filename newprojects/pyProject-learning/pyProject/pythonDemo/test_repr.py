class Student(object):
    """
    question：不是太明白所谓的在交互式环境中会使用到__repr__，应用场景在哪里呢？
    """
    def __init__(self, name):
        self.name = name

    def __str__(self):
        print '__str__'
        return 'name of this student is %s'%self.name

    def __repr__(self):
        print '__repr__'
        return 'name of this student is %s'%self.name

print Student('Xiaoming')
