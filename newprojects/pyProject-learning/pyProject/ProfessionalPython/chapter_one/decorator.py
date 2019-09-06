# simple decorate start---------------
# def decorated_by(func):
#     func.__doc__ += '\nDecorated by decorated_by.'
#     return func
#
#
# def decorated_by_also(func):
#     func.__doc__ += '\n also decorated'
#
#
# # @decorated_by_also
# # @decorated_by
# def add(x, y):
#     """Return the sum of a and y"""
#     return x + y
#
# add = decorated_by_also(decorated_by(add))
#
# help(add)
# simple decorate end---------------


# register decorate start---------------
# class Registry(object):
#     def __init__(self):
#         self._functions = []
#
#     def register(self, decorated):
#         self._functions.append(decorated)
#         return decorated
#
#     def run_all(self, *args, **kwargs):
#         return_value = []
#         for func in self._functions:
#             return_value.append(func(*args, **kwargs))
#         return return_value
#
#
# a = Registry()
# b = Registry()
#
#
# @a.register
# def foo(x=3):
#     return x
#
#
# @b.register
# def bar(x=5):
#     return x
#
#
# @a.register
# @b.register
# def baz(x=7):
#     return x
#
# print(a.run_all())
# print(b.run_all())
# print(a.run_all(x=4))
# register decorate end---------------


def requires_ints(decorated):
    def inner(*args, **kwargs):
        kwarg_values = [i for i in kwargs.values()]
        for arg in list(args) + kwarg_values:
            if not isinstance(arg, int):
                raise TypeError('%s only accepts integers as arguments.' % decorated.__name__)
        return decorated(*args, **kwargs)
    return inner


@requires_ints
def foo(x, y):
    """return the sum of x and y"""
    return x + y

# help(foo)
print(foo(3,"3"))