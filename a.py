class Hello:
    pass

a = Hello()
setattr(a, 'b', 123)
print(a.b)