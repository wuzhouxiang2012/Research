class A():
    def __init__(self):
        self.currentIndex = 0
        self.list = [1,2,3]
        self.currentValue = self.list[self.currentIndex]

a = A()
print(a.currentValue)
a.currentIndex += 1
print(a.currentValue)