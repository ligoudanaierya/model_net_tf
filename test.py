class Dog(object):
    name="Jerry"
    def __init__(self,name):
        self.name = name
    @classmethod
    def eat(self):
        print("%s is eating %s" %(self.name,"food"))
    def talk(self):
        print("%s is talking" % self.name)
d = Dog("Tom")
d.eat()
Dog.eat()