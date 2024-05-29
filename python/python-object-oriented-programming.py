# when a class inherits from another class, you do not need to override the init function

class Vehicle:

    def __init__(self):
        self.driveable = True
    
    def drive(self):
        print("Vehicle is driving")

# inherits from Vehicle - will inherit method and call Vehicle init function
class Car(Vehicle):
    pass

test = Car()
print(test.driveable) # True
test.drive() # Vehicle is driving


# With the super() function, you can gain access to inherited methods that have been overwritten in a class object.
#  For example, we may want to override one aspect of the parent method with certain functionality, but then call the rest of the original parent method to finish the method.

class Fish:

    def __init__(self, type="Fish"):
        self.isFish = True
        self.type = type

# inherits from Vehicle - will inherit method and call Vehicle init function
class Trout(Fish):
    
    def __init__(self):
        self.isTrout = True
        super().__init__("Trout") # NOTE that we do not pass 'self' in here

trout = Trout()
print(trout.isFish) # True
print(trout.isTrout) # True
print(trout.type) # "Trout"

class Animal:
    pass

# you can also inherit from mutliple classes like
class RainbowTrout(Trout, Animal):
    pass
    # note this is a bas example becuase Trout might also inherit from Animal as well, so we would not need RainbowTrout to inherit
    # from Animal. The two (or more) classes are usually completely independant of each other.



# more super override examples
class ChessPiece:

    def __init__(self):
        self.color = "white"

    def move(self, from_, to_):
        print("piece moved from", from_, "to", to_)

class Bishop(ChessPiece):

    def move(self):
        super().move("c1", "b4")
        print("diagonally")
        
b = Bishop()
b.move() # piece moved from c1 to b4 diagonally


