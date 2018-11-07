class Bus:
    def __init__(self):
        self.speed = 0
        self.odometer = 0
        self.time = 0

    def say_state(self):
        print("I'm going {} kph!".format(self.speed))

    def accelerate(self):
        self.speed += 5

    def brake(self):
        self.speed -= 5

    def step(self):
        self.odometer += self.speed
        self.time += 1

    def average_speed(self):
        return self.odometer / self.time


if __name__ == '__main__':
    my_bus = Bus()
    print("I'm a car!")
    while True:
        action = input("What should I do? [A]ccelerate, [B]rake, "
            "show [O]dometer, or show average [S]peed?").upper()
        if action not in "ABOS" or len(action) != 1:
            print("I don't know how to do that")
            continue
        if action == 'A':
            my_bus.accelerate()
        elif action == 'B':
            my_bus.brake()
        elif action == 'O':
            print("The car has driven {} kilometers".format(my_bus.odometer))
        elif action == 'S':
            print("The car's average speed was {} kph".format(my_bus.average_speed()))
        my_bus.step()
        my_bus.say_state()