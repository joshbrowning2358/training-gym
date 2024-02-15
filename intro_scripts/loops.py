from time import sleep

students = ["Breck", "Diego", "Maevery", "Stella"]
for student in students:
    print("Welcome to class, " + student + "!")


for i in range(20):
    print("Hi, I'm an annoying duck!")


i = 0
while i < 20:
    print("Hi, I'm an annoying duck!")
    i = i + 1


scream = "AAA"
while len(scream) < 100:
    print(scream + "!!")
    scream = scream * 2


while True:
    print("This is the song that never ends")
    print("Yes, it goes on and on my friends")
    print("Some people started singing it not knowing what it was")
    print("And they'll continue singing it forever just because...")
    print()
    sleep(7)


i = 0
while i < 20:
    print("Hi, I'm an annoying duck!")
