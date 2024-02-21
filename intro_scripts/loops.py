from time import sleep

students = ["Breck", "Diego", "Maevery", "Stella", "Josh"]
students[1]
students[4]
for x in students:
    print("Welcome to class, " + x + "!")


for i in range(1):
    print("Hi, I'm an annoying duck!")


for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]:
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
    sleep(1)


i = 0
while i < 20:
    print("Hi, I'm an annoying duck!")
