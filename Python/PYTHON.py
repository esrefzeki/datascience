###############################################
# PYTHON PROGRAMMING FOR DATA SCIENCE CRASH COURSE
###############################################

# Numbers & String
# Assignments & Variables
# Print Types
# String Methods
# Input
# Data Types (List, Dict, Tuple, Set)
# Functions
# Default Arguments / Parameters
# Return Functions
# Local & Global Variables
# Flow Control & Condition
# Loops
# Enumerate, zip
# List Comprehensions
# Dict Comprehensions
# List & Dict Comprehensions Applications
# Bonus: Decorators

###############################################
# NUMBERS & STRINGS
###############################################

# string
print("Hello AI Era")

# integer
9

# float
9.2

# types
type(9)
type(9.2)
type("123")

###############################################
# ASSIGNMENTS & VARIABLES
###############################################

a = 9
b = 10
a * b
b - a
a * 5

hi = "Hello AI Era"

# del hi
str(9)
int(9.1)
float(9)

###############################################
# PRINT TYPES
###############################################

# print
print("hello ai era")
name = "Rode"
age = 35
print(name, age)

# %
"Name: %s" % name
"Name: %s. Age: %s" % (name, age)

# str.format()
"Name: {}. Age: {}".format(name, age)
person = {"name": "Rode", "age": 35}
"Name:{}. Age: {}".format(person["name"], person["age"])

# fstring
f"Name: {name} Age: {age}"

###############################################
# STRING METHODS
###############################################

name = "Enes"

# len
len(name)
len("MVK")
len("1")

# upper() & lower()
"mvk".upper()
"MVK".lower()

# replace
dir("mvk")
hi = "Hello AI Era"
hi.replace("l", "p")
hi = hi.replace("l", "p")

# split
"Hello AI Era".split()

# strip
" ofofo ".strip()
"ofofo".strip("o")


###############################################
# INPUT
###############################################

number = input()
type(number)
number * 3
# number / 3 # TypeError
int(number) / 3

num1 = int(input())
num2 = int(input())
num1 * num2

###############################################
# DATA STRUCTURES
###############################################

# Numbers (int, float, complex)
# String
# Boolean TRUE-FALSE
# List
# Dictionary
# Tuple
# Set

###############################################
# LIST
###############################################


notes = [1, 2, 3, 4]
names = ["a", "b", "v"]
not_nam = [1, 2, 3, "a", "b"]
notes[0] = 99

# notes[5]

###############################################
# LIST METHODS
###############################################

dir(names)
names.append("MVK")
names.pop(0)

###############################################
# DICTIONARY
###############################################

dictionary = {"REG": "Regression",
              "LOG": "Logistic Regression",
              "CART": "Classification and Reg"}

len(dictionary)

dictionary = {"REG": 10,
              "LOG": 20,
              "CART": 30}

dictionary = {"REG": ["RMSE", 10],
              "LOG": ["MSE", 20],
              "CART": ["SSE", 30]}

dictionary["REG"]

###############################################
# TUPLE
###############################################

t = ("john", "mark", 1, 2)
t[0:3]
names[0] = "999"
t[0] = "999"

###############################################
# FUNCTIONS
###############################################

print("a", "b", sep="_")


def summer(arg1, arg2):
    """
    Sum of two numbers

    args:
    -----
        arg1: int, float
        arg2: int, float

    """
    print(arg1 + arg2)


help(summer)

summer(7, 8)
summer(77, 18)


def multiplication(a, b):
    c = a * b
    print(c)


multiplication(8, 9)


def say_hi():
    print("Merhaba")
    print("Hi")
    print("Hello")


say_hi()

list_store = []


# del liste


def add_element(a, b):
    list_store.append(a * b)
    print(list_store)


add_element(10, 9)

add_element(18, 1)

add_element(180, 1)


###############################################
# DEFAULT ARGUMENTS/PARAMETERS
###############################################


def divide(a, b=1):
    print(a / b)


divide(9, 2)

###############################################
# FUNCTIONS CORRECTLY?
###############################################

# DRY (dont repeat yourself)
# DoT (Do one Thing)
# Modularity

###############################################
# WHEN DO WE NEED TO WRITE A FUNCTION?
###############################################

# varm, moisture, charge

(56 + 15) / 80
(17 + 45) / 70
(17 + 45) / 70


def calculate(varm, moisture, charge):
    print((varm + moisture) / charge)


calculate(90, 12, 12)

###############################################
# RETURN FUNCTIONS
###############################################


# calculate(90, 12, 12) * 10 # type error

type(calculate(90, 12, 12))


def calculate(varm, moisture, charge):
    return int((varm + moisture) / charge)


calculate(90, 12, 12) * 10


def standardization(a, p):
    return a * 10 / 100 * p * p


standardization(10, 9)


def all_calculation(varm, moisture, charge, p):
    a = calculate(varm, moisture, charge)
    b = standardization(a, p)
    print(b * 10)


all_calculation(10, 90, 87, 10)


def all_calculation(varm, moisture, charge, p):
    say_hi()
    a = calculate(varm, moisture, charge)
    b = standardization(a, p)
    print(b * 10)


all_calculation(10, 90, 87, 10)

###############################################
# LOCAL & GLOBAL VARIABLES
###############################################


list_store = [1, 2]


def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)


add_element(10, 8)

###############################################
# FLOW CONTROL & CONDITIONS
###############################################

# 1 == 1


if 1 == 1:
    print("something")

number = 11

if number == 10:
    print("10")


def number_check(number):
    if number == 10:
        print("equal to 10")


number_check(100)

number_check(10)


def number_check(number):
    if number > 10:
        print("greater than 10")
    else:
        print("not greater than 10")


number_check(9)


def number_check(number):
    if number > 10:
        print("greater than 10")
    elif number < 10:
        print("less than 10")
    else:
        print("equal to 10")


number_check(11)

###############################################
# LOOPS
###############################################

# for
students = ["John", "Mark", "Venessa", "Mariam"]

students[0]
students[1]
students[2]
students[3]

for student in students:
    print(student)

for student in students:
    print(student + "_")

for student in students:
    print(f"Old Name: {student}, New Name: {student.upper()}")

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(salary)

1000 * 20 / 100 + 1000


def new_salary(x):
    return x * 20 / 100 + x


new_salary(5000)

for salary in salaries:
    print(int(new_salary(salary)))


def raise_up(x):
    print(x * 10 / 100 + x)


def raise_down(x):
    print(x * 20 / 100 + x)


for salary in salaries:
    if salary >= 3000:
        raise_up(salary)
    else:
        raise_down(salary)


# Try to chance a string like this:
# before: "hi my name is john and i am learning python"
# after: "Hi mY NaMe iS JoHn aNd i aM LeArNiNg pYtHoN"


def alternating(string):
    new_string = ""
    for string_index in range(len(string)):
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        else:
            new_string += string[string_index].lower()

    print(new_string)


alternating("hi my name is john and i am learning python")

###############################################
# BREAK & CONTINUE & WHILE
###############################################

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    if salary == 3000:
        print("break point")
        break
    print(salary)

for salary in salaries:
    if salary == 3000:
        continue
    print(salary)

number = 1
while number < 9:
    print(number)
    number += 1

###############################################
# ENUMERATE: LOOP WITH AUTOMATIC COUNTER (INDEXER)
###############################################

# Divide students into 2 group based on their index number (even or odd)

students = ["John", "Mark", "Venessa", "Mariam"]

for student in students:
    print(student.upper())

for index, student in enumerate(students):
    print(index, student)

A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)

# with function
students = ["John", "Mark", "Venessa", "Mariam"]


def divide_students(students):
    groups = [[], []]
    for index, student in enumerate(students, 1):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)


divide_students(students)


def alternating(string):
    new_string = ""
    for index, letter in enumerate(string):
        if index % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)


alternating("hi my name is john and i am learning python")

###############################################
# ZIP
###############################################

students = ["John", "Mark", "Venessa", "Mariam"]
departments = ["mathematics", "statistics", "physics", "astronomy"]
ages = [23, 30, 26, 22]
print(list(zip(students, departments, ages)))


###############################################
# LAMBDA, MAP, FILTER, REDUCE
###############################################

def summer(a, b):
    return a + b


summer(0, 1)

new_sum = lambda a, b: a + b

new_sum(9, 1)

# MAP
salaries = [1000, 2000, 3000, 4000, 5000]


def new_salary(x):
    return x * 20 / 100 + x


new_salary(1000)

for salary in salaries:
    print(new_salary(salary))

list(map(new_salary, salaries))
list(map(lambda x: x * 20 / 100 + x, salaries))
list(map(lambda x: x ** 2, salaries))
list(map(lambda x: x.upper(), "john"))

# FILTER
list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(filter(lambda x: x % 2 == 0, list_store))

# REDUCE
from functools import reduce
list_store = [1, 2, 3, 4]
reduce(lambda a, b: a + b, list_store)


###############################################
# LIST COMPEHENSIONS
###############################################

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(new_salary(salary))

null_list = []

for salary in salaries:
    null_list.append(new_salary(salary))

null_list = []

for salary in salaries:
    if salary > 3000:
        null_list.append(new_salary(salary))
    else:
        null_list.append(new_salary(salary * 2))

# list comp
salaries = [1000, 2000, 3000, 4000, 5000]

[salary * 2 for salary in salaries]

[salary * 2 for salary in salaries if salary < 3000]

[salary * 2 if salary < 3000 else salary * 0 for salary in salaries]

[new_salary(salary * 2) if salary < 3000 else new_salary(salary) for salary in salaries]

students = ["John", "Mark", "Venessa", "Mariam"]
students_no = ["John", "Venessa"]

[student.upper() if student not in students_no else student.lower() for student in students]

[student.lower() if student in students_no else student.upper() for student in students]

###############################################
# DICT COMPREHENSIONS
###############################################

dictionary = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
dictionary["a"]
dictionary["b"]

dictionary.keys()
dictionary.values()
dictionary.items()

new_dict = {k: v ** 2 for (k, v) in dictionary.items()}

{k * 2: v for (k, v) in dictionary.items()}

numbers = range(10)
new_dict = {}

for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n ** 2

{n: n ** 2 for n in numbers if n % 2 == 0}

###############################################
# LIST & DICT COMPREHENSION APPLICATIONS
###############################################

###############################################
# Changing the variable names of a data set
###############################################

# before:
# ['total', 'speeding', 'alcohol', 'not_distracted', 'no_previous', 'ins_premium', 'ins_losses', 'abbrev']

# after:
# ['TOTAL', 'SPEEDING', 'ALCOHOL', 'NOT_DISTRACTED', 'NO_PREVIOUS', 'INS_PREMIUM', 'INS_LOSSES', 'ABBREV']


import seaborn as sns
df = sns.load_dataset("car_crashes")
df.head()

df.columns

for col in df.columns:
    print(col.upper())

A = []

for col in df.columns:
    A.append(col.upper())

type(A)

type(df.columns)

df.columns = A


df = sns.load_dataset("car_crashes")
df.columns = [col.upper() for col in df.columns]
df.columns


###############################################
# We want to write "FLAG" at the beginning of variables that contain the word "INS" in their name
# and NO FLAG to the others.
###############################################

# before:
# ['TOTAL', 'SPEEDING', 'ALCOHOL', 'NOT_DISTRACTED', 'NO_PREVIOUS', 'INS_PREMIUM', 'INS_LOSSES', 'ABBREV']

# after:
# ['NO_FLAG_TOTAL',
#  'NO_FLAG_SPEEDING',
#  'NO_FLAG_ALCOHOL',
#  'NO_FLAG_NOT_DISTRACTED',
#  'NO_FLAG_NO_PREVIOUS',
#  'FLAG_INS_PREMIUM',
#  'FLAG_INS_LOSSES',
#  'NO_FLAG_ABBREV']


[col for col in df.columns if "INS" in col]

["FLAG_" + col for col in df.columns if "INS" in col]

["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]

df.columns = ["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]


###############################################
# We want to write "CAT" at the beginning of categorical variable names
###############################################

# before:
# ['total', 'speeding', 'alcohol', 'not_distracted', 'no_previous', 'ins_premium', 'ins_losses', 'abbrev']

# after:
# ['TOTAL',
#  'SPEEDING',
#  'ALCOHOL',
#  'NOT_DISTRACTED',
#  'NO_PREVIOUS',
#  'INS_PREMIUM',
#  'INS_LOSSES',
#  'CAT_ABBREV']


df = sns.load_dataset("car_crashes")

[col for col in df.columns if df[col].dtype == "O"]

["CAT_" + col.upper() if df[col].dtype == "O" else col.upper() for col in df.columns]

# df.columns = ["CAT_" + col.upper() if df[col].dtype == "O" else col.upper() for col in df.columns]


###############################################
# We want to create a dictionary as follows:
###############################################

# Output
# {'total': ['mean', 'min', 'max', 'var'],
#  'speeding': ['mean', 'min', 'max', 'var'],
#  'alcohol': ['mean', 'min', 'max', 'var'],
#  'not_distracted': ['mean', 'min', 'max', 'var'],
#  'no_previous': ['mean', 'min', 'max', 'var'],
#  'ins_premium': ['mean', 'min', 'max', 'var'],
#  'ins_losses': ['mean', 'min', 'max', 'var']}

df = sns.load_dataset("car_crashes")
df.columns

# first solution
num_cols = [col for col in df.columns if df[col].dtype != "O"]
agg_list = ['mean', 'min', 'max', 'sum']

dictionary = {}

for col in num_cols:
    dictionary[col] = agg_list

# second solution
new_dict = {col: agg_list for col in num_cols}

# what can be done with it?
df.groupby("abbrev").agg(new_dict)

# any other example?
df = sns.load_dataset("tips")
num_cols = [col for col in df.columns if df[col].dtype in [int, float]]
new_dict = {col: agg_list for col in num_cols}
df.groupby("time").agg(new_dict)


###############################################
# We want to create a dictionary as follows:
###############################################

# before:

# {'total_bill': ['mean', 'min', 'max', 'sum'],
#  'tip': ['mean', 'min', 'max', 'sum'],
#  'size': ['mean', 'min', 'max', 'sum']}

# after:

# {'total': ['total_mean', 'total_min', 'total_max', 'total_var'],
#  'speeding': ['speeding_mean', 'speeding_min', 'speeding_max', 'speeding_var'],
#  'alcohol': ['alcohol_mean', 'alcohol_min', 'alcohol_max', 'alcohol_var']


df = sns.load_dataset("car_crashes")
num_cols = [col for col in df.columns if df[col].dtype != "O"]
agg_list = ["mean", "min", "max", "sum"]
new_dict = {col: agg_list for col in num_cols}

new_dict = {col.upper(): [str(col) + "_" + c for c in agg_list] for col in num_cols}


###############################################
# We want to create a dictionary as follows:
###############################################

# AMAC: Bir listenin ilk elemanını key diğer eleman setini de value olarak atamak isteyelim.

# before
#    total  speeding  alcohol  not_distracted  no_previous
# 0   18.8     7.332    5.640          18.048       15.040
# 1   18.1     7.421    4.525          16.290       17.014
# 2   18.6     6.510    5.208          15.624       17.856
# 3   22.4     4.032    5.824          21.056       21.280
# 4   12.0     4.200    3.360          10.920       10.680


# after:
# {18.8: [7, 5, 18, 15],
#  18.1: [7, 4, 16, 17],
#  18.6: [6, 5, 15, 17],
#  22.4: [4, 5, 21, 21],
#  12.0: [4, 3, 10, 10]}


df = sns.load_dataset("car_crashes")
num_cols = [col for col in df.columns if df[col].dtype != "O"]
df["total"].head()
df[num_cols].head()
new_df = df[num_cols].iloc[0:5, 0:5]

{new_df.values[i, :][0]: [int(s) for s in new_df.values[i, :][1:]] for i in range(len(new_df))}


###############################################
# BONUS: DECORATORS: decorators wrap a function, modifying its behavior.
###############################################

# Functions as arguments
def say_hi(name):
    return f"Hello {name}!"

say_hi("Sinan")

def talk(func):
    return func("Sinan")


talk(say_hi)

def say_hey(name):
    return f"Hey hey {name}!"

talk(say_hey)


# Inner Functions: define functions inside other functions
def meeting():
    print("Hi guys! I am the host")

    def hi_from_john():
        print("Hello!")

    def hi_from_erik():
        print("Hey hey!")

    hi_from_john()
    hi_from_erik()



meeting()

# hi_from_erik()


# Returning Functions From Functions
def meeting():
    print("Hi guys! I am the host")

    def hi_from_john():
        print("Hello!")

    def hi_from_erik():
        print("Hey hey!")

    hi_from_john()
    hi_from_erik()

    return hi_from_john, hi_from_erik

hi_from_john, hi_from_erik = meeting()

hi_from_erik()


# Basic Decorator Template
# decorators wrap a function, modifying its behavior.
def my_decorator(func):
    def wrapper():
        print("Do something before the function call.")
        func()
        print("Do something after the function call.")
    return wrapper


def say_hi():
    print("Hello!")

say_hi()

say_hi = my_decorator(say_hi)

say_hi()

# Other usage
@my_decorator
def say_hi():
    print("Hello!")

say_hi()

say_hi
say_hi.__name__

# naming for actual function
import functools

def my_decorator(func):
    @functools.wraps(func)
    def wrapper():
        print("Do something before the function call.")
        func()
        print("Do something after the function call.")
    return wrapper


@my_decorator
def say_hi():
    print("Hello!")

say_hi.__name__

# Decorating Functions With Arguments
@my_decorator
def say_hi(name):
    print("Hello!", name)

say_hi("Sinan")


# Final
def my_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("Do something before the function call.")
        func(*args, **kwargs)
        print("Do something after the function call.")
    return wrapper


@my_decorator
def say_hi(name):
    print("Hello!", name)

say_hi("Sinan")


# PARTY BOY DECORATOR
def party_boy(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("Welcome our home!")
        func(*args, **kwargs)
        print("Bye bye!")
    return wrapper


@party_boy
def do_party():
    print("eat, meet, play")

do_party()


@party_boy
def do_corona_party():
    print("do something silly")

do_corona_party()

# REAL WORLD EXAMPLE: TIMING FUNCTION
import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()

        value = func(*args, **kwargs)

        end_time = time.perf_counter()

        run_time = end_time - start_time

        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


@timer
def sum_of_even_numbers(numbers):
    even_sum = 0
    for number in range(numbers):
        if number % 2 == 0:
            even_sum += number
    return even_sum


sum_of_even_numbers(10)
sum_of_even_numbers(100*100)
sum_of_even_numbers(100*10000)


###############################################
# ÖDEVLER:
# ZORUNLU ODEV 1: Komut satırından Python kodu çalıştırma.
# ZORUNLU ODEV 2: Veri Okuryazarlığı Sertifika
# ZORUNLU ODEV 3: List Comprehension Applications
# KEYFİ ÖDEV 1: "setler" konusu izlenecek ve Python 101, 102, 103 sertifikaları alınacak.
# KEYFİ ÖDEV 2: Python scriptine konsoldan arguman vermek konusunu araştırınız ve bir örnek uygulama yapınız.
###############################################


###############################################
# ZORUNLU ODEV 1: Komut satırından Python kodu çalıştırma.
###############################################

# Yazacak olduğunuz "py" uzantılı bir python dosyasını komut satırından çalıştırmanız beklenmektedir.
# Örneğin hi.py isimli bir dosyanız olsun ve içinde print("isim soy isim") kodu olsun.
# Bilgisayarınızın konsolunu açıp konsoldan hi.py dosyasının olduğu dizine gelip buradan "python hi.py" kodunu
# çalıştırdığınızda ekranınızda "isim soy isim" yazmalı.
# Adım adım nasıl yapılacağı anlatılmıştır.

# Adım 1: PyCharm'da "hi.py" isminde python dosyası oluştur.
# Adım 2: Bu dosyanın içirisine şu kodu kendine göre yaz ve kaydet: print("Ben Sinan Artun ÖDEV tamam, bu çok kolaymış")
# Adım 3: Şimdi konsoldan "hi.py" dosyasının olduğu dizine (klasöre) gitmen gerekiyor.
# Neyse ki PyCharm ile bu çok kolay. Sol tarafta yer alan menüde hi.py dosyası hangi klasördeyse
# o klasöre sağ tuş ile tıklayıp şu seçimi yap: "open in > terminal".
# PyCharm'ın alt tarafında terminal ekranı açılacak. Şu anda hi.py dosyası ile aynı dizindesin (klasörde).
# Adım 4: Konsolda şu kodu yazmalısın: python hi.py
# Adım 5: Çıktını ekran görüntüsünü alıp grubunda paylaş.


###############################################
# ZORUNLU ODEV 2: Veri Okuryazarlığı Sertifika
###############################################

# Aşağıdaki adreste yer alan "Veri Okuryazarlığı" sınavına girilecek ve sertifika alınacak.
# https://gelecegiyazanlar.turkcell.com.tr/konu/veri-okuryazarligi


###############################################
# ZORUNLU ODEV 3: List Comprehension Applications
###############################################

###############################################
# Görev 1: car_crashes verisindeki numeric değişkenlerin isimlerini büyük harfe çeviriniz ve başına NUM ekleyiniz.
###############################################

# Veri setini baştan okutarak aşağıdaki çıktıyı elde etmeye çalışınız.

# ['NUM_TOTAL',
#  'NUM_SPEEDING',
#  'NUM_ALCOHOL',
#  'NUM_NOT_DISTRACTED',
#  'NUM_NO_PREVIOUS',
#  'NUM_INS_PREMIUM',
#  'NUM_INS_LOSSES',
#  'ABBREV']

# Notlar:
# Numerik olmayanların da isimleri büyümeli.
# Tek bir list comp yapısı ile yapılmalı.


###############################################
# Görev 1 Çözüm
###############################################



###############################################
# Görev 2: İsminde "no" BARINDIRMAYAN değişkenlerin isimlerininin SONUNA "FLAG" yazınız.
###############################################

# Tüm değişken isimleri büyük olmalı.
# Tek bir list comp ile yapılmalı.

# Beklenen çıktı:

# ['TOTAL_FLAG',
#  'SPEEDING_FLAG',
#  'ALCOHOL_FLAG',
#  'NOT_DISTRACTED',
#  'NO_PREVIOUS',
#  'INS_PREMIUM_FLAG',
#  'INS_LOSSES_FLAG',
#  'ABBREV_FLAG']


###############################################
# Görev 2 Çözüm
###############################################


###############################################
# Görev 3: Aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin isimlerini seçerek yeni bir df oluşturunuz.
###############################################

# df.columns
# og_list = ["abbrev", "no_previous"]

# Önce yukarıdaki listeye göre list comprehension kullanarak new_cols adında yeni liste oluşturunuz.
# Sonra df["new_cols"] ile bu değişkenleri seçerek yeni bir df oluşturunuz adını new_df olarak isimlendiriniz.

# Beklenen çıktı:

# new_df.head()
#
#    total  speeding  alcohol  not_distracted  ins_premium  ins_losses
# 0 18.800     7.332    5.640          18.048      784.550     145.080
# 1 18.100     7.421    4.525          16.290     1053.480     133.930
# 2 18.600     6.510    5.208          15.624      899.470     110.350
# 3 22.400     4.032    5.824          21.056      827.340     142.390
# 4 12.000     4.200    3.360          10.920      878.410     165.630

###############################################
# Görev 3 Çözüm
###############################################


###############################################
# KEYFİ ÖDEV 1: (TAKİP EDİLMEYECEK - SORULAR YANITLANAMAYACAKTIR)
# Geleceği Yazanlarda "set" konusu izlenecek ve Python 101,102 ve 103 sertifikaları alınacak.
###############################################

###############################################
# KEYFİ ÖDEV 2: (TAKİP EDİLMEYECEK - SORULAR YANITLANAMAYACAKTIR)
# Python scriptine konsoldan arguman vermek konusunu araştırınız ve bir örnek uygulama yapınız.
# İpucu: argparse-CLIs (command line interfaces)
###############################################