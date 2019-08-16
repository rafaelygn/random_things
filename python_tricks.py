
'''
String
    f-String, r-string

Tuples
    unpack, * and _

Functional Programming
    Function
        lambda, map, filter
    Comprehension
        List , Dict

Generators
    Hack a number_password

        Testar todas as possibilidades
        porém sem precisar ter memória

Decorator
    timeit, tryit, loggit, curry

        Mostrar alguns exemplos de decorators que são bastante
        utilizados

Class
    speciak methods (__init__, __len__, __add__, ...)

        Por que somar duas listas, dois inteiros e dois arrays
        dão resultados diferentes?
        Podemos utilizar esses recursos na nossa classe!

    decorators again!

        Tenho minha classe pronta, mas queria alterar um formato
        de input da minha classe, mas sem alterar o código dela


'''

# Import packages
import os
import time
import numpy as np
from collections import namedtuple
from sys import getsizeof
from getpass import getpass


# --------------------------------------
# Strings
# --------------------------------------

# Setting some variables
country = 'Brasil'
cc = 'BR'
pop = 220_000_000
pib = 2_056_000_000_000

# F-STRING
msg_f = (
    f'O {country}({cc}) com a população de {pop:,} teve em 2017 '
    f'o PIB de U$ {(pib/10**12):.2f} trihões. \nPoranto a renda per '
    f'capita do {country} é {pib/pop:,} dólares'
)
print(msg_f)

# R-STRING
msg_r = (
    r'O {country}({cc}) com a população de {pop:,} teve em 2017 '
    r'o PIB de U$ {(pib/10**12):.2f} trihões. \nPoranto a renda per '
    r'capita do {country} é {pib/pop:,} dólares'
)
print(msg_r)

# --------------------------------------
# Tuples
# --------------------------------------

tup = (1, 2, 3, 4)
type(tup)


def silly_function():
    return 1, 2, 3, 4


# Let's check if they are the same
print(tup == silly_function())

# unpack tuples
a, b, c, d = silly_function()
print(a, b, c, d)

# ignore some item
a, b, _, d = silly_function()
print(a, b, d)

# take the last one
_, filename = os.path.split('/workspace/yoshraf/project/model/lgbm_model.pkl')
print(filename)

# * operator
a, b, *c = silly_function()
print(a, b, c)

# c, now, is a list!

# NAMED TUPLES
# Here we create a Named Tuple Class
# So, we must named it and set its methods
City = namedtuple('City', 'name country population coordinates')
tokyo = City('Tokyo', 'JP', 36_933, (35.689, 139.691))
print(tokyo.country, tokyo.coordinates)

# --------------------------------------
# Functional Programming
# --------------------------------------

original_list = [1, 2, 3, 4, 5]

# FUNCTIONS MAP, FILTER and LAMBDA
# Usual way
new_list = []
for element in original_list:
    new_list.append(element**2)

# Map & lambda
def lf_map(x): return x**2
list(map(lf_map, original_list))

# filter & Lambda
def lf_f(x): return x % 2 == 0
list(filter(lf_f, original_list))

# Map & Filter & lambda
list(map(lf_map, list(filter(lf_f, original_list))))

# LIST COMPREHENSION
# Sintaxe: [ EXPRESSION for element in list if CONDITION]
new_list = [element**2 for element in original_list]
print(new_list)

# List comprehension with condition
new_list = [element**2 for element in original_list if element % 2 == 0]
print(new_list)

# List comprehension with if and eose
new_list = [element**2 if element %
            2 == 0 else np.nan for element in original_list]
print(new_list)

# SOME PROBLEM
# 1. Label axis X in date format {YYYYMM}
[str(year*100 + month) for year in list(range(2015, 2020))
 for month in list(range(1, 13))]

# 2. Exploration to Grid Search
penalty_list = ['l1', 'l2']
c_list = [0.001, 0.01, 0.1, 1]
class_weight_list = [None, 'balanced']

# Usual Way
random_search = []
for p in penalty_list:
    for c in c_list:
        for w in class_weight_list:
            random_search.append((p, c, w))
print(random_search)

# List Comprehension with more than 1 list
random_search = [(p, c, w) for p in penalty_list
                 for c in c_list
                 for w in class_weight_list]
print(random_search)

# Until now, we iterate over lists sequentially
# But if, we want to iterate over 2 lists in parallel
# Use zip function!
for string, number in zip(['one', 'two', 'three'], [1, 2, 3]):
    print(string, number)

# DICTIONARIES
# Transform 2 list into a dict
dict_from_lists = dict(zip(['one', 'two', 'three'], [1, 2, 3]))

dial_codes = [
    (86, 'China'),
    (91, 'India'),
    (1, 'United States'),
    (62, 'Indonesia'),
    (55, 'Brazil'),
    (92, 'Pakistan'),
    (880, 'Bangladesh'),
    (234, 'Nigeria'),
    (7, 'Russia'),
    (81, 'Japan')
]

# Create a dictionary from tuples inside a list
country_code_dict = {country: code for code,
                     country in dial_codes if code < 50}

# Rather than use zip, you can use dict comprehension
{code: country.upper() for country, code in country_code_dict.items()}

# But if you don't want a dictionary, you can iterate in a for loop
# Iterate over its keys
for key in country_code_dict.keys():
    print(key)

# Iterate over its values
for value in country_code_dict.values():
    print(value)

# Iterate over both: key and value
for key, value in country_code_dict.items():
    print(key, value)

# --------------------------------------
# Generators
# --------------------------------------

# Generator doesn't hold the result in memory!
# It yields one result at time

# Setting a usual function
# That return a list


def square_numbers(original_list: list) -> list:
    result = []
    for element in original_list:
        result.append(element**2)
    return result


# This functions returns a generator
def gen_square_numbers(original_list: list) -> int:
    for element in original_list:
        yield (element**2)


# Using a Generator
gen_result = square_numbers(original_list)
print(next(gen_result))
print(next(gen_result))
print(next(gen_result))

for element in gen_result:
    print(element)

# We can write the generator as list comprehension!
gen_result = (element**2 for element in original_list)
print(gen_result)

# Let's hack!
password = getpass(prompt='Password: ')
# password = 10_000


def hack_password(iterator, see_size=False):
    if see_size:
        print(f'Size of iterator: {getsizeof(iterator)/10**3:.2f} KB')
    for try_pssw in iterator:
        if try_pssw == password:
            print(f'The password is {try_pssw}')
            return try_pssw  


gen_password = (try_pssw for try_pssw in range(10**10))
lis_password = [try_pssw for try_pssw in range(10**5)]

# Let's Compare a list and a generator!
ti = time.perf_counter()
hack_password(lis_password, see_size=True)
tf = time.perf_counter()
print(f'It took {(tf -ti):.2f} seconds')

ti = time.perf_counter()
hack_password(gen_password, see_size=True)
tf = time.perf_counter()
print(f'It took {(tf -ti):.2f} seconds')


# --------------------------------------
# ARGS AND KWARGS
# --------------------------------------

# args and kwargs
def silly_function_2(*args, **kwargs):
    print('*args: ',*args)
    print('args: ',args)
    print('kwargs: ', kwargs)
    return None
    
silly_function_2('lala', 'lele', 'lili', kw=1, kw2=2)

def train_one_model(log_time=None, *args, **kwargs):
    if log_time:
        print('Do log time')
    print(kwargs)
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(**kwargs)
    print(clf)
    return None

train_one_model(log_time=True,
    criterion='gini', max_depth=10, min_samples_leaf=15)


def train_many_models(**kwargs):
    print(kwargs)
    if 'log_time' in kwargs:
        print('There is log time in **kwargs')
        print(kwargs['log_time'])
    if 'tree_model' in kwargs:
        print('There is tree model in ** kwargs')
        print(kwargs['tree_model'])
        silly_function_2(**kwargs['tree_model'])
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(**kwargs['tree_model'])
        print(clf)
    if 'logit_reg' in kwargs:
        pass 
    return None

train_many_models_kwargs = {
    'log_time': True,
    'tree_model': {'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 15},
    'logit_reg': {'l1': 0, 'l2': 0}
}

train_many_models(
    log_time= True,
    tree_model= {'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 15},
    logit_reg= {'l1': 0, 'l2': 0})
train_many_models(**train_many_models_kwargs)

# --------------------------------------
# DECORATORS
# --------------------------------------

'''
Decoradores de função nos permite marcar funções no código fonte para modificar
o seu comportamento de alguma maneira. É um recurso poderoso, mas dominá-lo
exige compreender as clouseres

Um decorador é um invocável (callable) que aceita outra função como argumento
(a função decorada). O decorador pode realizar algum processamento com a 
função decorada e devolvê-la ou substituí-la por outra função ou um objeto
'''

# timeit decorator
def timeit(method: callable) -> callable:

    def time_method(*args, **kwargs):
        ti = time.perf_counter()
        method_result = method(*args, **kwargs)
        tf = time.perf_counter()
        print(f'The {method.__name__} function took {(tf -ti):.2f} s')
        return method_result
    return time_method

# Altering original function
# Now we can measure how long a function lasts
@timeit
def hack_password(iterator, see_size=False):
    if see_size:
        print(f'Size of iterator: {getsizeof(iterator)/10**3:.2f} KB')
    for try_pssw in iterator:
        if try_pssw == password:
            print(f'The Password is {try_pssw}')
            break

# Let's experiment it
hack_password(lis_password, see_size=True)


'''
O decorador timeit implementado tem algumas deficiências:
Ele não aceita argumentos nomeados e mascara __name__
e __doc__ da função decorada.

Para entender isso o @timeit

@timeit
def hack_password(arg):
    [do something]
    return result

é equivalente á:

def hack_password(arg):
    [do something]
    return result

hack_password = timeit(hack_password)
'''

print(hack_password.__name__)
print(hack_password.__doc__)

'''Para corrigir isso podemos utilizar o functools.wraps
Assim os atributos importantes relevantes também são copiados'''

from functools import wraps 

def timeit(method: callable) -> callable:
    @wraps(method)
    def time_method(*args, **kwargs):
        ti = time.perf_counter()
        method_result = method(*args, **kwargs)
        tf = time.perf_counter()
        print(f'The {method.__name__} function took {(tf -ti):.2f} s')
        return method_result
    return time_method

# Altering original function
# Now we can measure how long a function lasts
@timeit
def hack_password(iterator, see_size=False):
    '''
    Through an iterator we discover a numeric password
    '''
    if see_size:
        print(f'Size of iterator: {getsizeof(iterator)/10**3:.2f} KB')
    for try_pssw in iterator:
        if try_pssw == password:
            print(f'The Password is {try_pssw}')
            break

print(hack_password.__name__)
print(hack_password.__doc__)

# Python tem 3 funções embutidas criadas para decorar métodos:
#   property, classmethod, staticmethod

# Another decorator
def my_logger(original_func):

    import logging
    file = f'{original_func.__name__}.log'
    print(file)
    logging.basicConfig(filename=file,
    level= logging.INFO)

    def wrapper(*args, **kwargs):
        result = original_func(*args, **kwargs)
        log = (
            f'Ran with args {args}, and kwargs {kwargs} '
            f'and result: {result}'
        )
        logging.info(log)
        print(log)
        return result
    
    return wrapper

@my_logger
def hack_password_v1(iterator, see_size=False):
    '''
    Through an iterator we discover a numeric password
    '''
    if see_size:
        print(f'Size of iterator: {getsizeof(iterator)/10**3:.2f} KB')
    for try_pssw in iterator:
        if try_pssw == password:
            print(f'The Password is {try_pssw}')
            return try_pssw
    
            
        
hack_password_v1(lis_password, see_size=True)

# --------------------------------------
# Class
# --------------------------------------

# Here we'll see a vector basic class
# Let's understand better how special methods works
class Vector:

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        # Return the canonical string representation of the object.
        # Without __repr__, it would return: 
        # <Vector object at 0x10e100070>.
        return f'Vector({self.x},{self.y})'

    def __abs__(self):
        # This is the absolute value of your class
        # in my case, it returns the euclidian distance
        return (self.x**2 + self.y**2)**(0.5)

    def __bool__(self):
        return bool(self.x or self.y)

    def __add__(self, other):
        # This method allows you sum 2 vectors class 
        # as a float/integers
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __add2__(self, other):
        # This method allows you sum 2 vectors class 
        # as a string and then convert to a int
        x = int(str(self.x) + str(other.x))
        y = int(str(self.y) + str(other.y))
        return Vector(x,y)

    def __mul__(self, scalar):
        # This method allows you product 2 vectors class
        return Vector(self.x * scalar, self.y * scalar)

    def __len__(self):
        return 2


# Test
v1 = Vector(2, 4)
v2 = Vector(1, 3)
print(repr(v1))
print(abs(v1))
print(bool(v1))
print(bool(Vector()))
print(len(v1))
print(v1.__len__())
print(v1.__add__(v2))

print(v1*3)

# Decorators as input




# --------------------------------------
# Closures
# --------------------------------------

b = 6
def f1(a):
    print(a)
    print(b)

# Se b não foi definida anterioramente, isso causará um erro
# Isto é, b precisa ser uma variável global

def f2(a):
    print(a)
    print(b)
    b = 9

# Mesmo b sendo uma variável global isso causará um erro.
# Pois estamos atribuindo b dentro de uma função
# a função f2 primeiro entende que b é um variável local
# e. portando, gera um erro.

def f3(a):
    global b
    print(a)
    print(b)
    b = 9

# Neste caso, como fixamos b como global mesmo dentro de f3
# Isso signinifca que printaremos b como 6, mas ao final da
# função f3 b terá valor de 9 globalmente
# A princípio isso é mecanismo de defesa do Python,
# Não queremos que as variáveis Globais sejam alteradas dentro
# de uma função

# Closures

# Queremos calcular a média que vai se atualizando diariamente

# 1. Utilizando uma Classe
class Averager():

    def __init__(self):
        self.series = []

    def __call__(self, new_value):
        self.series.append(new_value)
        total = sum(self.series)
        return total / len(self.series)
    
# Testando
avg = Averager()
avg(10)
avg(11)
avg(12)


# 2. Utilizando funções
def make_averager():
    '''
    Quando chamada make_averager devolve um objeto função averager.
    Sempre que um averager é chamado, ele concatena o argumento passado
    á série e calcula a média atual.
    '''
    series = []

    def averager(new_value):
        series.append(new_value)
        total = sum(series)
        return  total / len(series)
    
    return averager

# Testendo
avg = make_averager()
avg(10)
avg(11)
avg(12)

'''
O local em que avg da classe Averager mantém o histórico é óbvio
No atributo de instância self.series

Mas em que lugar a função avg no segundo exmplo encontra series?

Observe que series é uma variável local de make_avgerager
porque a inicilização está no corpo dessa função. Porém,
quando avg(10) é chamado, make_averager já retornou e seu escopo
local não existe mais

Em averager, series é uma variável livre. Esse é um termo técnico
que indica que uma variável não tem associação no escopo local
'''

# onde está series?
avg.__code__.co_freevars
avg.__closure__[0].cell_contents

'''
Uma Closure é uma função que preserva as associações com as 
variáveis livres existentes quando a função é definida, de modo
que elas possam ser utilizadas posteriormente quando a função
for chamada e o escopo de definição não estiver mais disponível

Note que a única situação em que uma função pode precisar lidar
com variáveis externas que não sejam globais é quando ela está
aninhada em uma outra função
'''

# Declaração non local
# Fazendo uma implementação mais eficiente que anterior

def make_averager():
    count = 0
    total = 0

    def averager(new_value):
        nonlocal count, total
        count += 1
        total += new_value
        return total / count

    return averager

avg = make_averager()
avg(10)
avg(11)
avg(12)

# --------------------------------------
# Partial
# --------------------------------------

from functools import partial

def dataprep_minmax():
    print('Fiz min-max')

def dataprep_dummy():
    print('Fiz dummy')

def dataprep_split(anomes=None):
    print(f'String in {anomes}')

def dataprep(feature_scaling, encoding, date_split):
    feature_scaling()
    encoding()
    date_split()
    return None

def set_arguments_split(*args, **kwargs):
    def dataprep_split(anomes=kwargs):
        print(f'String in {anomes}')
    return dataprep_split


dataprep_minmax_dummy = partial(dataprep, dataprep_minmax, dataprep_dummy)
dataprep_minmax_dummy(set_arguments_split(anomes=201701))