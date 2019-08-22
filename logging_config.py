"""
Logging Levels

-- DEBUG: Detailed information, typically of interest only when diagnosing problems
-- INFO: Confirmation that things are working as expected
-- WARNING: An indication that something unexpected happened, or indicative of some problem in the near future.
-- Error: Due to a more serius problem, the software has not been able to perform some function
-- Critical: A serius error, indicating that the problem itself may be unable to continue running

Default Value: Warning (included)

Let's go for some examples

------------------
For more examples:
https://docs.python.org/3/library/logging.html#logrecord-attributes
"""

# Setting our log class
import logging
# Set log name
logger =  logging.getLogger(__name__)
# Set log Level
logger.setLevel(logging.INFO)
# Set log formatter
# YYYY-MM-DD HH:MM:SS,sss: __name__: INFO: {sometext} 
formatter = logging.Formatter('"%(asctime)s:%(name)s:%(levelname)s:%(message)s"')
# Create filename and its attributes
file_handler = logging.FileHandler('log/logging_test.log') # name
file_handler.setFormatter(formatter) # formatter
logger.addHandler(file_handler)

# If we still see the result in terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

'''
With above config, now we can delete this basic config

logging.basicConfig(
    filename="log/logging_test.log",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

and we should replace
logging.level(msg) >> logger.level(msg)

example:
logging.info(f'Python file:{__name__}') >>
logger.info(f'Python file:{__name__}')

Create our own logging class is a good practice, because when we are
import some module with default logging its default logging overrides
the __main__ default logging
'''

logger.info(f'Inicializate thie Python file as : {__name__}')

def add(x, y):
    """Apply add function"""
    return x + y


def subtract(x, y):
    """Apply subtract fuction"""
    return x - y


def multiply(x, y):
    """Apply multiply function"""
    return x * y


def divide(x, y):
    """Apply divide function"""
    try:
        result = x / y
    except ZeroDivisionError:
        logger.exception('Tried to divide by zero')
    else:
        return result


# Testing math function
num_1 = 50
num_2 = 0


add_result = add(num_1, num_2)
logger.debug(f"add: {num_1} operation {num_2} = {add_result}")

subtract_result = subtract(num_1, num_2)
logger.debug(f"subtract: {num_1} operation {num_2} = {subtract_result}")

multiply_result = multiply(num_1, num_2)
logger.debug(f"multiply: {num_1} operation {num_2} = {multiply_result}")

divide_result = divide(num_1, num_2)
logger.debug(f"divide: {num_1} operation {num_2} = {divide_result}")

