from functools import wraps
import logging
import time
import pandas as pd
import numpy as np

# Set log name
logger = logging.getLogger(__name__)
# Set log Level
logger.setLevel(logging.INFO)
# Set log formatter
# YYYY-MM-DD HH:MM:SS,sss: __name__: INFO: {sometext}
formatter = logging.Formatter('"%(asctime)s:%(name)s:%(levelname)s:%(message)s"')
# Create filename and its attributes
file_handler = logging.FileHandler(
    "github_projects/random_things/log/logging_deco.log"
)  # name
file_handler.setFormatter(formatter)  # formatter
logger.addHandler(file_handler)

# If we still see the result in terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

df = pd.read_csv("dataset_random/titanic/train.csv")

# aux function
def df_shape(df: pd.DataFrame) -> str:
    row, columns = df.shape
    return f"[DataFrame] rows: {row} | columns: {columns}"


def np_shape(arr: np.ndarray) -> str:
    row = arr.shape
    return f"[np.Array] rows: {row}"


# argument function
def generic_convert(convert_fn: callable, convert_type: type) -> callable:
    """
    With naive convert function (callable) and target type,
    we can create a generic function (do conversion)

    Workflow
    if kwargs (dict) -- convert_fn(kwargs) --> kwargs (dict)
    if args (tuple) -- convert_fn(args) --> tuples (args)

    Example:
    generic_convert(np_shape, pd.Series)({'lala': df.Survived} ,'lili')
    generic_convert(np_shape, pd.Series)(tuple(('lala', 'lulu')))

    Insights:
    tuple(t for t in test_tuple) 
    if i don't put tuple it'll create a generator
    """

    def do_conversion(arguments):

        if isinstance(arguments, tuple):
            t = tuple(
                convert_fn(arg) if isinstance(arg, convert_type) else arg
                for arg in arguments
            )
            return t
        elif isinstance(arguments, dict):
            d = {
                kw: convert_fn(arg) if isinstance(arg, convert_type) else arg
                for kw, arg in arguments.items()
            }
            return d

    return do_conversion


def loggit(
    log_fn: callable,
    log_result: bool = False,
    log_time: bool = False,
    max_char: int = 300,
    encoder_str: list = [],
):
    """
    It's is a decorator with arguments. So this function defines a decorator
    Decorator's arguments:

    log_fn: logging object with its logging severity level method (Example: logger.info)
    log_result: If True, we log the result
    log_time: if True, we log time
    n_char = How many characters we should consider when logging
    encoder_str: list of callable like do_conversion. It's importante that callable
    is able to receive a *args|**kwargs and return this same *args|**kwargs
    """

    def decorator(original_function: callable):
        @wraps(original_function)
        def wrapper(*args, **kwargs):
            func_name = original_function.__name__
            # We create string arguments
            args_str = args
            kwargs_str = kwargs
            # If there is any encoder_str so apply it
            for enc in encoder_str:
                args_str = enc(args_str)
                kwargs_str = enc(kwargs_str)
            # final string-prep and apply max_char
            _args = ", ".join(
                repr(a) if len(repr(a)) < max_char else repr(a)[:max_char] + "..."
                for a in args_str
            )
            _kwargs = {
                key: repr(value)
                if len(repr(value)) < max_char
                else repr(value)[:max_char] + "..."
                for key, value in kwargs_str.items()
            }
            # log *args and **kwargs
            log_fn(f"[{func_name}] args: {_args}")
            log_fn(f"[{func_name}] kwargs: {_kwargs}")
            ti = time.perf_counter()
            func_result = original_function(*args, **kwargs)
            tf = time.perf_counter()
            if log_time:
                log_fn(f"[{func_name}] time: {tf-ti} seconds")
            if log_result:
                log_fn(f"[{func_name}] result: {func_result}")
            return func_result

        return wrapper

    return decorator


# TEST LOGGIT DECORATOR
@loggit(
    logger.info,
    log_result=False,
    log_time=False,
    encoder_str=[
        generic_convert(df_shape, pd.DataFrame),
        generic_convert(np_shape, pd.DataFrame),
        generic_convert(np_shape, pd.Series),
    ],
    max_char = 100
)
def cl_test(X_train, target, classifier):
    print(type(X_train))
    pass


# Test Generators
cl_test("aa", "bb", classifier=df)
cl_test(X_train=df, target=df.Survived, classifier=33)
cl_test(df, df.Survived, "cl")
