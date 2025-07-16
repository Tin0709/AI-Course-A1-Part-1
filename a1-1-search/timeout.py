import signal

# class timeout:
#     """To use it in grading.py:

#         with timeout.timeout(seconds=self.timeOuts[q]):
#                 getattr(gradingModule, q)(self)

#     Taken from https://aalvarez.me/posts/making-functions-timeout-in-python/
#     """
#     def __init__(self, seconds):
#         self._seconds = seconds

#     def __enter__(self):
#         # Register and schedule the signal with the specified time
#         signal.signal(signal.SIGALRM, timeout._raise_timeout)
#         signal.alarm(self._seconds)
#         return self

#     def __exit__(self, exc_type, exc_value, traceback):
#         # Unregister the signal so it won't be triggered if there is
#         # no timeout
#         signal.signal(signal.SIGALRM, signal.SIG_IGN)

#     @staticmethod
#     def _raise_timeout(signum, frame):
#         raise TimeoutError





# code to handle timeouts
#
# FIXME
# NOTE: TimeoutFuncton is NOT reentrant.  Later timeouts will silently
# disable earlier timeouts.  Could be solved by maintaining a global list
# of active time outs.  Currently, questions which have test cases calling
# this have all student code so wrapped.
#
import signal
import time
class TimeoutFunctionException(Exception):
    """Exception to raise on a timeout"""

    def __init__(self, secs):
        self.secs = secs
    pass


class TimeoutFunction:
    # https://aalvarez.me/posts/making-functions-timeout-in-python/
    def __init__(self, function, timeout):
        self.seconds = timeout
        self.function = function
        self.timeout = False    # has a timeout ocurred?
        self.handled = False    # have I handled myself the timeout and the re-raise it?

    def raise_timeout(self, signum, frame):
        self.timeout = True
        raise TimeoutFunctionException(self.seconds)

    def __call__(self, *args, **keyArgs):
        # print(f"Running under timeout: {self.timeout}")
        # If we have SIGALRM signal, use it to cause an exception if and
        # when this function runs too long.  Otherwise check the time taken
        # after the method has returned, and throw an exception then.
        if hasattr(signal, 'SIGALRM'):
            # https://docs.python.org/3/library/signal.html#signal.signal
            # Register the handler self.raise_timeout for SIGALRM and save the old handler to recover later
            print("has SIGALRM")
            old_handler = signal.signal(signal.SIGALRM, self.raise_timeout)
            signal.alarm(self.seconds)
            try:
                result = self.function(*args, **keyArgs)
            except TimeoutFunctionException as e:
                self.handled = True
                raise e
            finally:
                signal.signal(signal.SIGALRM, old_handler)  # bring back original handler
                signal.alarm(0)

                #  just in case submission code has caught the exception! 
                # Students should not caught any timeout or any free exception!
                if self.timeout and not self.handled:
                    print("Timeout but not caught! Potential interference with autograder, check manually! :-(")
                    raise TimeoutFunctionException(self.seconds)
        else:
            # if there is no SIGALRM, e.g., under Windows; wait until it is finished and time-it
            print("has not SIGALRM")  
            startTime = time.time()
            result = self.function(*args, **keyArgs)
            timeElapsed = time.time() - startTime
            if timeElapsed >= self.seconds:
                self.raise_timeout(None, None)
        return result

