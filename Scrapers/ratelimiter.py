import random
import time

class RateLimiter:
    def __init__(self, max_calls=5, period=60):
        self.max_calls = max_calls
        self.period = period
        self.timestamps = []
    
    def wait(self):
        now = time.time()
        self.timestamps = [t for t in self.timestamps if t > now - self.period]
        
        if len(self.timestamps) >= self.max_calls:
            sleep_time = self.period - (now - self.timestamps[0])
            time.sleep(sleep_time + random.uniform(0.5, 1.5))
        
        self.timestamps.append(time.time())

# Uso:
