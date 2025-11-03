import os
from redis import Redis
from rq import Worker, Queue, Connection

listen = ['default']
redis_url = os.environ.get('REDIS_URL', 'redis://redis:6379/0')

if __name__ == '__main__':
    redis_conn = Redis.from_url(redis_url)
    with Connection(redis_conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()
