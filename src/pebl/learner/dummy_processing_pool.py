import multiprocessing

class Pool(object):

    def __init__(self, processes=None):
        if processes is None:
            processes = multiprocessing.cpu_count()
        self._pool_capacity = processes
        self._workers = []
        self._pool_sema = multiprocessing.BoundedSemaphore(self._pool_capacity)
        self._results_queue = multiprocessing.Queue()

    def join(self):
        for p in self._workers:
            p.join()
        del self._workers[:]

    def _wrap(self, func):
        def f(*args, **kwargs):
            #print 'go'
            try:
                self._results_queue.put(func(*args, **kwargs))
                #print 'done'
            except Exception, e:
                raise e
            finally:
                self._pool_sema.release()
        return f

    def apply(self, func, args=(), kwds={}):
        #print 'ready'
        self._pool_sema.acquire()
        p = multiprocessing.Process(target=self._wrap(func), args=args, kwargs=kwds)
        p.start()
        p.join()
        return self._results_queue.get()

    def map(self, func, iterable):
        for i in iterable:
            #print 'ready'
            self._pool_sema.acquire()
            p = multiprocessing.Process(target=self._wrap(func), args=(i,))
            self._workers.append(p)
            p.start()
        self.join()
        results = []
        while True:
            try:
                results.append(self._results_queue.get(False))
            except multiprocessing.queues.Empty:
                break
        return results

if __name__ == '__main__':
    import time, random

    def func(x):
        time.sleep(2)
        if random.random() < 0.3:
            raise Exception, 'i did this intentionally'
        return x**2

    pool = Pool(processes=10)
    #print pool.apply(func, args=(4,))
    print pool.map(func, range(10))
