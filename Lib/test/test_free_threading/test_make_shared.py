import gc
import random
import sys
import threading
import time
import unittest
import weakref

from test import support
from test.support import import_helper, threading_helper

_testinternalcapi = import_helper.import_module("_testinternalcapi")
make_shared = _testinternalcapi.make_shared
refcount_state = _testinternalcapi.refcount_state
spin_without_safe_point = _testinternalcapi.spin_without_safe_point

_REF_MERGED = 3  # shared_flags of a merged refcount


class Obj:
    pass


def make_objects():
    # Mortal objects of various kinds, created by the calling thread. Values
    # are computed at run time so that none of them is an immortal constant.
    n = random.randrange(10**6, 10**7)
    return [Obj(), object(), [n], {n: n}, {n}, str(n) * 3, float(n),
            n * 10**30, bytes(str(n), "ascii")]


def refcounts(objs):
    return [sys.getrefcount(o) for o in objs]


def is_merged(obj):
    state = refcount_state(obj)
    return (state["ob_tid"] == 0 and state["local"] == 0
            and state["shared_flags"] == _REF_MERGED)


def merged(objs):
    return [is_merged(o) for o in objs]


class Owner:
    """A thread that creates objects and then runs `idle` until stopped."""

    def __init__(self, idle="wait"):
        self.objs = []
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.idle = idle
        self.thread = threading.Thread(target=self._run)

    def _run(self):
        self.objs.extend(make_objects())
        self.ready.set()
        if self.idle == "wait":
            # Runs Python code, but is mostly detached in the wait.
            while not self.stop.wait(0.01):
                pass
        elif self.idle == "busy":
            # Stays attached and regularly reaches safe points.
            while not self.stop.is_set():
                pass
        elif self.idle == "block":
            # Detached until stopped.
            self.stop.wait()

    def __enter__(self):
        self.thread.start()
        self.ready.wait()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


@threading_helper.requires_working_threading()
class TestMakeShared(unittest.TestCase):

    def check_make_shared(self, idle):
        with Owner(idle) as owner:
            objs = owner.objs
            self.assertFalse(any(merged(objs)))
            before = refcounts(objs)
            self.assertEqual(make_shared(objs, summary=True),
                             (True, ["shared"] * len(objs)))
            self.assertEqual(merged(objs), [True] * len(objs))
            self.assertEqual(refcounts(objs), before)

    def test_owner_waiting(self):
        self.check_make_shared("wait")

    def test_owner_running(self):
        # The owner merges at a safe point.
        self.check_make_shared("busy")

    def test_owner_blocked(self):
        # The owner is detached: the objects are merged on its behalf.
        self.check_make_shared("block")

    def test_own_objects(self):
        objs = make_objects()
        before = refcounts(objs)
        self.assertEqual(make_shared(objs), ["shared"] * len(objs))
        self.assertEqual(merged(objs), [True] * len(objs))
        self.assertEqual(refcounts(objs), before)
        wr = weakref.ref(objs[0])
        del objs
        self.assertIsNone(wr())

    def test_many_owners_at_once(self):
        # Busy, blocked and exited owners, our own objects and immortals, all
        # in one call.
        exited = []
        t = threading.Thread(target=lambda: exited.extend(make_objects()))
        t.start()
        t.join()
        mine = make_objects()
        with Owner("busy") as busy, Owner("block") as blocked:
            objs = busy.objs + blocked.objs + exited + mine + [None, 1]
            before = refcounts(objs)
            ok, status = make_shared(objs, summary=True)
            self.assertTrue(ok)
            self.assertEqual(status, ["shared"] * (len(objs) - 2)
                                     + ["immortal"] * 2)
            self.assertEqual(merged(objs[:-2]), [True] * (len(objs) - 2))
            self.assertEqual(refcounts(objs), before)

    def test_permanent(self):
        # A merged object never gets an owner again: not after running
        # Python code (safe points), a collection, or use by other threads.
        with Owner("busy") as owner:
            objs = owner.objs
            self.assertEqual(make_shared(objs), ["shared"] * len(objs))

            def use():
                for _ in range(1000):
                    refs = [o for o in objs for _ in range(3)]
                    del refs

            threads = [threading.Thread(target=use) for _ in range(4)]
            with threading_helper.start_threads(threads):
                use()
            gc.collect()
            self.assertEqual(merged(objs), [True] * len(objs))

    def test_refcount_traffic_afterwards(self):
        def run():
            with Owner("wait") as owner:
                objs = owner.objs
                self.assertEqual(make_shared(objs), ["shared"] * len(objs))
                wrs = [weakref.ref(o) for o in objs if isinstance(o, Obj)]
                barrier = threading.Barrier(2)

                def churn():
                    barrier.wait()
                    for _ in range(2000):
                        refs = [o for o in objs for _ in range(3)]
                        del refs

                # The old owner churns in a new thread; we churn at the same
                # time.
                t = threading.Thread(target=churn)
                t.start()
                churn()
                t.join()
                self.assertEqual(merged(objs), [True] * len(objs))
                owner.objs = None
                del objs
            return all(wr() is None for wr in wrs)

        # Warm up twice: the first runs populate caches.
        self.assertTrue(run())
        self.assertTrue(run())
        total = getattr(sys, "gettotalrefcount", None)
        if total is None:
            return
        deltas = []
        before = total()  # so that each round replaces a (mortal) int
        for _ in range(3):
            before = total()
            self.assertTrue(run())
            deltas.append(total() - before)
        self.assertEqual(deltas, [0, 0, 0])

    def test_concurrent_traffic(self):
        # Producers keep creating objects (owned by them) and putting them in
        # shared containers that other threads mutate and read, while the
        # main thread repeatedly makes the current contents shared.
        stop = threading.Event()
        shared_list = []
        shared_dict = {}
        lock = threading.Lock()
        errors = []

        def producer():
            try:
                while not stop.is_set():
                    for obj in make_objects():
                        shared_list.append(obj)
                        shared_dict[random.randrange(64)] = obj
                    with lock:
                        del shared_list[:-64]
            except BaseException as e:
                errors.append(e)

        def hammer():
            try:
                while not stop.is_set():
                    try:
                        shared_list.append(shared_list[-1])
                        shared_list.pop()
                    except IndexError:
                        pass
                    for i in range(64):
                        shared_dict.get(i)
                    shared_dict[random.randrange(64, 128)] = shared_list
                    shared_dict.pop(random.randrange(64, 128), None)
            except BaseException as e:
                errors.append(e)

        workers = [threading.Thread(target=producer) for _ in range(2)]
        workers += [threading.Thread(target=hammer) for _ in range(4)]
        with threading_helper.start_threads(workers):
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                objs = list(shared_list) + list(shared_dict.values())
                ok, status = make_shared(objs, summary=True)
                self.assertTrue(ok, set(status))
                self.assertLessEqual(set(status), {"shared"})
            stop.set()
        self.assertEqual(errors, [])

    def test_mutual_requests(self):
        barrier = threading.Barrier(2)
        objs = [None, None]
        results = [None, None]

        def worker(me):
            objs[me] = make_objects()
            barrier.wait()
            other = objs[1 - me]
            barrier.wait()
            results[me] = make_shared(other, timeout=support.SHORT_TIMEOUT)
            barrier.wait()  # keep running until both are done

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(2)]
        with threading_helper.start_threads(threads):
            pass
        for status, other in zip(results, reversed(objs)):
            self.assertEqual(status, ["shared"] * len(other))
            self.assertEqual(merged(other), [True] * len(other))

    def test_owner_without_safe_point(self):
        # An attached owner that never reaches a safe point cannot answer.
        ready = threading.Event()
        objs = []

        def worker():
            objs.extend(make_objects())
            ready.set()
            spin_without_safe_point(1.0)

        t = threading.Thread(target=worker)
        t.start()
        try:
            ready.wait()
            time.sleep(0.05)  # let the worker start spinning
            before = refcounts(objs)
            start = time.monotonic()
            ok, status = make_shared(objs, timeout=0.1, summary=True)
            self.assertLess(time.monotonic() - start, 0.9)
            self.assertEqual((ok, status),
                             (False, ["timeout"] * len(objs)))
            self.assertEqual(merged(objs), [False] * len(objs))
            self.assertEqual(refcounts(objs), before)
        finally:
            t.join()

    def test_owner_wakes_before_deadline(self):
        ready = threading.Event()
        objs = []

        def worker():
            objs.extend(make_objects())
            ready.set()
            spin_without_safe_point(0.2)
            for _ in range(100):  # safe points
                pass

        t = threading.Thread(target=worker)
        t.start()
        try:
            ready.wait()
            status = make_shared(objs, timeout=support.SHORT_TIMEOUT)
            self.assertEqual(status, ["shared"] * len(objs))
            self.assertEqual(merged(objs), [True] * len(objs))
        finally:
            t.join()

    def test_owner_exited(self):
        # Objects of an exited thread are abandoned; we merge them. (If a
        # live thread has reused the id, it counts as the owner and merges
        # them itself.)
        objs = []
        t = threading.Thread(target=lambda: objs.extend(make_objects()))
        t.start()
        t.join()
        before = refcounts(objs)
        self.assertEqual(make_shared(objs), ["shared"] * len(objs))
        self.assertEqual(merged(objs), [True] * len(objs))
        self.assertEqual(refcounts(objs), before)
        wrs = [weakref.ref(o) for o in objs if isinstance(o, Obj)]
        del objs
        self.assertEqual([wr() for wr in wrs], [None] * len(wrs))

    def test_owner_exits_during_request(self):
        for _ in range(20):
            ready = threading.Event()
            objs = []

            def worker():
                objs.extend(make_objects())
                ready.set()
                spin_without_safe_point(0.01)

            t = threading.Thread(target=worker)
            t.start()
            ready.wait()
            status = make_shared(objs, timeout=support.SHORT_TIMEOUT)
            t.join()
            # Answered by the owner, by the owner on its way out, or by us
            # after it exited.
            self.assertEqual(status, ["shared"] * len(objs))
            self.assertEqual(merged(objs), [True] * len(objs))

    def test_queued(self):
        ready = threading.Event()
        box = []

        def worker():
            obj = Obj()
            box.extend([obj, obj])  # two references, both counted locally
            del obj
            ready.set()
            # Stay attached without safe points, so that the queued object is
            # neither merged by us nor on our behalf before the request.
            spin_without_safe_point(0.5)

        t = threading.Thread(target=worker)
        t.start()
        try:
            ready.wait()
            time.sleep(0.05)  # let the worker start spinning
            obj = box.pop()
            wr = weakref.ref(obj)
            # A non-owner drops a reference while the shared count is zero:
            # the object is pushed onto the owner's merge queue.
            del obj
            obj = box.pop()
            status = make_shared([obj], timeout=support.SHORT_TIMEOUT)
            self.assertEqual(status, ["shared"])
            self.assertTrue(is_merged(obj))
        finally:
            t.join()
        # The owner merged its queue afterwards; the object is still alive
        # and freed when the last reference goes.
        self.assertIs(wr(), obj)
        del obj
        self.assertIsNone(wr())

    def test_special_cases(self):
        mine = Obj()
        # Already merged: the owner drops its last local reference while
        # another thread holds a (shared) reference.
        box = []

        def worker():
            box.append(Obj())
            ready.set()
            taken.wait()
            box.clear()

        ready = threading.Event()
        taken = threading.Event()
        t = threading.Thread(target=worker)
        t.start()
        ready.wait()
        already = box[0]
        taken.set()
        t.join()
        self.assertTrue(is_merged(already))

        status = make_shared([None, 1, "abc", already, mine])
        self.assertEqual(status, ["immortal", "immortal", "immortal",
                                  "shared", "shared"])
        self.assertTrue(is_merged(mine))
        self.assertEqual(make_shared([]), [])

    def test_duplicates(self):
        with Owner("wait") as owner:
            obj = owner.objs[0]
            before = sys.getrefcount(obj)
            self.assertEqual(make_shared([obj, obj, obj]), ["shared"] * 3)
            self.assertEqual(sys.getrefcount(obj), before)
            self.assertTrue(is_merged(obj))

    def test_summary(self):
        # The flag is true iff every object is shared or immortal.
        self.assertEqual(make_shared([], summary=True), (True, []))
        self.assertEqual(make_shared([None, Obj()], summary=True),
                         (True, ["immortal", "shared"]))

        ready = threading.Event()
        objs = []

        def worker():
            objs.append(Obj())
            ready.set()
            spin_without_safe_point(1.0)

        t = threading.Thread(target=worker)
        t.start()
        try:
            ready.wait()
            time.sleep(0.05)  # let the worker start spinning
            mine = Obj()
            self.assertEqual(make_shared(objs + [mine], timeout=0.1,
                                         summary=True),
                             (False, ["timeout", "shared"]))
        finally:
            t.join()

    def test_refcount_state(self):
        obj = Obj()
        state = refcount_state(obj)
        self.assertEqual(set(state), {"owned_by_current_thread", "ob_tid",
                                      "local", "shared_count", "shared_flags",
                                      "is_immortal"})
        self.assertTrue(state["owned_by_current_thread"])
        self.assertFalse(state["is_immortal"])
        self.assertTrue(refcount_state(None)["is_immortal"])

    def test_bad_arguments(self):
        with self.assertRaises(TypeError):
            make_shared((1, 2))
        with self.assertRaises(ValueError):
            make_shared([], timeout=-1)


if __name__ == "__main__":
    unittest.main()
