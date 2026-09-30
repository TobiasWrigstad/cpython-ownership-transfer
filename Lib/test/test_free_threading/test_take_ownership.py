import random
import sys
import threading
import time
import unittest
import weakref

from test import support
from test.support import import_helper, threading_helper

_testinternalcapi = import_helper.import_module("_testinternalcapi")
take_ownership = _testinternalcapi.take_ownership
refcount_state = _testinternalcapi.refcount_state
spin_without_safe_point = _testinternalcapi.spin_without_safe_point


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


def owned(objs):
    return [refcount_state(o)["owned_by_current_thread"] for o in objs]


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
class TestTakeOwnership(unittest.TestCase):

    def setUp(self):
        # The GC temporarily reuses ob_tid and restores it from the allocating
        # thread, which undoes transfers of GC-tracked objects. Keep automatic
        # collections out of the way of checks on ob_tid.
        cm = support.disable_gc()
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)

    def check_transfer(self, idle):
        with Owner(idle) as owner:
            objs = owner.objs
            self.assertEqual(owned(objs), [False] * len(objs))
            before = refcounts(objs)
            status = take_ownership(objs)
            self.assertEqual(status, ["transferred"] * len(objs))
            self.assertEqual(owned(objs), [True] * len(objs))
            self.assertEqual(refcounts(objs), before)

    def test_basic_owner_waiting(self):
        self.check_transfer("wait")

    def test_basic_owner_running(self):
        # The owner hands the objects over at a safe point.
        self.check_transfer("busy")

    def test_basic_owner_blocked(self):
        # The owner is detached: the objects are taken on its behalf.
        self.check_transfer("block")

    def test_refcount_traffic_afterwards(self):
        def run():
            with Owner("wait") as owner:
                objs = owner.objs
                self.assertEqual(take_ownership(objs),
                                 ["transferred"] * len(objs))
                wrs = [weakref.ref(o) for o in objs if isinstance(o, Obj)]
                barrier = threading.Barrier(2)

                def churn():
                    barrier.wait()
                    for _ in range(2000):
                        refs = [o for o in objs for _ in range(3)]
                        del refs

                # The old owner churns in a new thread; we (the new owner)
                # churn at the same time.
                t = threading.Thread(target=churn)
                t.start()
                churn()
                t.join()
                self.assertEqual(owned(objs), [True] * len(objs))
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
        nthreads = 4
        stop = threading.Event()
        with Owner("wait") as owner:
            items = owner.objs
            shared_list = list(items)
            shared_dict = {i: o for i, o in enumerate(items)}
            containers = [shared_list, shared_dict]
            errors = []

            def hammer():
                try:
                    while not stop.is_set():
                        shared_list.append(random.choice(items))
                        try:
                            shared_list.pop()
                        except IndexError:
                            pass
                        for i in range(len(items)):
                            shared_dict.get(i)
                            try:
                                shared_list[i]
                            except IndexError:
                                pass
                        shared_dict[len(items)] = random.choice(items)
                        shared_dict.pop(len(items), None)
                except BaseException as e:
                    errors.append(e)

            def taker():
                # Pull the objects and the containers around between threads.
                try:
                    while not stop.is_set():
                        take_ownership(items + containers)
                except BaseException as e:
                    errors.append(e)

            workers = [threading.Thread(target=hammer)
                       for _ in range(nthreads)]
            workers += [threading.Thread(target=taker) for _ in range(2)]
            with threading_helper.start_threads(workers):
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    status = take_ownership(items + containers)
                    self.assertLessEqual(
                        set(status),
                        {"transferred", "already-owned", "changed",
                         "merged"})
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
            results[me] = take_ownership(other, timeout=support.SHORT_TIMEOUT)
            barrier.wait()  # keep running until both are done

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(2)]
        with threading_helper.start_threads(threads):
            pass
        for status, other in zip(results, reversed(objs)):
            self.assertEqual(status, ["transferred"] * len(other))

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
            before = refcounts(objs)
            start = time.monotonic()
            status = take_ownership(objs, timeout=0.1)
            self.assertLess(time.monotonic() - start, 0.9)
            # The owner may reach a safe point before it starts spinning.
            if status[0] == "timeout":
                self.assertEqual(status, ["timeout"] * len(objs))
                self.assertEqual(owned(objs), [False] * len(objs))
            else:
                self.assertEqual(status, ["transferred"] * len(objs))
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
            status = take_ownership(objs, timeout=support.SHORT_TIMEOUT)
            self.assertEqual(status, ["transferred"] * len(objs))
            self.assertEqual(owned(objs), [True] * len(objs))
        finally:
            t.join()

    def test_taken_away_while_waiting(self):
        # While we wait for a slow owner, another thread takes an object we
        # own. The result reflects the state on return: that object is
        # reported as "changed", and everything reported as transferred or
        # already owned is ours.
        mine = Obj()
        ready = threading.Event()
        waiting = threading.Event()
        slow = []
        other_status = []

        def slow_owner():
            slow.append(Obj())
            ready.set()
            spin_without_safe_point(0.5)

        def other():
            waiting.wait()
            time.sleep(0.1)  # let the main thread start waiting
            other_status.extend(take_ownership([mine]))
            other_status.append(refcount_state(mine)["owned_by_current_thread"])

        threads = [threading.Thread(target=slow_owner),
                   threading.Thread(target=other)]
        with threading_helper.start_threads(threads):
            ready.wait()
            waiting.set()
            status = take_ownership([slow[0], mine],
                                    timeout=support.SHORT_TIMEOUT)
            self.assertEqual(status, ["transferred", "changed"])
            self.assertEqual(owned(slow), [True])
            self.assertFalse(refcount_state(mine)["owned_by_current_thread"])
        self.assertEqual(other_status, ["transferred", True])

    def test_owner_exited(self):
        # Objects of an exited thread are abandoned; we claim them. (If a live
        # thread has reused the id, it counts as the owner and hands them over
        # the usual way.)
        objs = []
        t = threading.Thread(target=lambda: objs.extend(make_objects()))
        t.start()
        t.join()
        before = refcounts(objs)
        status = take_ownership(objs)
        self.assertEqual(status, ["transferred"] * len(objs))
        self.assertEqual(owned(objs), [True] * len(objs))
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
            status = take_ownership(objs, timeout=support.SHORT_TIMEOUT)
            t.join()
            # Answered by the owner, by the owner on its way out, or by us
            # after it exited: always transferred.
            self.assertEqual(status, ["transferred"] * len(objs))
            self.assertEqual(owned(objs), [True] * len(objs))

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
            # The owner merges a queued object instead of transferring it.
            # (If the worker reached a safe point before spinning, it merged
            # its queue first: also "merged".)
            status = take_ownership([obj], timeout=support.SHORT_TIMEOUT)
            self.assertEqual(status, ["merged"])
            self.assertEqual(refcount_state(obj)["ob_tid"], 0)
        finally:
            t.join()
        self.assertIs(wr(), obj)
        del obj
        self.assertIsNone(wr())

    def test_special_cases(self):
        mine = Obj()
        # Merged: the owner drops its last local reference while another
        # thread holds a (shared) reference.
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
        merged = box[0]
        taken.set()
        t.join()
        self.assertEqual(refcount_state(merged)["ob_tid"], 0)

        status = take_ownership([None, 1, "abc", mine, merged, mine])
        self.assertEqual(status, ["immortal", "immortal", "immortal",
                                  "already-owned", "merged", "already-owned"])
        self.assertEqual(take_ownership([]), [])

    def test_summary(self):
        # The flag is true iff no object timed out or changed hands, i.e.
        # no other thread can do plain refcounting on any of the objects.
        self.assertEqual(take_ownership([], summary=True), (True, []))
        mine = Obj()
        self.assertEqual(take_ownership([None, mine], summary=True),
                         (True, ["immortal", "already-owned"]))
        with Owner("block") as owner:
            ok, status = take_ownership(owner.objs, summary=True)
            self.assertTrue(ok)
            self.assertEqual(status, ["transferred"] * len(owner.objs))

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
            ok, status = take_ownership(objs + [mine], timeout=0.1,
                                        summary=True)
            self.assertEqual((ok, status),
                             (False, ["timeout", "already-owned"]))
        finally:
            t.join()

    def test_duplicates(self):
        with Owner("wait") as owner:
            obj = owner.objs[0]
            before = sys.getrefcount(obj)
            status = take_ownership([obj, obj, obj])
            self.assertEqual(status, ["transferred", "already-owned",
                                      "already-owned"])
            self.assertEqual(sys.getrefcount(obj), before)
            # Dropping the references we took before the transfer (the list
            # and the call's own) did not make us give up ownership.
            self.assertTrue(refcount_state(obj)["owned_by_current_thread"])

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
            take_ownership((1, 2))
        with self.assertRaises(ValueError):
            take_ownership([], timeout=-1)


if __name__ == "__main__":
    unittest.main()
