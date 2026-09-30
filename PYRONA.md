# Ownership transfer for free-threaded CPython (`ownership-transfer` branch)

This branch adds a C function that lets a thread become the *owner* of a set
of objects currently owned by other threads, in the free-threaded build
(`--disable-gil`). It is a building block for region-based ownership: after
the call returns successfully, no other thread can do non-atomic reference
counting on the objects until the calling thread next reaches a safe point or
detaches.

The branch is based on the CPython 3.15 branch (`aa69ad1f172`). Everything is
under `#ifdef Py_GIL_DISABLED`; the default (GIL) build is unchanged.


## Background: why ownership needs a handshake

In the free-threaded build every object has an owning thread, stored in
`ob_tid`. The owner updates `ob_ref_local` with plain (non-atomic) stores;
every other thread updates `ob_ref_shared` atomically. This is biased
reference counting (BRC, `Python/brc.c`). Because the owner never uses
atomics on `ob_ref_local`, nobody else may change `ob_tid` while the owner
might be in the middle of a refcount update. Ownership can therefore only
change hands when the owner is known not to be refcounting: when it is at a
safe point, or when it is detached and held in a "suspended" state.


## The function

```c
#include "internal/pycore_brc.h"

int _Py_brc_take_ownership(PyObject **objects, Py_ssize_t n,
                           PyTime_t timeout, int *results);
```

| Parameter | Meaning |
|---|---|
| `objects` | Array of `n` objects. The caller must hold a strong reference to every object for the whole call. Duplicates are allowed. |
| `n` | Number of objects; 0 is allowed. |
| `timeout` | Maximum time to wait for owners to respond, in nanoseconds (`PyTime_t`). |
| `results` | Array of `n` `int`s; receives one `_Py_ownership_status` per object, in order. |

| Return value | Meaning |
|---|---|
| `1` | The guarantee below holds for **every** object (no object is `TIMEOUT` or `CHANGED`). |
| `0` | It does not hold for at least one object; see `results`. |
| `-1` | Error, with an exception set. Only `MemoryError` can occur. |

The calling thread must have an attached thread state, as for any C API call.

### Statuses

The statuses describe the objects **when the call returns**, not when their
previous owner handled the request.

| Status | Meaning | Guarantee holds? |
|---|---|---|
| `_Py_OWNERSHIP_TRANSFERRED` | Ownership moved to the calling thread. | yes: owned by caller |
| `_Py_OWNERSHIP_ALREADY_OWNED` | The caller already owned it (or it was listed twice). | yes: owned by caller |
| `_Py_OWNERSHIP_MERGED` | The object has no owner (`ob_tid == 0`): every thread refcounts it atomically. It was not taken over. | yes: no owner |
| `_Py_OWNERSHIP_IMMORTAL` | Immortal; refcounting does not apply. | not applicable |
| `_Py_OWNERSHIP_TIMEOUT` | The owner did not respond before the deadline; the request was withdrawn. The object is still owned by that thread. | **no** |
| `_Py_OWNERSHIP_CHANGED` | The object is owned by yet another thread: it changed hands during the call (another thread won a race for it, or took it from the caller while the caller was waiting). | **no** |

`TIMEOUT` and `CHANGED` objects can simply be requested again.

### The guarantee, and how long it lasts

When the function returns 1, no thread other than the caller can do plain
(non-atomic) refcounting on any of the objects. This lasts until the calling
thread **next reaches a safe point or detaches**. After that, other threads
may request the objects and the caller will hand them over, and a garbage
collection may run (see "Limitations").

Code that relies on the guarantee must therefore do neither:

- **Safe points** are where the interpreter checks the eval breaker:
  function entry (`RESUME`), loop back-edges (`JUMP_BACKWARD`), the end of
  calls, and JIT side exits. In practice: don't run Python code.
- **Detaching** happens in `Py_BEGIN_ALLOW_THREADS` (I/O, `sleep`, ...), when
  a `PyMutex_Lock` or a critical section has to wait, and in any blocking C
  API call.

Note that the function itself detaches while it waits, so **any critical
sections the caller holds are suspended (released) during the call**, as with
any blocking call. Take the critical sections you need *after* the call, and
make sure acquiring them cannot block, or re-check ownership afterwards.

### Calling it from a C extension

The function is internal, unstable API: it lives in an internal header and may
change without notice. An extension outside CPython can still call it (it is
exported) by opting in to internal headers, which are installed under
`include/python3.15t/internal/`:

```c
#define Py_BUILD_CORE_MODULE 1      // before Python.h: allows internal headers
#include <Python.h>

#ifndef Py_GIL_DISABLED
#  error "ownership transfer requires the free-threaded build"
#endif
#include "internal/pycore_brc.h"    // _Py_brc_take_ownership()

// Take ownership of the n objects in `objs` (the caller holds references).
// Returns 1 if every object is now safe from foreign non-atomic refcounting,
// 0 if not, -1 on error.
static int
claim_objects(PyObject **objs, Py_ssize_t n)
{
    int *status = PyMem_Malloc((n ? n : 1) * sizeof(int));
    if (status == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    int ok = _Py_brc_take_ownership(objs, n, 100 * 1000 * 1000 /* 100 ms */,
                                    status);
    if (ok == 0) {
        for (Py_ssize_t i = 0; i < n; i++) {
            if (status[i] == _Py_OWNERSHIP_TIMEOUT ||
                status[i] == _Py_OWNERSHIP_CHANGED) {
                // objs[i] still has another owner: retry, or give up.
            }
        }
    }
    else if (ok == 1) {
        // From here until the next safe point or detach, no other thread
        // does non-atomic refcounting on objs[0..n).
    }
    PyMem_Free(status);
    return ok;
}
```

Code inside CPython includes `"pycore_brc.h"` directly.

### What happens to the objects

- **Refcount fields.** For a transferred object, the whole reference count is
  moved into `ob_ref_local` and the shared count is set to zero (flags kept):
  the object looks as if the new owner had just allocated it. The total is
  unchanged. Without this, the new owner would give up ownership again as
  soon as it dropped references it had taken before the transfer.
- **Lists, dicts and sets** get the "shared" GC bit set before the transfer,
  so the new owner frees their backing storage through QSBR rather than
  immediately (other threads may still be reading it without a lock). This
  happens even if the transfer then fails.
- **Objects queued for merging** (another thread dropped a reference and put
  the object on its owner's BRC merge queue) are merged on the spot and
  reported `MERGED`.
- **Objects of exited threads** still carry the dead thread's id. They are
  claimed and reported `TRANSFERRED`. (Leaving them is not safe: a new thread
  can get the same id and would then refcount them non-atomically.)
- **Merged objects** are left merged; nothing gives them an owner before the
  caller's next safe point.


## How it works

### The mechanism in one pass

When a list of objects is passed to `_Py_brc_take_ownership`, we walk the
list and look at who owns each object. Immortal objects, objects we already
own, and objects with no owner (`ob_tid == 0`, merged) are left alone. Merged
objects already satisfy the guarantee: every thread updates their refcount
atomically, and nothing can give them an owner before our next safe point.
The remaining objects are owned by some other thread. We group them by owner,
so each owner gets one request for all of its objects, and before sending
anything we set the shared bit on lists, dicts and sets, so that the new owner
will never free their storage while another thread might still be reading it
without a lock.

For each owner we take that owner's bucket mutex and look at its state. If
the owner has exited, its objects are abandoned but still carry its id; we do
the handover ourselves, which is safe because a new thread that happens to
get the same id must register under this same mutex. If the owner is
detached, we freeze it by compare-and-swapping its thread state from
`DETACHED` to `SUSPENDED`, do the handover on its behalf, and release it;
while it is suspended it cannot reattach, so it cannot be refcounting. If the
owner is attached, we link the request into its list and set its eval-breaker
bit, and it does the handover itself at its next safe point.

The handover is the same code in all three cases, and runs under the owner's
bucket mutex. For each object: if it has become merged or immortal in the
meantime, we say so; if it is already ours (listed twice), we say so; if yet
another thread owns it, it is `CHANGED`; if it is queued for merging, it is
merged on the spot and reported `MERGED`; otherwise its whole refcount moves
into `ob_ref_local` and `ob_tid` becomes our id: `TRANSFERRED`.

While waiting, we sleep in 1 ms slices, detached, so that stop-the-world can
proceed. Between slices we answer requests addressed to us, so two threads
asking each other cannot deadlock, and we retry freezing owners that have
blocked since. At the deadline, whatever is still unanswered is withdrawn:
`TIMEOUT`.

Finally, still attached and with nothing left that can block, we re-check
every object reported `TRANSFERRED` or `ALREADY_OWNED`. Any that is no longer
ours, because another thread took it from us while we waited or it got
merged, is reported as `CHANGED`, `MERGED` or `IMMORTAL`. The result: objects
reported `TRANSFERRED`, `ALREADY_OWNED`, `MERGED` or `IMMORTAL` satisfy the
guarantee, and only `TIMEOUT` and `CHANGED` objects may still have another
owner. The function returns 1 exactly when neither of those two statuses
appears.

### Step by step

1. **Classify** each object: immortal, owned by the caller, merged
   (`ob_tid == 0`), or foreign. Foreign objects are grouped by owner, so each
   owner gets one request for all of its objects.
2. **Mark** lists, dicts and sets as shared.
3. **For each owner**, under that owner's BRC bucket mutex (the per-thread
   lock BRC already uses for its merge queues):
   - *owner has exited*: no thread with that id can exist or appear while the
     mutex is held (new threads register under it), so the caller does the
     handover itself;
   - *owner is detached* (blocked in I/O, a lock, `sleep`): the caller freezes
     it with a compare-and-swap on its thread state (`DETACHED` →
     `SUSPENDED`, `_PyThreadState_TrySuspendDetached`), does the handover on
     its behalf, and releases it. A suspended thread cannot reattach, so it
     cannot be refcounting;
   - *owner is attached*: the request is linked into the owner's request list
     and the owner's `_PY_EVAL_EXPLICIT_MERGE_BIT` is set. The owner does the
     handover itself at its next safe point, before merging its BRC queue.
4. **The handover** is the same code in all three cases, run under the
   owner's bucket mutex, per object: if it is merged or immortal by now,
   report that; if another thread owns it, report `CHANGED`; if it is queued,
   merge it and report `MERGED`; otherwise move its count into `ob_ref_local`,
   store the caller's id in `ob_tid` (release) and report `TRANSFERRED`.
5. **Wait** in 1 ms slices, detached (so stop-the-world and threads acting on
   the caller's behalf can proceed). Between slices: answer requests addressed
   to the caller (so two threads requesting each other's objects cannot
   deadlock), and retry freezing owners that have detached since.
6. **At the deadline**, withdraw unanswered requests under the owner's bucket
   mutex and report `TIMEOUT`.
7. **Re-check**, attached, with nothing left that can block: every object
   reported `TRANSFERRED` or `ALREADY_OWNED` that is no longer owned by the
   caller (taken while it waited, or merged) is reported `CHANGED`, `MERGED`
   or `IMMORTAL`. Compute the return value.

### `_PY_EVAL_EXPLICIT_MERGE_BIT`

This eval-breaker bit tells a thread that it has refcount bookkeeping to do
for other threads. When the thread sees it at a safe point,
`_Py_HandlePending` clears it and calls `_Py_brc_merge_refcounts`, which,
under the thread's bucket mutex, first answers pending ownership requests and
then merges the thread's BRC queue as before.

- **In stock CPython** the bit has a single meaning: "objects to merge". It is
  set in exactly one place, `_Py_brc_queue_object`, when a thread pushes an
  object onto an *attached* owner's merge queue. That happens when thread B
  drops a reference to an object owned by thread A while the shared count is
  zero, i.e. the reference was counted in A's local count (typically: A
  created the object and handed it to B without B increfing it). B cannot
  touch A's local count, so it queues the object and sets A's bit. This is
  common in programs that create objects in one thread and release them in
  another (producer/consumer queues, thread-pool results, futures); it never
  happens in single-threaded programs. If the owner is detached, no bit is
  set: the queuing thread merges on the owner's behalf (gh-157838).
- **On this branch** `post_request` also sets the bit when it leaves an
  ownership request for an attached owner, so the bit now means "objects to
  merge, or ownership requests pending". Reusing it avoids a new bit: all
  eight event bits are taken, and adding one would shift the instrumentation
  version stored in the upper bits of the same word.

Consequences of sharing the bit: if a request is withdrawn after the bit was
set, the owner later runs the handler with nothing to do (one lock
acquisition and an empty merge); and whenever the bit fires for ordinary BRC
reasons, the owner also looks at its request list, which is usually empty (a
single pointer check under a lock it takes anyway).

### Latency

An attached owner answers at its next safe point, which is usually very
soon; an owner running a long C call that never returns to the
interpreter (a big `sorted()`, a regex match) answers only after it returns,
which is what produces `TIMEOUT`. A detached owner is handled immediately.

### Locking

At most one bucket mutex is held at a time, and nothing that blocks,
allocates, takes a critical section or runs Python code happens while one is
held. The detailed protocol, memory-ordering argument and invariants are in the
design note in `Python/brc.c` (section "Ownership transfer").


## Limitations relevant to regions

- **The GC can undo a transfer.** During a collection the free-threaded GC
  reuses `ob_tid` of GC-tracked objects and afterwards restores it from the
  thread that allocated the memory (`gc_restore_tid` in
  `Python/gc_free_threading.c`), not from the previous owner. A collection
  needs a stop-the-world, which needs the caller at a safe point or detached,
  so it cannot happen within the guarantee window. This is being addressed in
  a separate patch.
- **Some references are not counted in the object itself.** Heap types, code
  objects and dicts with per-thread refcounting (module globals) are counted
  through per-thread counters (`tstate->refcounts`) that every thread
  increments non-atomically, regardless of the owner. Objects with deferred
  refcounting can be referenced from other threads' stacks without any
  refcount operation. Neither is visible in the object's refcount; a region
  algorithm that reasons from refcounts must treat these objects specially.
- **Merged objects are not taken over**; they are reported `MERGED`.
- **Owners stuck in C code** without safe points cause `TIMEOUT`.
- **Interpreter scope.** As with BRC itself, objects are assumed to be
  refcounted only by threads of their own interpreter, and a thread is
  assumed not to run with a second thread state of the same interpreter while
  one is detached.


## Changes to CPython on this branch

| File | Change |
|---|---|
| `Python/brc.c` | The ownership-transfer protocol (new section with design note), plus fixes below. |
| `Include/internal/pycore_brc.h` | `_Py_brc_take_ownership()`, `_Py_ownership_status`, and a per-thread list of pending requests in `struct _brc_thread_state`. |
| `Python/ceval_gil.c` | Comment only: `_PY_EVAL_EXPLICIT_MERGE_BIT` now also means "ownership requests pending". |
| `Modules/_testinternalcapi.c` | Python-level helpers for testing (below). |
| `Lib/test/test_free_threading/test_take_ownership.py` | Tests. |

Changes to existing BRC behaviour, needed for correctness once ownership can
move:

- **Queue race.** A thread pushing an object onto its owner's merge queue now
  re-reads `ob_tid` under the owner's bucket mutex and retries with the new
  owner if it changed. Before, it could push an object onto the queue of a
  thread that had just given it away.
- **Merge loops** assert that every queued object is still owned by the
  queue's thread, and forward it to its current owner otherwise.
- **Thread exit** answers pending requests in the same critical section that
  removes the thread from the BRC table; **fork** drops requests from threads
  that do not exist in the child.
- **No new eval-breaker bit**: requests reuse `_PY_EVAL_EXPLICIT_MERGE_BIT`,
  so the bit layout (and the instrumentation version stored next to it) is
  unchanged.

Nothing here is visible to Python code or to C extensions that do not opt in
to internal headers.


## Building and testing

```sh
./configure --disable-gil --with-pydebug && make -j
./python -m test test_free_threading.test_take_ownership -v
```

For a ThreadSanitizer build add `--with-thread-sanitizer` (with clang). On
kernels with high ASLR entropy TSan binaries may fail with "unexpected memory
mapping"; run them under `setarch $(uname -m) -R`.

Test helpers in `_testinternalcapi` (free-threaded build only; they raise
`NotImplementedError` in the default build):

- `take_ownership(objects: list, timeout=5.0, summary=False)`: calls the C
  function and returns the statuses as strings (`"transferred"`,
  `"already-owned"`, `"immortal"`, `"merged"`, `"timeout"`, `"changed"`), or
  `(flag, statuses)` with `summary=True`.
- `refcount_state(obj)`: dict with `owned_by_current_thread`, `ob_tid`,
  `local`, `shared_count`, `shared_flags`, `is_immortal`.
- `spin_without_safe_point(seconds)`: busy-waits while attached without
  reaching a safe point (to provoke `TIMEOUT`).

**Known flaky test.** `test_concurrent_traffic` fails in roughly 1 in 100–300
runs on debug builds with `Assertion 'Py_REFCNT((PyObject*)mp) > 0' failed`
in `_PyDict_NotifyEvent`. This is a pre-existing race in stock CPython 3.15,
not in this branch: the debug assertion reads the two refcount fields while
`_Py_ExplicitMergeRefcount` is between its two updates. It has been reproduced
on unmodified 3.15 with a pure-Python script and is to be reported upstream.
The test triggers it often because transfers leave the shared count at zero,
so objects are queued and merged frequently. Release builds are unaffected.


## Performance

Benchmarked with pyperformance on free-threaded PGO+LTO release builds, two
independent builds each of the base commit and of the branch, to separate
the effect of the change from build-to-build layout noise. Over 22 benchmarks
the branch lands between the two baseline builds (+0.4% and −1.1% by
geometric mean), and the remaining per-benchmark differences are within the
spread between builds of identical code (up to 13% for layout-sensitive
benchmarks such as `spectral_norm`). No measurable cost. This is expected:
single-threaded programs never set the merge bit, so none of the new code
runs; the only change they see is a 16-byte list head added to the thread
state.
