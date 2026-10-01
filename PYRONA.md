# Making objects shared in free-threaded CPython (`ownership-transfer` branch)

This branch adds a C function, `_Py_brc_make_shared()`, that takes a set of
objects owned by any threads and makes them *shared*: their reference counts
are merged, so that no thread owns them and every thread refcounts them
atomically. It is a building block for region-based ownership: once the call
returns successfully, **no thread will ever again do non-atomic reference
counting on those objects**.

The branch is based on the CPython 3.15 branch (`aa69ad1f172`). Everything is
under `#ifdef Py_GIL_DISABLED`; the default (GIL) build is unchanged. (An
earlier version of the branch, in commit `f1fbbad73eb`, transferred ownership
to the calling thread instead; this version replaces it.)


## Background: owners, and why merging needs a handshake

In the free-threaded build every object has an owning thread, stored in
`ob_tid`. The owner updates `ob_ref_local` with plain (non-atomic) stores;
every other thread updates `ob_ref_shared` atomically. This is biased
reference counting (BRC, `Python/brc.c`). An object can also have no owner:
then `ob_tid` is 0, the whole count is in `ob_ref_shared` with the flag
`_Py_REF_MERGED`, and every thread uses atomics. BRC calls this *merged*.
Stock CPython merges an object when its owner drops its last local reference
while other threads still hold references, and (lazily) objects whose owner
has exited.

Merging is a one-way street: a merged object never gets an owner again (only
a dead object can be resurrected, and the GC keeps merged objects merged).

Because the owner never uses atomics on `ob_ref_local`, nobody else may merge
an object while the owner might be in the middle of a refcount update. So
merging another thread's object is only safe when that owner is at a safe
point, is detached and held in a "suspended" state, or has exited. That is
what the handshake below arranges.


## The function

```c
#include "internal/pycore_brc.h"

int _Py_brc_make_shared(PyObject **objects, Py_ssize_t n,
                        PyTime_t timeout, int *results);
```

| Parameter | Meaning |
|---|---|
| `objects` | Array of `n` objects. The caller must hold a strong reference to every object for the whole call. Duplicates are allowed. |
| `n` | Number of objects; 0 is allowed. |
| `timeout` | Maximum time to wait for owners to respond, in nanoseconds (`PyTime_t`). |
| `results` | Array of `n` `int`s; receives one `_Py_shared_status` per object, in order. |

| Return value | Meaning |
|---|---|
| `1` | Every object is now merged (or immortal). |
| `0` | At least one object is `TIMEOUT` or `CHANGED`; see `results`. |
| `-1` | Error, with an exception set. Only `MemoryError` can occur. |

The calling thread must have an attached thread state, as for any C API call.

### Statuses

| Status | Meaning | Safe from non-atomic refcounting? |
|---|---|---|
| `_Py_SHARED_MERGED` | The object is merged: no owner, refcounted atomically by all threads (whether it was merged by this call or already before). | **yes, permanently** |
| `_Py_SHARED_IMMORTAL` | Immortal; refcounting does not apply. | not applicable |
| `_Py_SHARED_TIMEOUT` | The owner did not respond before the deadline; the request was withdrawn. The object is still owned by that thread. | **no** |
| `_Py_SHARED_CHANGED` | The object turned out to be owned by a different thread than the one asked. This can only come from thread-id reuse or from the GC resetting `ob_tid` (see "Limitations"). | **no** |

`TIMEOUT` and `CHANGED` objects can simply be passed to another call.

### The guarantee

When the function returns 1, every object is merged or immortal. A merged
object stays merged for the rest of its lifetime, so **no thread, including
the caller, will ever do non-atomic refcounting on it**. Unlike an ownership
transfer, this does not end at the caller's next safe point or detach: the
caller may run Python code, block, or let the GC run afterwards.

The price is that every refcount operation on these objects is atomic from
then on, by every thread, which is slower for heavily used objects.

Note that the function detaches while it waits, so **any critical sections
the caller holds are suspended (released) during the call**, as with any
blocking call.

### Calling it from a C extension

The function is internal, unstable API: it lives in an internal header and may
change without notice. An extension outside CPython can still call it (it is
exported) by opting in to internal headers, which are installed under
`include/python3.15t/internal/`:

```c
#define Py_BUILD_CORE_MODULE 1      // before Python.h: allows internal headers
#include <Python.h>

#ifndef Py_GIL_DISABLED
#  error "make_shared requires the free-threaded build"
#endif
#include "internal/pycore_brc.h"    // _Py_brc_make_shared()

// Make the n objects in `objs` shared (the caller holds references).
// Returns 1 if all of them are now merged or immortal, 0 if not, -1 on error.
static int
share_objects(PyObject **objs, Py_ssize_t n)
{
    int *status = PyMem_Malloc((n ? n : 1) * sizeof(int));
    if (status == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    int ok = _Py_brc_make_shared(objs, n, 100 * 1000 * 1000 /* 100 ms */,
                                 status);
    if (ok == 0) {
        for (Py_ssize_t i = 0; i < n; i++) {
            if (status[i] == _Py_SHARED_TIMEOUT ||
                status[i] == _Py_SHARED_CHANGED) {
                // objs[i] still has an owner: retry, or give up.
            }
        }
    }
    PyMem_Free(status);
    return ok;
}
```

Code inside CPython includes `"pycore_brc.h"` directly.


## How it works

### The mechanism in one pass

When a list of objects is passed to `_Py_brc_make_shared`, we walk the list
and look at who owns each object. Immortal objects and objects that are
already merged are left alone: they already satisfy the guarantee. The rest
are owned by some thread, possibly the caller itself. We group them by owner,
so each owner gets one request for all of its objects.

We then **notify first, work later**. For every other owner that is still
alive, we take its bucket mutex, link the request into its list and set its
eval-breaker bit. That is all: a few nanoseconds per owner, so every attached
owner learns about its work almost at once and starts on it at its next safe
point, in parallel with everything else.

Only then do we do the work that needs no owner. We merge our own objects. For
an owner that has exited, its objects are abandoned but still carry its id; we
merge them ourselves under its bucket mutex, which is safe because a new
thread that happens to get the same id must register under this same mutex.
For an owner that is detached (blocked in I/O, a lock, `sleep`), we freeze it
by compare-and-swapping its thread state from `DETACHED` to `SUSPENDED`,
merge its objects on its behalf, and release it; while it is suspended it
cannot reattach, so it cannot be refcounting.

Merging is the same code whoever does it, and runs under the owner's bucket
mutex. For each object: if it is merged (or immortal) by now, there is
nothing to do; if a different thread owns it, it is `CHANGED`; otherwise its
local count is folded into the shared count, which is marked merged, and
`ob_tid` is cleared: `MERGED`.

While waiting, we sleep in 1 ms slices, detached, so that stop-the-world can
proceed; we wake up as soon as the last owner has answered. Between slices we
answer requests addressed to us, so two threads asking each other cannot
deadlock, and we retry freezing owners that have blocked since. At the
deadline, whatever is still unanswered is withdrawn: `TIMEOUT`. The function
returns 1 exactly when no object is `TIMEOUT` or `CHANGED`.

### Step by step

1. **Classify** each object: immortal, already merged (`ob_tid == 0`), or
   owned. Owned objects are grouped by owner (including the caller), one
   request per owner.
2. **Notify** every other live owner, under its bucket mutex: link the
   request into its `share_requests` list and set its
   `_PY_EVAL_EXPLICIT_MERGE_BIT`. No merging happens in this step.
3. **Work on behalf**, under the respective bucket mutex: merge the caller's
   own objects; merge objects of owners that have exited; freeze detached
   owners (`_PyThreadState_TrySuspendDetached`), answer all their pending
   requests, and release them.
4. **Attached owners** answer at their next safe point, in
   `_Py_brc_merge_refcounts()`, before merging their BRC queue.
5. **Wait** in 1 ms slices, detached. Between slices: answer requests
   addressed to the caller, and retry step 3 for owners that have detached
   since.
6. **At the deadline**, withdraw unanswered requests under the owner's bucket
   mutex and report `TIMEOUT`.

The detailed protocol, the order of the stores in a merge, the
memory-ordering argument and the request states are in the design note in
`Python/brc.c` (section "Making objects shared").

### What happens to the objects

- **Refcount fields.** The local count is folded into `ob_ref_shared`, which
  gets the `_Py_REF_MERGED` flag; then `ob_ref_local` and `ob_tid` are set to
  0. The total is unchanged. The stores are ordered so that a thread reading
  the two fields concurrently (`Py_REFCNT`) may briefly see too high a count,
  never too low. This differs from stock `_Py_ExplicitMergeRefcount()`, which
  clears the local count first (it must, because the object may be freed as
  soon as it is marked merged); here the caller's reference keeps the object
  alive.
- **Objects in a BRC merge queue** are simply merged. When the owner later
  processes its queue, it merges them again and drops the queue's reference,
  which BRC already supports.
- **Lists, dicts and sets** need no special treatment. A merged container
  has no owner, so every thread that reads it without a lock is a non-owner
  and marks it shared before its first such read, which makes resizes free
  the old storage through QSBR (list and dict resizes by a non-owner also
  mark it shared themselves). This is the normal handling of merged
  containers in stock CPython. The former owner cannot be in the middle of a
  lock-free read when its object is merged: it is at a safe point, suspended
  or gone.

### `_PY_EVAL_EXPLICIT_MERGE_BIT`

This eval-breaker bit tells a thread that it has refcount bookkeeping to do
for other threads. When the thread sees it at a safe point,
`_Py_HandlePending` clears it and calls `_Py_brc_merge_refcounts`, which,
under the thread's bucket mutex, first answers pending make-shared requests
and then merges the thread's BRC queue as before.

- **In stock CPython** the bit means "objects to merge". It is set in exactly
  one place, `_Py_brc_queue_object`, when a thread pushes an object onto an
  *attached* owner's merge queue. That happens when thread B drops a
  reference to an object owned by thread A while the shared count is zero,
  i.e. the reference was counted in A's local count (typically: A created the
  object and handed it to B without B increfing it). B cannot touch A's local
  count, so it queues the object and sets A's bit. This is common in programs
  that create objects in one thread and release them in another
  (producer/consumer queues, thread-pool results, futures); it never happens
  in single-threaded programs. If the owner is detached, no bit is set: the
  queuing thread merges on the owner's behalf (gh-157838).
- **On this branch** the bit is also set when a make-shared request is left
  for an attached owner, so it now means "objects to merge, or make-shared
  requests pending". Reusing it avoids a new bit: all eight event bits are
  taken, and adding one would shift the instrumentation version stored in the
  upper bits of the same word.

Consequences of sharing the bit: if a request is withdrawn after the bit was
set (or answered on the owner's behalf), the owner later runs the handler
with nothing to do (one lock acquisition and an empty merge); and whenever
the bit fires for ordinary BRC reasons, the owner also looks at its request
list, which is usually empty.

### Latency

An attached owner answers at its next eval-breaker check: function entry
(`RESUME`), a loop back-edge (`JUMP_BACKWARD`), the end of a call, or a JIT
side exit. That is usually very soon. An owner running a long C call that
never returns to the interpreter (a big `sorted()`, a regex match) answers
only after it returns, which is what produces `TIMEOUT`. A detached or exited
owner is handled by the caller immediately (step 3), without waiting.

### Locking

At most one bucket mutex is held at a time, and nothing that blocks,
allocates, takes a critical section or runs Python code happens while one is
held. Under a bucket mutex, only two leaf operations take other locks: waking
a waiter (the parking lot's internal raw mutex, as in every contended
`PyMutex_Unlock`) and holding a detached owner suspended (which blocks that
thread from attaching, briefly; the same pattern as gh-157838).


## Limitations relevant to regions

- **Atomic refcounting for good.** Merged objects are refcounted atomically by
  every thread, including the region's own, for the rest of their lifetime.
- **Some references are not counted in the object itself.** Heap types, code
  objects and dicts with per-thread refcounting (module globals) are counted
  through per-thread counters (`tstate->refcounts`) that every thread
  increments non-atomically, regardless of the owner. Objects with deferred
  refcounting can be referenced from other threads' stacks without any
  refcount operation. Neither is visible in the object's refcount; a region
  algorithm that reasons from refcounts must treat these objects specially.
- **Owners stuck in C code** without safe points cause `TIMEOUT`.
- **`CHANGED` is not followed.** It can only arise from thread-id reuse or
  from the GC: during a collection the free-threaded GC reuses `ob_tid` of
  GC-tracked objects and restores it from the thread that allocated the
  memory (`gc_restore_tid`), which can differ from the owner the request was
  addressed to. (Merged objects are always restored as merged.)
- **Interpreter scope.** As with BRC itself, objects are assumed to be
  refcounted only by threads of their own interpreter, and a thread is
  assumed not to run with a second thread state of the same interpreter while
  one is detached.


## Changes to CPython on this branch

| File | Change |
|---|---|
| `Python/brc.c` | New section "Making objects shared" (with design note); hooks: answer requests at safe points (`_Py_brc_merge_refcounts`), when merging on behalf of a detached owner (`_Py_brc_queue_object`), at thread exit, and drop them after fork. |
| `Include/internal/pycore_brc.h` | `_Py_brc_make_shared()`, `_Py_shared_status`, and a per-thread list of pending requests in `struct _brc_thread_state`. |
| `Python/ceval_gil.c` | Comment only: `_PY_EVAL_EXPLICIT_MERGE_BIT` now also means "make-shared requests pending". |
| `Modules/_testinternalcapi.c` | Python-level helpers for testing (below). |
| `Lib/test/test_free_threading/test_make_shared.py` | Tests. |

No existing BRC behaviour changes: the merge queue, its merge loops and the
handling of exited threads are untouched, because merging (unlike
transferring) only ever changes `ob_tid` to 0, which BRC already handles
everywhere. No new eval-breaker bit is added. Nothing here is visible to
Python code or to C extensions that do not opt in to internal headers.


## Building and testing

```sh
./configure --disable-gil --with-pydebug && make -j
./python -m test test_free_threading.test_make_shared -v
```

For a ThreadSanitizer build add `--with-thread-sanitizer` (with clang). On
kernels with high ASLR entropy TSan binaries may fail with "unexpected memory
mapping"; run them under `setarch $(uname -m) -R`.

Test helpers in `_testinternalcapi` (free-threaded build only; they raise
`NotImplementedError` in the default build):

- `make_shared(objects: list, timeout=5.0, summary=False)`: calls the C
  function and returns the statuses as strings (`"shared"`, `"immortal"`,
  `"timeout"`, `"changed"`), or `(all_shared, statuses)` with `summary=True`.
- `refcount_state(obj)`: dict with `owned_by_current_thread`, `ob_tid`,
  `local`, `shared_count`, `shared_flags` (3 means merged), `is_immortal`.
- `spin_without_safe_point(seconds)`: busy-waits while attached without
  reaching a safe point (to provoke `TIMEOUT`).

**A related stock CPython bug.** In free-threaded debug builds,
`assert(Py_REFCNT(mp) > 0)` in `_PyDict_NotifyEvent` can fail in a thread
mutating a dict while stock `_Py_ExplicitMergeRefcount` is merging it (the
assertion reads the two refcount fields between that function's two
updates). It has been reproduced on unmodified 3.15 with a pure-Python script
and is to be reported upstream. Release builds are unaffected.


## Performance

The hooks into existing code paths are the same in kind as in the earlier
(ownership-transfer) version of this branch: a 16-byte list head added to the
thread state, and a check of an (almost always empty) request list when the
merge bit fires, which never happens in single-threaded programs. That
version was benchmarked with pyperformance on free-threaded PGO+LTO release
builds, two independent builds each of the base commit and of the branch:
over 22 benchmarks the branch landed between the two baseline builds (+0.4%
and −1.1% by geometric mean), with all remaining per-benchmark differences
within the spread between builds of identical code. This version has not
been benchmarked separately.
