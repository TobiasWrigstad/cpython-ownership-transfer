#ifndef Py_INTERNAL_BRC_H
#define Py_INTERNAL_BRC_H

#include <stdint.h>
#include "pycore_llist.h"           // struct llist_node
#include "pycore_object_stack.h"    // _PyObjectStack

#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#ifdef Py_GIL_DISABLED

// Prime number to avoid correlations with memory addresses.
#define _Py_BRC_NUM_BUCKETS 257

// Hash table bucket
struct _brc_bucket {
    // Mutex protects both the bucket and thread state queues in this bucket.
    PyMutex mutex;

    // Linked list of _PyThreadStateImpl objects hashed to this bucket.
    struct llist_node root;
};

// Per-interpreter biased reference counting state
struct _brc_state {
    // Hash table of thread states by thread-id. Thread states within a bucket
    // are chained using a doubly-linked list.
    struct _brc_bucket table[_Py_BRC_NUM_BUCKETS];
};

// Per-thread biased reference counting state
struct _brc_thread_state {
    // Linked-list of thread states per hash bucket
    struct llist_node bucket_node;

    // Thread-id as determined by _PyThread_Id()
    uintptr_t tid;

    // Objects with refcounts to be merged (protected by bucket mutex)
    _PyObjectStack objects_to_merge;

    // Local stack of objects to be merged (not accessed by other threads)
    _PyObjectStack local_objects_to_merge;

    // Pending ownership-transfer requests addressed to this thread, linked
    // through `struct _brc_ownership_request.node` (protected by bucket mutex)
    struct llist_node ownership_requests;
};

// Per-object results of _Py_brc_take_ownership()
typedef enum {
    _Py_OWNERSHIP_ALREADY_OWNED,  // already owned by the calling thread
    _Py_OWNERSHIP_TRANSFERRED,    // ownership moved to the calling thread
    _Py_OWNERSHIP_IMMORTAL,       // immortal objects are skipped
    _Py_OWNERSHIP_MERGED,         // ob_tid == 0: no owner (not taken over)
    _Py_OWNERSHIP_TIMEOUT,        // the owner did not respond in time
    _Py_OWNERSHIP_CHANGED,        // owned by another thread by then
} _Py_ownership_status;

// Initialize/finalize the per-thread biased reference counting state
void _Py_brc_init_thread(PyThreadState *tstate);
void _Py_brc_remove_thread(PyThreadState *tstate);

// Initialize per-interpreter state
void _Py_brc_init_state(PyInterpreterState *interp);

void _Py_brc_after_fork(PyInterpreterState *interp);

// Enqueues an object to be merged by it's owning thread (tid). This
// steals a reference to the object.
void _Py_brc_queue_object(PyObject *ob);

// Merge the refcounts of queued objects for the current thread. Also handles
// pending ownership-transfer requests addressed to the current thread.
void _Py_brc_merge_refcounts(PyThreadState *tstate);

// Make the current thread the owner of `objects[0..n)`, asking the current
// owners to hand them over at their next safe point (or doing it on their
// behalf if they are detached). Waits at most `timeout` nanoseconds. Stores
// one _Py_ownership_status per object in `results`. The statuses describe
// the objects when the call returns: every object reported as transferred or
// already owned is then owned by the current thread, and every object
// reported as merged has no owner. Until the current thread next reaches a
// safe point or detaches, no other thread can do plain (non-atomic)
// refcounting on any of these objects; only objects reported as timed out or
// changed may still have another owner. The caller must hold a strong
// reference to every object for the duration of the call. Returns 1 if that
// holds for every object (no object timed out or changed), 0 if not (see
// `results`), and -1 with an exception set on failure (only MemoryError).
// Export for '_testinternalcapi' shared extension
PyAPI_FUNC(int) _Py_brc_take_ownership(PyObject **objects, Py_ssize_t n,
                                       PyTime_t timeout, int *results);

#endif /* Py_GIL_DISABLED */

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_BRC_H */
