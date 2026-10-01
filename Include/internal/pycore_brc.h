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

    // Pending make-shared requests addressed to this thread, linked through
    // `struct _brc_share_request.node` (protected by bucket mutex)
    struct llist_node share_requests;
};

// Per-object results of _Py_brc_make_shared()
typedef enum {
    _Py_SHARED_MERGED,      // no owner: refcounted atomically by all threads
    _Py_SHARED_IMMORTAL,    // immortal: refcounting does not apply
    _Py_SHARED_TIMEOUT,     // the owner did not respond in time
    _Py_SHARED_CHANGED,     // owned by a different thread than expected
} _Py_shared_status;

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
// pending make-shared requests addressed to the current thread.
void _Py_brc_merge_refcounts(PyThreadState *tstate);

// Merge the refcounts of `objects[0..n)`, so that afterwards no thread owns
// them and every thread, including the caller, refcounts them atomically. The
// owners are asked to merge at their next safe point (or the merge is done on
// their behalf if they are detached or have exited); the caller waits at most
// `timeout` nanoseconds. Stores one _Py_shared_status per object in
// `results`. A merged object never gets an owner again, so for every object
// reported as merged (or immortal), no thread will ever do plain (non-atomic)
// refcounting on it. The caller must hold a strong reference to every object
// for the duration of the call. Returns 1 if every object is merged or
// immortal, 0 if not (see `results`), and -1 with an exception set on failure
// (only MemoryError).
// Export for '_testinternalcapi' shared extension
PyAPI_FUNC(int) _Py_brc_make_shared(PyObject **objects, Py_ssize_t n,
                                    PyTime_t timeout, int *results);

#endif /* Py_GIL_DISABLED */

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_BRC_H */
