// Implementation of biased reference counting inter-thread queue.
//
// Biased reference counting maintains two refcount fields in each object:
// ob_ref_local and ob_ref_shared. The true refcount is the sum of these two
// fields. In some cases, when refcounting operations are split across threads,
// the ob_ref_shared field can be negative (although the total refcount must
// be at least zero). In this case, the thread that decremented the refcount
// requests that the owning thread give up ownership and merge the refcount
// fields. This file implements the mechanism for doing so.
//
// Each thread state maintains a queue of objects whose refcounts it should
// merge. The thread states are stored in a per-interpreter hash table by
// thread id. The hash table has a fixed size and uses a linked list to store
// thread states within each bucket.
//
// The queueing thread uses the eval breaker mechanism to notify the owning
// thread that it has objects to merge. Additionally, all queued objects are
// merged during GC.
//
// This file also implements _Py_brc_make_shared(), which asks the owners of a
// set of objects to merge them on request. See "Making objects shared" below.
#include "Python.h"
#include "pycore_object.h"      // _Py_ExplicitMergeRefcount
#include "pycore_brc.h"         // struct _brc_thread_state
#include "pycore_ceval.h"       // _Py_set_eval_breaker_bit
#include "pycore_llist.h"       // struct llist_node
#include "pycore_lock.h"        // PyEvent
#include "pycore_pystate.h"     // _PyThreadStateImpl
#include "pycore_time.h"        // _PyDeadline_Init

#ifdef Py_GIL_DISABLED

// Get the hashtable bucket for a given thread id.
static struct _brc_bucket *
get_bucket(PyInterpreterState *interp, uintptr_t tid)
{
    return &interp->brc.table[tid % _Py_BRC_NUM_BUCKETS];
}

// Find the thread state in a hash table bucket by thread id.
static _PyThreadStateImpl *
find_thread_state(struct _brc_bucket *bucket, uintptr_t thread_id)
{
    struct llist_node *node;
    llist_for_each(node, &bucket->root) {
        // Get the containing _PyThreadStateImpl from the linked-list node.
        _PyThreadStateImpl *ts = llist_data(node, _PyThreadStateImpl,
                                            brc.bucket_node);
        if (ts->brc.tid == thread_id) {
            return ts;
        }
    }
    return NULL;
}

static void process_share_requests_locked(struct _brc_thread_state *brc);

// Merge the refcounts of all objects in `stack`, keeping the queue's reference.
static void
merge_queued_refcounts(_PyObjectStack *stack)
{
    for (_PyObjectStackChunk *buf = stack->head; buf != NULL; buf = buf->prev) {
        for (Py_ssize_t i = 0; i < buf->n; i++) {
            _Py_ExplicitMergeRefcount(buf->objs[i], 0);
        }
    }
}

// Release the queue's reference to each merged object. This may run
// destructors, so the bucket mutex must not be held.
static void
decref_merged_objects(_PyObjectStack *stack)
{
    PyObject *ob;
    while ((ob = _PyObjectStack_Pop(stack)) != NULL) {
        Py_DECREF(ob);
    }
}

// Enqueue an object to be merged by the owning thread. This steals a
// reference to the object.
void
_Py_brc_queue_object(PyObject *ob)
{
    PyInterpreterState *interp = _PyInterpreterState_GET();

    uintptr_t ob_tid = _Py_atomic_load_uintptr(&ob->ob_tid);
    if (ob_tid == 0) {
        // The owning thread may have concurrently decided to merge the
        // refcount fields.
        Py_DECREF(ob);
        return;
    }

    struct _brc_bucket *bucket = get_bucket(interp, ob_tid);
    PyMutex_Lock(&bucket->mutex);
    _PyThreadStateImpl *tstate = find_thread_state(bucket, ob_tid);
    if (tstate == NULL) {
        // If we didn't find the owning thread then it must have already exited.
        // It's safe (and necessary) to merge the refcount. Subtract one when
        // merging because we've stolen a reference.
        Py_ssize_t refcount = _Py_ExplicitMergeRefcount(ob, -1);
        PyMutex_Unlock(&bucket->mutex);
        if (refcount == 0) {
            _Py_Dealloc(ob);
        }
        return;
    }

    if (_PyObjectStack_Push(&tstate->brc.objects_to_merge, ob) < 0) {
        PyMutex_Unlock(&bucket->mutex);

        // Fall back to stopping all threads and manually merging the refcount
        // if we can't enqueue the object to be merged.
        _PyEval_StopTheWorld(interp);
        Py_ssize_t refcount = _Py_ExplicitMergeRefcount(ob, -1);
        _PyEval_StartTheWorld(interp);

        if (refcount == 0) {
            _Py_Dealloc(ob);
        }
        return;
    }

    if (_PyThreadState_TrySuspendDetached(&tstate->base)) {
        // The owning thread is detached (e.g. blocked on a lock or in a
        // system call) and may not run Python code again for a long time,
        // so merge its queue on its behalf instead of waiting for it. While
        // it is held in the "suspended" state it cannot attach and therefore
        // cannot touch ob_ref_local or ob_tid. Also answer its pending
        // make-shared requests while we hold it.
        process_share_requests_locked(&tstate->brc);
        _PyObjectStack merged = {0};
        _PyObjectStack_Merge(&merged, &tstate->brc.objects_to_merge);
        merge_queued_refcounts(&merged);
        _PyThreadState_ResumeDetached(&tstate->base);
        PyMutex_Unlock(&bucket->mutex);

        decref_merged_objects(&merged);
        return;
    }

    // Notify owning thread
    _Py_set_eval_breaker_bit(&tstate->base, _PY_EVAL_EXPLICIT_MERGE_BIT);

    PyMutex_Unlock(&bucket->mutex);
}

static void
merge_queued_objects(_PyObjectStack *to_merge)
{
    PyObject *ob;
    while ((ob = _PyObjectStack_Pop(to_merge)) != NULL) {
        // Subtract one when merging because the queue had a reference.
        Py_ssize_t refcount = _Py_ExplicitMergeRefcount(ob, -1);
        if (refcount == 0) {
            _Py_Dealloc(ob);
        }
    }
}

// Process this thread's queue of objects to merge.
void
_Py_brc_merge_refcounts(PyThreadState *tstate)
{
    struct _brc_thread_state *brc = &((_PyThreadStateImpl *)tstate)->brc;
    struct _brc_bucket *bucket = get_bucket(tstate->interp, brc->tid);

    assert(brc->tid == _Py_ThreadId());

    // Append all objects into a local stack. We don't want to hold the lock
    // while calling destructors.
    PyMutex_Lock(&bucket->mutex);
    // We are at a safe point (eval breaker), so also answer make-shared
    // requests addressed to us.
    process_share_requests_locked(brc);
    _PyObjectStack_Merge(&brc->local_objects_to_merge, &brc->objects_to_merge);
    PyMutex_Unlock(&bucket->mutex);

    // Process the local stack until it's empty
    merge_queued_objects(&brc->local_objects_to_merge);
}

void
_Py_brc_init_state(PyInterpreterState *interp)
{
    struct _brc_state *brc = &interp->brc;
    for (Py_ssize_t i = 0; i < _Py_BRC_NUM_BUCKETS; i++) {
        llist_init(&brc->table[i].root);
    }
}

void
_Py_brc_init_thread(PyThreadState *tstate)
{
    struct _brc_thread_state *brc = &((_PyThreadStateImpl *)tstate)->brc;
    uintptr_t tid = _Py_ThreadId();

    // Add ourself to the hashtable
    struct _brc_bucket *bucket = get_bucket(tstate->interp, tid);
    PyMutex_Lock(&bucket->mutex);
    brc->tid = tid;
    llist_init(&brc->share_requests);
    llist_insert_tail(&bucket->root, &brc->bucket_node);
    PyMutex_Unlock(&bucket->mutex);
}

void
_Py_brc_remove_thread(PyThreadState *tstate)
{
    struct _brc_thread_state *brc = &((_PyThreadStateImpl *)tstate)->brc;
    if (brc->tid == 0) {
        // The thread state may have been created, but never bound to a native
        // thread and therefore never added to the hashtable.
        assert(tstate->_status.bound == 0);
        return;
    }

    struct _brc_bucket *bucket = get_bucket(tstate->interp, brc->tid);

    // We need to fully process any objects to merge before removing ourself
    // from the hashtable. It is not safe to perform any refcount operations
    // after we are removed. After that point, other threads treat our objects
    // as abandoned and may merge the objects' refcounts directly.
    bool empty = false;
    while (!empty) {
        // Process the local stack until it's empty
        merge_queued_objects(&brc->local_objects_to_merge);

        PyMutex_Lock(&bucket->mutex);
        empty = (brc->objects_to_merge.head == NULL);
        if (empty) {
            // Once we leave the table, requests for our objects no longer
            // find us, so answer the pending ones in the same critical
            // section. We no longer do any refcounting, so we can merge our
            // objects as their owner.
            process_share_requests_locked(brc);
            llist_remove(&brc->bucket_node);
        }
        else {
            _PyObjectStack_Merge(&brc->local_objects_to_merge,
                                 &brc->objects_to_merge);
        }
        PyMutex_Unlock(&bucket->mutex);
    }

    assert(brc->local_objects_to_merge.head == NULL);
    assert(brc->objects_to_merge.head == NULL);
}

void
_Py_brc_after_fork(PyInterpreterState *interp)
{
    // Unlock all bucket mutexes. Some of the buckets may be locked because
    // locks can be handed off to a parked thread (see lock.c). We don't have
    // to worry about consistency here, because no thread can be actively
    // modifying a bucket, but it might be paused (not yet woken up) on a
    // PyMutex_Lock while holding that lock.
    for (Py_ssize_t i = 0; i < _Py_BRC_NUM_BUCKETS; i++) {
        _PyMutex_at_fork_reinit(&interp->brc.table[i].mutex);
    }

    // Requests addressed to us came from threads that no longer exist in the
    // child; drop them without touching their (stale) memory.
    _PyThreadStateImpl *tstate = (_PyThreadStateImpl *)_PyThreadState_GET();
    llist_init(&tstate->brc.share_requests);
}


// ---------------------------------------------------------------------------
// Making objects shared
//
// _Py_brc_make_shared() lets a thread T1 make a set of objects "shared":
// their refcounts are merged (ob_tid = 0, the whole count in ob_ref_shared,
// flag _Py_REF_MERGED), so that every thread, T1 included, refcounts them
// atomically from then on. A merged object never gets an owner again (only a
// dead object can be resurrected, and the GC keeps merged objects merged), so
// once the call returns, no thread will ever do plain (non-atomic) refcounting
// on an object reported as merged.
//
// Merging an object owned by another thread T3 is only safe when T3 cannot be
// in the middle of a local refcount update: when T3 is at a safe point (eval
// breaker check), when T3 is detached and held in the "suspended" state (as
// BRC already does to merge the queue of a detached owner, gh-157838), or
// when T3 has exited.
//
// Protocol
// --------
// T1 (requester):
//  1. The caller holds a strong reference to every object for the whole call,
//     so the borrowed pointers in requests cannot dangle and no object can be
//     freed while it is being merged.
//  2. Classify: immortal, or already merged (ob_tid == 0), or owned (by T1 or
//     another thread). Group owned objects by ob_tid, one request per owner.
//  3. Notify: for each other owner, under its BRC bucket mutex, link the
//     request into its `share_requests` and set _PY_EVAL_EXPLICIT_MERGE_BIT.
//     This is done for all owners before any merging, so that attached owners
//     start working (in parallel) as early as possible.
//  4. Do the work that needs no owner: merge T1's own objects; under the
//     bucket mutex of an owner found to have exited, merge its abandoned
//     objects (no thread with that id can exist or appear while the mutex is
//     held, since new threads register under it, _Py_brc_init_thread); and
//     merge on behalf of owners that are detached (suspend, merge, resume).
//  5. Wait in ~1 ms slices with PyEvent_WaitTimed(detach=1), so that
//     stop-the-world and threads acting on T1's behalf can proceed. Between
//     slices, answer requests addressed to T1 (so that two threads asking
//     each other cannot deadlock) and retry owners that have detached since.
//  6. At the deadline, cancel each pending request under its owner's bucket
//     mutex: unlink it and report "timeout".
//
// T3 (owner), in _Py_brc_merge_refcounts() at a safe point, or any thread
// holding T3 suspended, or T3 itself on its way out (_Py_brc_remove_thread);
// always under T3's bucket mutex: for each object, if ob_tid is 0 report
// "merged" (or "immortal"); if it is another thread report "changed";
// otherwise merge it (merge_object_locked()) and report "merged". Then
// unlink the request, wake T1 if it was the last one, and store DONE. The
// owner then merges its BRC queue as usual.
//
// Request states: PENDING -> DONE (by whoever answers it) or PENDING ->
// CANCELLED (by the requester). Every transition happens under the owner's
// bucket mutex, and answering a request happens in a single critical
// section. A request is linked into its owner's list exactly while it is
// PENDING after step 3, and an exiting owner answers its pending requests in
// the same critical section that removes it from the table, so a linked
// request always has a live owner.
//
// Merge ordering
// --------------
// merge_object_locked() first adds ob_ref_local into ob_ref_shared (CAS,
// setting _Py_REF_MERGED), then clears ob_ref_local, then ob_tid. Other
// threads read the two count fields separately (Py_REFCNT), so this order
// only ever makes their sum too high in between, never too low; it also keeps
// the invariant that a zero ob_tid implies a merged refcount. Unlike
// _Py_ExplicitMergeRefcount(), which must clear ob_ref_local first because
// the object may be freed as soon as it is marked merged, this order is safe
// here because T1 holds a reference.
//
// Interaction with the BRC merge queue
// ------------------------------------
// No change to the queue is needed. An object may be queued (or about to be
// pushed by a thread that read ob_tid before we merged it) when we merge it.
// BRC already allows this: when the queue is processed, an already-merged
// object is merged again and the queue's reference is dropped, exactly as
// when an owner merges a queued object itself because its local count
// reached zero.
//
// Container storage
// -----------------
// Lists, dicts and sets free their old backing storage through QSBR only if
// they are marked shared (_PyGC_BITS_SHARED). Every non-owner marks a
// container shared before its first lock-free read (and list and dict
// resizes by a non-owner mark it too). A merged container has no owner, so
// every lock-free reader is a non-owner: this is the normal handling of
// merged containers and needs no change. The former owner cannot be in the
// middle of a lock-free read when we merge (it is at a safe point, suspended
// or gone).
//
// Memory ordering
// ---------------
// An owner merging at its safe point publishes its stores with the seq_cst
// CAS on ob_ref_shared and the release store of DONE, which T1 reads with
// acquire (or under the bucket mutex). The owner sees its own stores; other
// threads may still read the old ob_tid for a while, but it is foreign to
// them either way, so they use atomics. In the on-behalf path, the CAS in
// _PyThreadState_TrySuspendDetached() synchronizes with the owner's detach
// and _PyThreadState_ResumeDetached() publishes ob_tid = 0 before the owner
// can reattach. For an exited owner, a new thread with the same id registers
// under the bucket mutex after we release it, so it sees ob_tid = 0.
//
// Known limitations
// -----------------
//  - An owner that is attached but does not reach a safe point (e.g. a long C
//    computation) makes the request time out ("timeout").
//  - "changed" (ob_tid is a different thread than the one asked) can only
//    arise from thread id reuse or from the GC resetting ob_tid of a
//    GC-tracked object to the allocating thread (gc_restore_tid); such
//    objects are not followed to their new owner.
//  - Merged objects are refcounted atomically by every thread for the rest of
//    their lifetime, which is slower for heavily used objects.
//  - References counted outside the object (per-thread refcounts of heap
//    types, code objects and some dicts; deferred refcounting) are not
//    affected.
//  - Like the detached-merge path in _Py_brc_queue_object, the on-behalf path
//    assumes that a thread does not run with another thread state of the
//    same interpreter while this one is detached, and that objects are only
//    refcounted by threads of their own interpreter.

#define SHARE_WAIT_SLICE (1000 * 1000)  // 1 ms, in nanoseconds

enum {
    REQUEST_PENDING,
    REQUEST_DONE,
    REQUEST_CANCELLED,
};

// State shared by all requests of one _Py_brc_make_shared() call. It lives
// until the requester returns, which happens only once every request is DONE
// or CANCELLED.
struct share_call {
    PyEvent done;             // set when `remaining` drops to zero
    Py_ssize_t remaining;     // requests not yet DONE, +1 while posting
};

// A request to one owner. Linked into the owner's `share_requests` while it
// is PENDING (after the notify step).
struct _brc_share_request {
    struct llist_node node;
    struct share_call *call;
    _PyThreadStateImpl *owner;  // set when linked, NULL otherwise
    uintptr_t owner_tid;
    int state;                  // REQUEST_*
    Py_ssize_t n;
    PyObject **objects;         // borrowed: the requester holds references
    int *results;               // _Py_shared_status per object
};

// Merge the refcount of `op`. The caller acts as its owner (see "Merge
// ordering" above) and holds the owner's bucket mutex.
static void
merge_object_locked(PyObject *op)
{
    // Relaxed: we are acting as the owner, the only writer of ob_ref_local.
    uint32_t local = _Py_atomic_load_uint32_relaxed(&op->ob_ref_local);
    Py_ssize_t shared = _Py_atomic_load_ssize_relaxed(&op->ob_ref_shared);
    Py_ssize_t new_shared;
    do {
        Py_ssize_t count = Py_ARITHMETIC_RIGHT_SHIFT(Py_ssize_t, shared,
                                                     _Py_REF_SHARED_SHIFT);
        assert(count + (Py_ssize_t)local >= 1);  // the requester's reference
        new_shared = _Py_REF_SHARED(count + (Py_ssize_t)local,
                                    _Py_REF_MERGED);
        // Seq_cst CAS, like every other update of ob_ref_shared. Non-owners
        // may update it concurrently; on failure `shared` is reloaded.
    } while (!_Py_atomic_compare_exchange_ssize(&op->ob_ref_shared, &shared,
                                                new_shared));
    // Relaxed: ordered for readers by the CAS above, by the bucket mutex, or
    // by the request's DONE state (see "Memory ordering" above).
    _Py_atomic_store_uint32_relaxed(&op->ob_ref_local, 0);
    _Py_atomic_store_uintptr_relaxed(&op->ob_tid, _Py_UNOWNED_TID);
}

// Make one object owned by `owner` shared. The caller holds the owner's
// bucket mutex and is the owner at a safe point, holds it suspended, or has
// found that it has exited.
static int
share_object_locked(PyObject *op, uintptr_t owner)
{
    // Relaxed: only the owner changes ob_tid (to 0, when merging), and we
    // are acting as the owner.
    uintptr_t tid = _Py_atomic_load_uintptr_relaxed(&op->ob_tid);
    if (tid == _Py_UNOWNED_TID) {
        // Merged (or made immortal) since we classified it, or listed twice.
        return _Py_IsImmortal(op) ? _Py_SHARED_IMMORTAL : _Py_SHARED_MERGED;
    }
    if (tid != owner) {
        return _Py_SHARED_CHANGED;
    }
    merge_object_locked(op);
    return _Py_SHARED_MERGED;
}

// Answer `req` and mark it DONE. `req` must not be linked. The caller holds
// the owner's bucket mutex (see share_object_locked()).
static void
complete_request_locked(struct _brc_share_request *req)
{
    struct share_call *call = req->call;
    for (Py_ssize_t i = 0; i < req->n; i++) {
        req->results[i] = share_object_locked(req->objects[i],
                                              req->owner_tid);
    }
    // Wake the requester when this is its last request. This must happen
    // before the DONE store: once the requester sees all its requests DONE it
    // returns and frees `call`. Seq_cst: a read-modify-write shared between
    // the owners of the requests (in different buckets).
    if (_Py_atomic_add_ssize(&call->remaining, -1) == 1) {
        _PyEvent_Notify(&call->done);
    }
    // Release: pairs with the acquire load in request_is_pending(), making
    // the merges and results visible to the requester. This is the last
    // access to the request's memory.
    _Py_atomic_store_int_release(&req->state, REQUEST_DONE);
}

// Answer every pending request addressed to the thread owning `brc`. The
// caller holds its bucket mutex and is that thread at a safe point, holds it
// suspended, or is that thread on its way out.
static void
process_share_requests_locked(struct _brc_thread_state *brc)
{
    struct llist_node *node;
    llist_for_each_safe(node, &brc->share_requests) {
        struct _brc_share_request *req =
            llist_data(node, struct _brc_share_request, node);
        llist_remove(node);
        complete_request_locked(req);
    }
}

// If `owner` is detached, hold it suspended and answer its pending requests
// on its behalf. The caller holds the owner's bucket mutex.
static void
service_detached_owner_locked(_PyThreadStateImpl *owner)
{
    if (_PyThreadState_TrySuspendDetached(&owner->base)) {
        // While suspended the owner cannot attach, so it cannot be in the
        // middle of a local refcount update or write ob_tid.
        process_share_requests_locked(&owner->brc);
        _PyThreadState_ResumeDetached(&owner->base);
    }
}

// Link `req` into `owner`'s list and tell it to look at it. The caller holds
// the owner's bucket mutex.
static void
link_request_locked(struct _brc_share_request *req,
                    _PyThreadStateImpl *owner)
{
    req->owner = owner;
    llist_insert_tail(&owner->brc.share_requests, &req->node);
    _Py_set_eval_breaker_bit(&owner->base, _PY_EVAL_EXPLICIT_MERGE_BIT);
}

static int
request_is_pending(struct _brc_share_request *req)
{
    // Acquire: pairs with the release store of DONE in
    // complete_request_locked(), so that a DONE request's results and merges
    // are visible.
    return _Py_atomic_load_int_acquire(&req->state) == REQUEST_PENDING;
}

// Step 3: tell a live owner about its request. Owners that have exited are
// left for step 4.
static void
notify_owner(PyInterpreterState *interp, struct _brc_share_request *req)
{
    struct _brc_bucket *bucket = get_bucket(interp, req->owner_tid);
    PyMutex_Lock(&bucket->mutex);
    _PyThreadStateImpl *owner = find_thread_state(bucket, req->owner_tid);
    if (owner != NULL) {
        link_request_locked(req, owner);
    }
    PyMutex_Unlock(&bucket->mutex);
}

// Step 4, and the retries in step 5: do whatever part of the work does not
// need the owner to run.
static void
work_on_behalf(PyInterpreterState *interp, struct _brc_share_request *req)
{
    struct _brc_bucket *bucket = get_bucket(interp, req->owner_tid);
    PyMutex_Lock(&bucket->mutex);
    // Relaxed: under the mutex that orders every transition of `state`.
    if (_Py_atomic_load_int_relaxed(&req->state) == REQUEST_PENDING) {
        if (req->owner != NULL) {
            // Linked, so the owner is alive (see "Request states" above).
            service_detached_owner_locked(req->owner);
        }
        else {
            _PyThreadStateImpl *owner = find_thread_state(bucket,
                                                          req->owner_tid);
            if (owner == NULL) {
                // The owner has exited and no thread with its id can appear
                // while we hold the mutex: merge its objects ourselves.
                complete_request_locked(req);
            }
            else {
                // A new thread with that id appeared since step 3; it owns
                // the objects now.
                link_request_locked(req, owner);
                service_detached_owner_locked(owner);
            }
        }
    }
    PyMutex_Unlock(&bucket->mutex);
}

static void
cancel_request(PyInterpreterState *interp, struct _brc_share_request *req)
{
    struct _brc_bucket *bucket = get_bucket(interp, req->owner_tid);
    PyMutex_Lock(&bucket->mutex);
    // Relaxed: under the mutex. If the request was answered first, the mutex
    // makes the answer visible to us.
    if (_Py_atomic_load_int_relaxed(&req->state) == REQUEST_PENDING) {
        assert(req->owner != NULL);  // pending after step 4 implies linked
        llist_remove(&req->node);
        for (Py_ssize_t i = 0; i < req->n; i++) {
            req->results[i] = _Py_SHARED_TIMEOUT;
        }
        // Relaxed: only the requester (us) reads it after this.
        _Py_atomic_store_int_relaxed(&req->state, REQUEST_CANCELLED);
    }
    PyMutex_Unlock(&bucket->mutex);
}

// Answer requests addressed to the calling thread, which acts as their owner.
static void
service_own_requests(_PyThreadStateImpl *tstate)
{
    struct _brc_bucket *bucket = get_bucket(tstate->base.interp,
                                            tstate->brc.tid);
    PyMutex_Lock(&bucket->mutex);
    process_share_requests_locked(&tstate->brc);
    PyMutex_Unlock(&bucket->mutex);
}

// Merge our own objects. We act as their owner: we are not in the middle of
// any refcount operation on them.
static void
share_own_objects(_PyThreadStateImpl *tstate, struct _brc_share_request *req)
{
    struct _brc_bucket *bucket = get_bucket(tstate->base.interp,
                                            tstate->brc.tid);
    PyMutex_Lock(&bucket->mutex);
    complete_request_locked(req);
    PyMutex_Unlock(&bucket->mutex);
}

struct share_entry {
    uintptr_t tid;      // ob_tid when classified
    Py_ssize_t index;   // index into the caller's arrays
};

static int
compare_entries(const void *a, const void *b)
{
    const struct share_entry *x = a, *y = b;
    if (x->tid != y->tid) {
        return x->tid < y->tid ? -1 : 1;
    }
    return x->index < y->index ? -1 : (x->index > y->index);
}

int
_Py_brc_make_shared(PyObject **objects, Py_ssize_t n, PyTime_t timeout,
                    int *results)
{
    _PyThreadStateImpl *tstate = (_PyThreadStateImpl *)_PyThreadState_GET();
    PyInterpreterState *interp = tstate->base.interp;
    uintptr_t me = _Py_ThreadId();
    assert(tstate->brc.tid == me);

    if (n == 0) {
        return 1;
    }
    struct share_entry *entries = PyMem_Malloc(n * sizeof(*entries));
    PyObject **objs = PyMem_Malloc(n * sizeof(*objs));
    int *res = PyMem_Malloc(n * sizeof(*res));
    struct _brc_share_request *reqs = PyMem_Calloc(n, sizeof(*reqs));
    if (entries == NULL || objs == NULL || res == NULL || reqs == NULL) {
        PyMem_Free(entries);
        PyMem_Free(objs);
        PyMem_Free(res);
        PyMem_Free(reqs);
        PyErr_NoMemory();
        return -1;
    }

    // Step 2: classify.
    Py_ssize_t nowned = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *op = objects[i];
        // Relaxed: a stale value is merely some past owner; the owner side
        // re-checks under its bucket mutex.
        uintptr_t tid = _Py_atomic_load_uintptr_relaxed(&op->ob_tid);
        if (_Py_IsImmortal(op)) {
            results[i] = _Py_SHARED_IMMORTAL;
        }
        else if (tid == _Py_UNOWNED_TID) {
            results[i] = _Py_SHARED_MERGED;
        }
        else {
            entries[nowned].tid = tid;
            entries[nowned].index = i;
            nowned++;
        }
    }
    qsort(entries, nowned, sizeof(*entries), compare_entries);

    // One request per owner. `remaining` holds one extra count while
    // posting, so that `done` is not set before every request is counted.
    struct share_call call = {.remaining = 1};
    Py_ssize_t nreqs = 0;
    for (Py_ssize_t k = 0; k < nowned; ) {
        Py_ssize_t start = k;
        while (k < nowned && entries[k].tid == entries[start].tid) {
            objs[k] = objects[entries[k].index];
            k++;
        }
        struct _brc_share_request *req = &reqs[nreqs++];
        req->call = &call;
        req->owner_tid = entries[start].tid;
        req->state = REQUEST_PENDING;
        req->n = k - start;
        req->objects = &objs[start];
        req->results = &res[start];
        // Seq_cst: see complete_request_locked().
        _Py_atomic_add_ssize(&call.remaining, 1);
    }

    // Step 3: notify every other live owner before doing any work.
    for (Py_ssize_t r = 0; r < nreqs; r++) {
        if (reqs[r].owner_tid != me) {
            notify_owner(interp, &reqs[r]);
        }
    }
    // Step 4: our own objects, exited owners, detached owners.
    for (Py_ssize_t r = 0; r < nreqs; r++) {
        if (reqs[r].owner_tid == me) {
            share_own_objects(tstate, &reqs[r]);
        }
        else {
            work_on_behalf(interp, &reqs[r]);
        }
    }
    if (_Py_atomic_add_ssize(&call.remaining, -1) == 1) {
        _PyEvent_Notify(&call.done);
    }

    // Step 5: wait until every request is answered or the deadline passes.
    PyTime_t deadline = _PyDeadline_Init(timeout);
    for (;;) {
        int pending = 0;
        for (Py_ssize_t r = 0; r < nreqs; r++) {
            pending |= request_is_pending(&reqs[r]);
        }
        PyTime_t left = _PyDeadline_Get(deadline);
        if (!pending || left <= 0) {
            break;
        }
        (void)PyEvent_WaitTimed(&call.done, Py_MIN(left, SHARE_WAIT_SLICE),
                                /*detach=*/1);
        service_own_requests(tstate);
        for (Py_ssize_t r = 0; r < nreqs; r++) {
            if (request_is_pending(&reqs[r])) {
                work_on_behalf(interp, &reqs[r]);
            }
        }
    }
    // Step 6: cancel what is left.
    for (Py_ssize_t r = 0; r < nreqs; r++) {
        if (request_is_pending(&reqs[r])) {
            cancel_request(interp, &reqs[r]);
        }
    }
    // Every request is now DONE or CANCELLED: no other thread will access
    // `call` or `reqs` again.

    int all_shared = 1;
    for (Py_ssize_t k = 0; k < nowned; k++) {
        results[entries[k].index] = res[k];
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        if (results[i] == _Py_SHARED_TIMEOUT ||
            results[i] == _Py_SHARED_CHANGED) {
            all_shared = 0;
        }
    }

    PyMem_Free(entries);
    PyMem_Free(objs);
    PyMem_Free(res);
    PyMem_Free(reqs);
    return all_shared;
}

#endif  /* Py_GIL_DISABLED */
