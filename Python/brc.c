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
// This file also implements ownership transfer: a thread may ask the owners
// of a set of objects to hand ownership over to it. See the "Ownership
// transfer" section below.
#include "Python.h"
#include "pycore_object.h"      // _Py_ExplicitMergeRefcount
#include "pycore_brc.h"         // struct _brc_thread_state
#include "pycore_ceval.h"       // _Py_set_eval_breaker_bit
#include "pycore_gc.h"          // _PyObject_GC_SET_SHARED
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

static void process_ownership_requests_locked(struct _brc_thread_state *brc);

// Returns true if a queued object is now owned by a thread other than `tid`,
// i.e. its ownership was transferred after it was queued. This cannot happen:
// objects are pushed to `tid`'s queue only while holding its bucket mutex
// after re-checking that `tid` owns them (see _Py_brc_queue_object), and a
// transfer, also done under that mutex, skips objects flagged as queued. A
// zero ob_tid is fine: the owner may have merged the object itself.
static int
queued_object_changed_owner(PyObject *ob, uintptr_t tid)
{
    // Relaxed: only used to route the object; the merge itself is atomic.
    uintptr_t ob_tid = _Py_atomic_load_uintptr_relaxed(&ob->ob_tid);
    int changed = (ob_tid != 0 && ob_tid != tid);
    assert(!changed);
    return changed;
}

// Merge the refcounts of all objects in `stack` owned by `tid`, keeping the
// queue's reference. Objects owned by another thread are left unmerged (still
// flagged _Py_REF_QUEUED) for decref_merged_objects() to forward.
static void
merge_queued_refcounts(_PyObjectStack *stack, uintptr_t tid)
{
    for (_PyObjectStackChunk *buf = stack->head; buf != NULL; buf = buf->prev) {
        for (Py_ssize_t i = 0; i < buf->n; i++) {
            if (!queued_object_changed_owner(buf->objs[i], tid)) {
                _Py_ExplicitMergeRefcount(buf->objs[i], 0);
            }
        }
    }
}

// Release the queue's reference to each merged object, and forward objects
// that merge_queued_refcounts() skipped to their current owner. This may run
// destructors, so the bucket mutex must not be held.
static void
decref_merged_objects(_PyObjectStack *stack)
{
    PyObject *ob;
    while ((ob = _PyObjectStack_Pop(stack)) != NULL) {
        // Relaxed: nothing else can change the flags of an object that is
        // queued but not merged, and we just popped it from the only queue.
        Py_ssize_t shared = _Py_atomic_load_ssize_relaxed(&ob->ob_ref_shared);
        if (_Py_REF_IS_QUEUED(shared)) {
            _Py_brc_queue_object(ob);  // steals the queue's reference
        }
        else {
            Py_DECREF(ob);
        }
    }
}

// Enqueue an object to be merged by the owning thread. This steals a
// reference to the object.
void
_Py_brc_queue_object(PyObject *ob)
{
    PyInterpreterState *interp = _PyInterpreterState_GET();

    uintptr_t ob_tid;
    struct _brc_bucket *bucket;
    for (;;) {
        ob_tid = _Py_atomic_load_uintptr(&ob->ob_tid);
        if (ob_tid == 0) {
            // The owning thread may have concurrently decided to merge the
            // refcount fields.
            Py_DECREF(ob);
            return;
        }

        bucket = get_bucket(interp, ob_tid);
        PyMutex_Lock(&bucket->mutex);
        // The owner may have transferred the object to another thread after
        // we read ob_tid. Transfers happen under the old owner's bucket
        // mutex, so this re-read sees the current owner. Relaxed: the mutex
        // orders it after any transfer. The object cannot move to another
        // thread while we hold the mutex (the owner may still merge it,
        // setting ob_tid to 0, as before).
        if (_Py_atomic_load_uintptr_relaxed(&ob->ob_tid) == ob_tid) {
            break;
        }
        PyMutex_Unlock(&bucket->mutex);
    }

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
        // cannot touch ob_ref_local or ob_tid.
        // Also answer its pending ownership-transfer requests, which must be
        // done before merging so that queued objects are reported as such.
        process_ownership_requests_locked(&tstate->brc);
        _PyObjectStack merged = {0};
        _PyObjectStack_Merge(&merged, &tstate->brc.objects_to_merge);
        merge_queued_refcounts(&merged, tstate->brc.tid);
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
merge_queued_objects(_PyObjectStack *to_merge, uintptr_t tid)
{
    PyObject *ob;
    while ((ob = _PyObjectStack_Pop(to_merge)) != NULL) {
        if (queued_object_changed_owner(ob, tid)) {
            // Forward to the current owner (steals the queue's reference).
            _Py_brc_queue_object(ob);
            continue;
        }
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
    // We are at a safe point (eval breaker), so this is where we hand over
    // objects that other threads asked for. Do it before merging so that
    // queued objects are reported as such.
    process_ownership_requests_locked(brc);
    _PyObjectStack_Merge(&brc->local_objects_to_merge, &brc->objects_to_merge);
    PyMutex_Unlock(&bucket->mutex);

    // Process the local stack until it's empty
    merge_queued_objects(&brc->local_objects_to_merge, brc->tid);
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
    llist_init(&brc->ownership_requests);
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
        merge_queued_objects(&brc->local_objects_to_merge, brc->tid);

        PyMutex_Lock(&bucket->mutex);
        empty = (brc->objects_to_merge.head == NULL);
        if (empty) {
            // Once we leave the table, requests for our objects no longer
            // find us, so answer the pending ones in the same critical
            // section. We no longer do any refcounting, so it is safe to
            // hand over our objects as an owner would.
            process_ownership_requests_locked(brc);
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
    llist_init(&tstate->brc.ownership_requests);
}


// ---------------------------------------------------------------------------
// Ownership transfer
//
// _Py_brc_take_ownership() lets a thread T1 become the owner (ob_tid) of
// objects owned by other threads. Only the owner may write ob_ref_local (with
// plain/relaxed stores), so ownership can only change hands when the owner T3
// cannot be in the middle of a local refcount operation: either T3 is at a
// safe point (eval breaker check), or T3 is detached and held in the
// "suspended" state by another thread, which is the same mechanism BRC uses
// to merge the queue of a detached owner (gh-157838).
//
// Protocol
// --------
// T1 (requester):
//  1. The caller holds a strong reference to every object for the whole call,
//     so the borrowed pointers in requests cannot dangle.
//  2. Classify each object: immortal, already owned by T1, merged
//     (ob_tid == 0), or foreign. Group foreign objects by ob_tid, one request
//     per owner.
//  3. Mark lists, dicts and sets as shared (see "Container storage").
//  4. For each request, under the owner's BRC bucket mutex: look up the live
//     thread state for that ob_tid. If there is none, the owner has exited
//     and its objects are abandoned: answer the request ourselves, as the
//     owner would (see "Exited owners"). Otherwise link the request into
//     the owner's `ownership_requests`, then either answer it on the owner's
//     behalf if the owner is detached, or set _PY_EVAL_EXPLICIT_MERGE_BIT.
//  5. Wait in ~1 ms slices with PyEvent_WaitTimed(detach=1). Detaching lets
//     stop-the-world and threads acting on our behalf proceed. Between
//     slices: answer requests addressed to T1 itself (so that two threads
//     taking objects from each other cannot deadlock), and retry the
//     on-behalf path for owners that have detached since.
//  6. At the deadline, cancel each still-pending request under its owner's
//     bucket mutex: unlink it and report "timeout".
//  7. Re-check every object reported "transferred" or "already-owned": while
//     T1 waited, another thread may have taken it from T1 (T1 answers
//     requests between slices, and can be held suspended while detached), or
//     it may have been merged. Report "changed", "merged" or "immortal" for
//     those. T1 is
//     attached from here on, so no other thread can give these objects a new
//     owner until T1 next reaches a safe point or detaches: when the call
//     returns, every object reported "transferred" or "already-owned" is
//     owned by T1.
//
// T3 (owner), in _Py_brc_merge_refcounts() when the eval breaker bit is set,
// or any thread holding a detached T3 suspended; always under T3's bucket
// mutex (process_ownership_requests_locked()):
//  - for each object: if ob_tid != T3, report "merged" or "immortal" if
//    ob_tid is 0, else "changed"; if the _Py_REF_QUEUED flag is set, merge
//    the refcount now and report "merged" (see "Queued objects"); otherwise
//    move the whole count to ob_ref_local (see "Refcount fields"), store
//    ob_tid = T1 (release) and report "transferred".
//  - unlink the request, wake T1 if it is the last one, and store DONE.
//  Then the owner merges its BRC queue as usual.
//
// Request states: PENDING -> DONE (by the owner side) or PENDING -> CANCELLED
// (by the requester). Every transition happens under the owner's bucket
// mutex, and claiming, transferring and completing a request happen in a
// single critical section, so a separate CLAIMED state is not needed: when
// the requester holds the mutex, a request is either still PENDING (and
// linked, so it can be cancelled) or already DONE. A request is PENDING
// exactly while it is linked into its owner's list, and an exiting owner
// answers its pending requests in the same critical section that removes it
// from the table, so a PENDING request always has a live owner.
//
// Guarantee on return
// -------------------
// When the call returns, no thread other than T1 can do plain (non-atomic)
// refcounting on an object reported "transferred" or "already-owned" (owned
// by T1) or "merged" (no owner: every thread uses atomics, and nothing can
// give it an owner before T1's next safe point, since T1 holds a reference
// and the GC cannot run). Only "timeout" and "changed" objects may still have
// another owner.
//
// Queued objects
// --------------
// An object flagged _Py_REF_QUEUED is in (or entering) its owner's merge
// queue, and the owner keeps doing plain refcounting on it until it merges
// the queue. Transferring it would break the queue invariant below, so the
// owner side merges it right away instead (merge_queued_locked(), keeping
// the queue's reference). Merging an object that is still queued is
// already allowed by BRC: when the queue is processed, the object is merged
// again and the queue's reference is dropped, just as when an owner merges a
// queued object itself because its local count reached zero. A thread still
// on its way to push the object re-reads ob_tid under the bucket mutex, sees
// zero and just drops its reference (see "BRC queue race").
//
// Exited owners
// -------------
// Objects of an exited thread keep its id in ob_tid. Leaving them there is
// not safe: a new thread may get the same id and would then do plain
// refcounting on them as their owner. But while we hold the bucket mutex for
// that id and find no thread state, no thread with that id exists or can
// appear, since a new thread registers under the same mutex
// (_Py_brc_init_thread), and nobody does plain refcounting on the objects.
// Other threads only merge them, under the same mutex (_Py_brc_queue_object).
// So the requester answers the request itself with the same per-object logic
// as a live owner, which also moves ob_tid away from the dead id.
//
// Refcount fields
// ---------------
// Only the sum of ob_ref_local and ob_ref_shared matters for correctness, so
// any split is valid. But BRC gives up ownership whenever ob_ref_local drops
// to zero while other references remain (_Py_MergeZeroLocalRefcount), and
// ob_ref_local does not record whose references it counts. Just before the
// transfer it holds T3's references, while T1's (taken as a non-owner, e.g.
// the caller's reference to each object) are in ob_ref_shared. Keeping that
// split, T1 would lose ownership as soon as it dropped more references than
// T3 had counted locally; swapping the counts has the mirror-image problem.
// Instead the owner moves the whole count into ob_ref_local and leaves
// ob_ref_shared with a zero count (keeping its flags): the state of an object
// freshly allocated by T1. T1's own releases can then not merge the object
// while references remain. Releases by other threads (including T3) go to
// ob_ref_shared; the first one that would make its count negative queues the
// object for T1 to merge, exactly as for any object shared by its owner. The
// move is skipped if the total would not fit in ob_ref_local.
//
// Memory ordering
// ---------------
// T3's plain/relaxed stores to ob_ref_local happen before its release store
// of ob_tid and, after that, its release store of DONE. T1 reads DONE with
// acquire (or under the bucket mutex), so it sees ob_ref_local and ob_tid as
// T3 left them before T1 does any refcounting as the new owner. T3 itself
// wrote ob_tid, so it immediately takes the non-owner (atomic) paths. Other
// threads may read a stale ob_tid (T3) or the new one (T1); both are foreign
// to them, so they use ob_ref_shared either way. In the on-behalf path, the
// seq_cst CAS in _PyThreadState_TrySuspendDetached() synchronizes with T3's
// detach, and _PyThreadState_ResumeDetached() publishes the new ob_tid to T3
// before it can reattach.
//
// BRC queue race
// --------------
// A non-owner N that drops a reference sets _Py_REF_QUEUED (CAS) and then
// pushes the object onto its owner's queue. Previously N read ob_tid once,
// before taking the bucket mutex, so T3 could transfer the object to T1 in
// between and N would still push it onto T3's queue; T3 would then merge an
// object it no longer owns, racing with T1's plain stores to ob_ref_local.
// Now N re-reads ob_tid under the bucket mutex and retries with the new owner
// if it changed. Since N sets the flag before taking the mutex and T3 checks
// the flag under the mutex, either T3 sees the flag and merges the object
// instead of transferring it, or N takes the mutex after the transfer and
// sees T1. Hence every object in
// T3's queue has ob_tid == T3 (or 0 if T3 merged it itself); the merge loops
// assert this and forward the object to its current owner otherwise.
//
// Container storage
// -----------------
// Lists, dicts and sets free their backing storage immediately when the
// owner resizes them, unless the _PyGC_BITS_SHARED bit says another thread
// may be reading it without a lock (then QSBR delays the free). A non-owner
// sets the bit on its first lock-free read, but the new owner T1 would never
// set it for T3's reads. We therefore set it before posting the request,
// under the object's critical section like the "ensure shared" helpers do.
// This is conservative: the bit is set even if the transfer fails.
//
// Known limitations
// -----------------
//  - Merged objects (ob_tid == 0) are reported "merged" and not taken over,
//    and queued objects are merged rather than transferred.
//  - An owner that is attached but does not reach a safe point (e.g. a long
//    C computation) makes the request time out.
//  - The GC reuses ob_tid of tracked objects during a collection and then
//    restores it from the mimalloc segment's thread id (gc_restore_tid), which
//    undoes a transfer of a GC-tracked object. This is to be addressed
//    separately.
//  - Like the detached-merge path in _Py_brc_queue_object, the on-behalf path
//    assumes that a thread does not run with another thread state of the
//    same interpreter while this one is detached. Likewise, like BRC's own
//    handling of abandoned objects, claiming objects of exited threads
//    assumes that objects are only refcounted by threads of their own
//    interpreter (the thread table is per interpreter).
//  - The ownership guarantee on return lasts only until T1 next reaches a
//    safe point or detaches; nothing stops other threads from then taking
//    the objects.

#define OWNERSHIP_WAIT_SLICE (1000 * 1000)  // 1 ms, in nanoseconds

enum {
    REQUEST_PENDING,
    REQUEST_DONE,
    REQUEST_CANCELLED,
};

// State shared by all requests of one _Py_brc_take_ownership() call. It lives
// until the requester returns, which happens only once every request is DONE
// or CANCELLED.
struct ownership_call {
    PyEvent done;             // set when `remaining` drops to zero
    Py_ssize_t remaining;     // requests not yet DONE, +1 while posting
    uintptr_t requester;      // thread id of the requesting thread
};

// A request to one owner. Linked into the owner's `ownership_requests`
// exactly while it is PENDING.
struct _brc_ownership_request {
    struct llist_node node;
    struct ownership_call *call;
    _PyThreadStateImpl *owner;  // NULL if no live owner was found
    uintptr_t owner_tid;
    int state;                  // REQUEST_*
    Py_ssize_t n;
    PyObject **objects;         // borrowed: the requester holds references
    int *results;               // _Py_ownership_status per object
};

// Merge the refcount of a queued object on behalf of its owner (see "Queued
// objects" above). Unlike _Py_ExplicitMergeRefcount(), which must clear
// ob_ref_local first because the object may be freed as soon as it is marked
// merged, this folds ob_ref_local into ob_ref_shared first and clears it
// afterwards, so that a concurrent reader summing the fields over-counts
// rather than under-counts (the shared count may be negative here). That is
// safe because the requester holds a reference. `local` and `shared` are the
// current values; the caller acts as the owner under its bucket mutex.
static void
merge_queued_locked(PyObject *op, uint32_t local, Py_ssize_t shared)
{
    Py_ssize_t new_shared;
    do {
        Py_ssize_t count = Py_ARITHMETIC_RIGHT_SHIFT(Py_ssize_t, shared,
                                                     _Py_REF_SHARED_SHIFT);
        assert(count + (Py_ssize_t)local >= 1);
        new_shared = _Py_REF_SHARED(count + (Py_ssize_t)local,
                                    _Py_REF_MERGED);
        // Seq_cst CAS, like every other update of ob_ref_shared; on failure
        // `shared` is reloaded.
    } while (!_Py_atomic_compare_exchange_ssize(&op->ob_ref_shared, &shared,
                                                new_shared));
    // Relaxed: we are acting as the owner. ob_tid is cleared last, keeping
    // the invariant that a zero ob_tid implies a merged refcount. Threads
    // that re-read ob_tid under the bucket mutex see it once we release it.
    _Py_atomic_store_uint32_relaxed(&op->ob_ref_local, 0);
    _Py_atomic_store_uintptr_relaxed(&op->ob_tid, _Py_UNOWNED_TID);
}

// Hand one object from `owner` to `requester`. The caller holds the owner's
// bucket mutex and is either the owner at a safe point, holds the owner
// suspended, or has found that the owner has exited; so no local refcount
// operation on the object can be in progress and no other thread can change
// ob_tid to another thread id.
static int
transfer_object_locked(PyObject *op, uintptr_t owner, uintptr_t requester)
{
    // Relaxed: only the owner changes ob_tid (to 0 when merging), and we are
    // acting as the owner.
    uintptr_t tid = _Py_atomic_load_uintptr_relaxed(&op->ob_tid);
    if (tid == requester) {
        // Listed more than once: an earlier entry already moved it.
        return _Py_OWNERSHIP_ALREADY_OWNED;
    }
    if (tid == _Py_UNOWNED_TID) {
        // Merged (or made immortal) by the owner since we classified it.
        return _Py_IsImmortal(op) ? _Py_OWNERSHIP_IMMORTAL
                                  : _Py_OWNERSHIP_MERGED;
    }
    if (tid != owner) {
        return _Py_OWNERSHIP_CHANGED;
    }
    // Relaxed: a non-owner sets _Py_REF_QUEUED before taking this bucket
    // mutex to push the object; the mutex orders that store before this load,
    // or orders its re-read of ob_tid after our store below (see "BRC queue
    // race" above).
    Py_ssize_t shared = _Py_atomic_load_ssize_relaxed(&op->ob_ref_shared);
    // Relaxed: we are acting as the owner, the only writer of ob_ref_local.
    uint32_t local = _Py_atomic_load_uint32_relaxed(&op->ob_ref_local);
    // Non-owners read the two fields separately (Py_REFCNT), so each update
    // below is ordered to only ever make their sum too high in between,
    // never too low: debug assertions in other threads check that it is
    // positive. This relies on the requester's reference: the object cannot
    // be freed while we update it.
    for (;;) {
        if (_Py_REF_IS_QUEUED(shared)) {
            // See "Queued objects" above.
            merge_queued_locked(op, local, shared);
            return _Py_OWNERSHIP_MERGED;
        }
        assert(!_Py_REF_IS_MERGED(shared));
        // Move the whole count to ob_ref_local (see "Refcount fields"
        // above), unless it would not fit; the flags are kept. The count is
        // positive here (a count is only negative after the object was
        // queued), so storing the new local count first over-counts.
        Py_ssize_t count = Py_ARITHMETIC_RIGHT_SHIFT(Py_ssize_t, shared,
                                                     _Py_REF_SHARED_SHIFT);
        Py_ssize_t total = (Py_ssize_t)local + count;
        assert(count >= 0);
        if (count == 0 || total >= (Py_ssize_t)_Py_IMMORTAL_REFCNT_LOCAL) {
            break;
        }
        // Relaxed: we are acting as the owner, the only writer.
        _Py_atomic_store_uint32_relaxed(&op->ob_ref_local, (uint32_t)total);
        // Seq_cst CAS, like every other update of ob_ref_shared. Non-owners
        // may update it concurrently; on failure `shared` is reloaded and we
        // restore ob_ref_local before trying again.
        if (_Py_atomic_compare_exchange_ssize(
                &op->ob_ref_shared, &shared,
                shared & _Py_REF_SHARED_FLAG_MASK)) {
            break;
        }
        _Py_atomic_store_uint32_relaxed(&op->ob_ref_local, local);
    }
    // Release: the owner's earlier stores to ob_ref_local happen before the
    // new ob_tid is visible. The requester synchronizes through the request's
    // DONE state, which is stored (release) after this.
    _Py_atomic_store_uintptr_release(&op->ob_tid, requester);
    return _Py_OWNERSHIP_TRANSFERRED;
}

// Answer `req` and mark it DONE. `req` must not be linked. The caller holds
// the owner's bucket mutex (see transfer_object_locked()).
static void
complete_request_locked(struct _brc_ownership_request *req)
{
    struct ownership_call *call = req->call;
    for (Py_ssize_t i = 0; i < req->n; i++) {
        req->results[i] = transfer_object_locked(
            req->objects[i], req->owner_tid, call->requester);
    }
    // Wake the requester when this is its last request. This must happen
    // before the DONE store: once the requester sees all its requests DONE it
    // returns and frees `call`. Seq_cst: a read-modify-write shared between
    // the owners of the requests (in different buckets).
    if (_Py_atomic_add_ssize(&call->remaining, -1) == 1) {
        _PyEvent_Notify(&call->done);
    }
    // Release: pairs with the acquire load in request_is_pending(), making
    // the ob_tid stores and results visible to the requester. This is the
    // last access to the request's memory.
    _Py_atomic_store_int_release(&req->state, REQUEST_DONE);
}

// Answer every pending request addressed to the thread owning `brc`. The
// caller holds its bucket mutex and is that thread at a safe point, or holds
// it suspended.
static void
process_ownership_requests_locked(struct _brc_thread_state *brc)
{
    struct llist_node *node;
    llist_for_each_safe(node, &brc->ownership_requests) {
        struct _brc_ownership_request *req =
            llist_data(node, struct _brc_ownership_request, node);
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
        process_ownership_requests_locked(&owner->brc);
        _PyThreadState_ResumeDetached(&owner->base);
    }
}

static void
post_request(PyInterpreterState *interp, struct _brc_ownership_request *req)
{
    struct _brc_bucket *bucket = get_bucket(interp, req->owner_tid);
    PyMutex_Lock(&bucket->mutex);
    req->owner = find_thread_state(bucket, req->owner_tid);
    if (req->owner == NULL) {
        // The owner has exited; see "Exited owners" above.
        complete_request_locked(req);
    }
    else {
        llist_insert_tail(&req->owner->brc.ownership_requests, &req->node);
        service_detached_owner_locked(req->owner);
        // Relaxed: we hold the mutex that orders all accesses to `state`
        // other than the requester's own acquire loads.
        if (_Py_atomic_load_int_relaxed(&req->state) == REQUEST_PENDING) {
            _Py_set_eval_breaker_bit(&req->owner->base,
                                     _PY_EVAL_EXPLICIT_MERGE_BIT);
        }
    }
    PyMutex_Unlock(&bucket->mutex);
}

static int
request_is_pending(struct _brc_ownership_request *req)
{
    // Acquire: pairs with the release store of DONE in
    // complete_request_locked(), so that a DONE request's results and ob_tid
    // stores are visible.
    return _Py_atomic_load_int_acquire(&req->state) == REQUEST_PENDING;
}

// Retry the on-behalf path for an owner that may have detached since the
// request was posted.
static void
retry_request(PyInterpreterState *interp, struct _brc_ownership_request *req)
{
    struct _brc_bucket *bucket = get_bucket(interp, req->owner_tid);
    PyMutex_Lock(&bucket->mutex);
    // Relaxed: under the mutex that orders every transition of `state`.
    if (_Py_atomic_load_int_relaxed(&req->state) == REQUEST_PENDING) {
        // PENDING implies linked, which implies the owner is still alive.
        service_detached_owner_locked(req->owner);
    }
    PyMutex_Unlock(&bucket->mutex);
}

static void
cancel_request(PyInterpreterState *interp, struct _brc_ownership_request *req)
{
    struct _brc_bucket *bucket = get_bucket(interp, req->owner_tid);
    PyMutex_Lock(&bucket->mutex);
    // Relaxed: under the mutex. If the owner answered first, the mutex makes
    // its stores visible to us.
    if (_Py_atomic_load_int_relaxed(&req->state) == REQUEST_PENDING) {
        llist_remove(&req->node);
        for (Py_ssize_t i = 0; i < req->n; i++) {
            req->results[i] = _Py_OWNERSHIP_TIMEOUT;
        }
        // Relaxed: only the requester (us) reads it after this.
        _Py_atomic_store_int_relaxed(&req->state, REQUEST_CANCELLED);
    }
    PyMutex_Unlock(&bucket->mutex);
}

// Answer requests addressed to the calling thread.
static void
service_own_requests(_PyThreadStateImpl *tstate)
{
    struct _brc_bucket *bucket = get_bucket(tstate->base.interp,
                                            tstate->brc.tid);
    PyMutex_Lock(&bucket->mutex);
    process_ownership_requests_locked(&tstate->brc);
    PyMutex_Unlock(&bucket->mutex);
}

struct ownership_entry {
    uintptr_t tid;      // ob_tid when classified
    Py_ssize_t index;   // index into the caller's arrays
};

static int
compare_entries(const void *a, const void *b)
{
    const struct ownership_entry *x = a, *y = b;
    if (x->tid != y->tid) {
        return x->tid < y->tid ? -1 : 1;
    }
    return x->index < y->index ? -1 : (x->index > y->index);
}

// Set the shared bit on containers that free their storage based on it, so
// that the new owner never frees storage that another thread may still be
// reading without a lock.
static void
mark_container_shared(PyObject *op)
{
    if (PyList_Check(op) || PyDict_Check(op) || PyAnySet_Check(op)) {
        Py_BEGIN_CRITICAL_SECTION(op);
        if (!_PyObject_GC_IS_SHARED(op)) {
            _PyObject_GC_SET_SHARED(op);
        }
        Py_END_CRITICAL_SECTION();
    }
}

int
_Py_brc_take_ownership(PyObject **objects, Py_ssize_t n, PyTime_t timeout,
                       int *results)
{
    _PyThreadStateImpl *tstate = (_PyThreadStateImpl *)_PyThreadState_GET();
    PyInterpreterState *interp = tstate->base.interp;
    uintptr_t me = _Py_ThreadId();
    assert(tstate->brc.tid == me);

    if (n == 0) {
        return 1;
    }
    struct ownership_entry *entries = PyMem_Malloc(n * sizeof(*entries));
    PyObject **objs = PyMem_Malloc(n * sizeof(*objs));
    int *res = PyMem_Malloc(n * sizeof(*res));
    struct _brc_ownership_request *reqs = PyMem_Calloc(n, sizeof(*reqs));
    if (entries == NULL || objs == NULL || res == NULL || reqs == NULL) {
        PyMem_Free(entries);
        PyMem_Free(objs);
        PyMem_Free(res);
        PyMem_Free(reqs);
        PyErr_NoMemory();
        return -1;
    }

    // Classify.
    Py_ssize_t nforeign = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *op = objects[i];
        // Relaxed: a stale value is merely some past owner; the owner side
        // re-checks under its bucket mutex.
        uintptr_t tid = _Py_atomic_load_uintptr_relaxed(&op->ob_tid);
        if (_Py_IsImmortal(op)) {
            results[i] = _Py_OWNERSHIP_IMMORTAL;
        }
        else if (tid == me) {
            results[i] = _Py_OWNERSHIP_ALREADY_OWNED;
        }
        else if (tid == _Py_UNOWNED_TID) {
            results[i] = _Py_OWNERSHIP_MERGED;
        }
        else {
            entries[nforeign].tid = tid;
            entries[nforeign].index = i;
            nforeign++;
        }
    }

    // Group foreign objects by owner.
    qsort(entries, nforeign, sizeof(*entries), compare_entries);
    for (Py_ssize_t k = 0; k < nforeign; k++) {
        objs[k] = objects[entries[k].index];
        mark_container_shared(objs[k]);
    }

    // Holds one extra count while posting, so that `done` is not set before
    // every request has been posted.
    struct ownership_call call = {.remaining = 1, .requester = me};
    Py_ssize_t nreqs = 0;
    for (Py_ssize_t k = 0; k < nforeign; ) {
        Py_ssize_t start = k;
        while (k < nforeign && entries[k].tid == entries[start].tid) {
            k++;
        }
        struct _brc_ownership_request *req = &reqs[nreqs++];
        req->call = &call;
        req->owner_tid = entries[start].tid;
        req->state = REQUEST_PENDING;
        req->n = k - start;
        req->objects = &objs[start];
        req->results = &res[start];
        // Seq_cst: see complete_request_locked().
        _Py_atomic_add_ssize(&call.remaining, 1);
        post_request(interp, req);
    }
    if (_Py_atomic_add_ssize(&call.remaining, -1) == 1) {
        _PyEvent_Notify(&call.done);
    }

    // Wait until every request is answered or the deadline passes.
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
        (void)PyEvent_WaitTimed(&call.done,
                                Py_MIN(left, OWNERSHIP_WAIT_SLICE),
                                /*detach=*/1);
        service_own_requests(tstate);
        for (Py_ssize_t r = 0; r < nreqs; r++) {
            if (request_is_pending(&reqs[r])) {
                retry_request(interp, &reqs[r]);
            }
        }
    }
    for (Py_ssize_t r = 0; r < nreqs; r++) {
        if (request_is_pending(&reqs[r])) {
            cancel_request(interp, &reqs[r]);
        }
    }
    // Every request is now DONE or CANCELLED: no other thread will access
    // `call` or `reqs` again.

    for (Py_ssize_t k = 0; k < nforeign; k++) {
        results[entries[k].index] = res[k];
    }

    // Report the state on return (step 7 of the protocol). From here on we
    // stay attached (nothing below blocks), so ob_tid of these objects can
    // only be changed by us.
    for (Py_ssize_t i = 0; i < n; i++) {
        if (results[i] != _Py_OWNERSHIP_TRANSFERRED &&
            results[i] != _Py_OWNERSHIP_ALREADY_OWNED) {
            continue;
        }
        // Relaxed: a transfer to us was made visible by the acquire load of
        // DONE or by the bucket mutex; transfers away from us and merges on
        // our behalf happened while we were detached, and re-attaching
        // synchronizes with them.
        uintptr_t tid = _Py_atomic_load_uintptr_relaxed(&objects[i]->ob_tid);
        if (tid != me) {
            // Given to another thread while we waited, reset by the GC
            // (gc_restore_tid), or merged (or made immortal).
            if (tid != _Py_UNOWNED_TID) {
                results[i] = _Py_OWNERSHIP_CHANGED;
            }
            else if (_Py_IsImmortal(objects[i])) {
                results[i] = _Py_OWNERSHIP_IMMORTAL;
            }
            else {
                results[i] = _Py_OWNERSHIP_MERGED;
            }
        }
    }

    // Summary: only objects reported as timed out or changed may still have
    // another owner (see "Guarantee on return").
    int established = 1;
    for (Py_ssize_t i = 0; i < n; i++) {
        if (results[i] == _Py_OWNERSHIP_TIMEOUT ||
            results[i] == _Py_OWNERSHIP_CHANGED) {
            established = 0;
            break;
        }
    }

    PyMem_Free(entries);
    PyMem_Free(objs);
    PyMem_Free(res);
    PyMem_Free(reqs);
    return established;
}

#endif  /* Py_GIL_DISABLED */
