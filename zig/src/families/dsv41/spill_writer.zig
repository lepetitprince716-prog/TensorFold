//! Spill writes off the engine thread (TF_DS_SPILL_ASYNC=1): a writer thread copies and writes each queued state.
const std = @import("std");
const spill = @import("spill.zig");
const Spill = spill.Spill;
const Name = spill.Name;

/// What a write waits on before its first device read (the engine's work that made the state), freed once it ran.
pub const Gate = struct {
    ctx: *anyopaque,
    wait: *const fn (ctx: *anyopaque) anyerror!void,
    free: *const fn (ctx: *anyopaque) void,
};

/// Where the writer reads time: nanoseconds on a monotonic clock.
pub const Clock = struct {
    ctx: *anyopaque,
    now: *const fn (ctx: *anyopaque) u64,
};

/// What runs once on the writer thread before its first job (the device context made current there).
pub const Hook = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque) void,
};

/// Writes queued or running at most: past it the engine thread waits for the oldest to run before it queues another.
pub const max_queue = 8;

/// One queued write: a kept state's names (reserved on the tier at its plan), the blocks it writes, its sources.
pub const Job = struct {
    seq: u64 = 0,
    id: u64, // the kept state (Snap.id; on a follower its buffer's) whose extent rows and slot part it reads
    key: u64,
    at: usize,
    names: []Name, // the tier's state table takes them when it lands
    todo: []usize, // the blocks no state named at the plan, in order
    base: usize,
    slot_src: u64,
    dir: [:0]u8,
    gate: ?Gate = null,
    // the writer's, read by the engine thread once the job is done
    result: anyerror!void = {},
    wrote: std.ArrayList(usize) = .empty,
    ms: u64 = 0,
    // the engine thread's
    attach: ?[]Name = null, // rank 0: the names its kept state takes when it lands
    engine_ns: u64 = 0, // the time the engine thread spent queueing it

    /// Frees what the job holds but its names (the tier's once landed or dropped) and its attach names.
    pub fn deinit(j: *Job, gpa: std.mem.Allocator) void {
        if (j.gate) |g| g.free(g.ctx);
        j.wrote.deinit(gpa);
        gpa.free(j.todo);
        gpa.free(j.dir);
        gpa.destroy(j);
    }
};

/// A write of state `key` planned on the engine thread: its names reserved (no forget unlinks them meanwhile).
pub fn plan(sp: *Spill, id: u64, key: u64, at: usize, nm: []const Name, base: usize, slot_src: u64) !*Job {
    const dir = sp.dir orelse return error.SpillClosed;
    if (at / sp.block != nm.len or at % sp.block != 0) return error.NotABlockEnd;
    const gpa = sp.gpa;
    var todo: std.ArrayList(usize) = .empty;
    defer todo.deinit(gpa);
    for (nm, 0..) |n, k| if (!sp.refs.contains(n) or sp.bad.contains(n)) try todo.append(gpa, k);
    const j = try gpa.create(Job);
    errdefer gpa.destroy(j);
    const names = try gpa.dupe(Name, nm);
    errdefer gpa.free(names);
    const d = try gpa.dupeSentinel(u8, dir, 0);
    errdefer gpa.free(d);
    const list = try todo.toOwnedSlice(gpa);
    errdefer gpa.free(list);
    try sp.refs.ensureUnusedCapacity(gpa, @intCast(nm.len));
    for (nm) |n| {
        const r = sp.refs.getOrPutAssumeCapacity(n);
        r.value_ptr.* = if (r.found_existing) r.value_ptr.* + 1 else 1;
    }
    j.* = .{ .id = id, .key = key, .at = at, .names = names, .todo = list, .base = base, .slot_src = slot_src, .dir = d };
    return j;
}

/// A written job's state on the tier (its names already counted); its rewritten blocks are no longer bad.
pub fn landed(sp: *Spill, j: *Job) !void {
    try sp.states.ensureUnusedCapacity(sp.gpa, 1);
    for (j.wrote.items) |k| _ = sp.bad.remove(j.names[k]);
    if (sp.states.fetchRemove(j.key)) |old| sp.release(old.value); // a rewrite of a state the tier had
    sp.states.putAssumeCapacity(j.key, j.names);
}

/// A failed job's reserved names given back (blocks no other state names are deleted).
pub fn dropped(sp: *Spill, j: *Job) void {
    sp.release(j.names);
}

/// The writer thread and its queue. The tier's tables stay the engine thread's: a job only copies and writes files.
pub const Writer = struct {
    gpa: std.mem.Allocator,
    sp: *const Spill, // read for what init fixed only (planes, block, slot bytes, O_DIRECT, the free floor)
    stage: []align(spill.page) u8, // the writer's own pinned staging (the engine thread's loads keep the tier's)
    dev: spill.Dev, // copies on the writer's own stream
    clock: ?Clock = null,
    hook: ?Hook = null,
    lock: std.atomic.Mutex = .unlocked,
    queue: std.ArrayList(*Job) = .empty, // queued, oldest first (under lock)
    stopping: bool = false, // (under lock)
    pushed: u64 = 0, // the engine thread's: the last queued job's seq
    done: std.atomic.Value(u64) = .init(0), // every job up to this seq ran (results and wrote set)
    wake: [2]std.c.fd_t = .{ -1, -1 },
    thread: ?std.Thread = null,
    missing: std.AutoHashMapUnmanaged(Name, void) = .empty, // the thread's: blocks a failed job left unwritten

    pub fn start(gpa: std.mem.Allocator, sp: *const Spill, stage: []align(spill.page) u8, dev: spill.Dev, clock: ?Clock, hook: ?Hook) !*Writer {
        if (stage.len < sp.blockFile() or stage.len % spill.page != 0) return error.StagingTooSmall;
        const w = try gpa.create(Writer);
        errdefer gpa.destroy(w);
        w.* = .{ .gpa = gpa, .sp = sp, .stage = stage, .dev = dev, .clock = clock, .hook = hook };
        if (std.c.pipe(&w.wake) != 0) return error.SpillWriterPipe;
        errdefer for (w.wake) |fd| {
            _ = std.c.close(fd);
        };
        w.thread = try std.Thread.spawn(.{}, loop, .{w});
        return w;
    }

    /// Runs what is queued, stops the thread and frees the writer; the queue's jobs stay the caller's.
    pub fn stop(w: *Writer) void {
        w.acquire();
        w.stopping = true;
        w.release();
        w.poke();
        if (w.thread) |t| t.join();
        for (w.wake) |fd| _ = std.c.close(fd);
        w.queue.deinit(w.gpa);
        w.missing.deinit(w.gpa);
        w.gpa.destroy(w);
    }

    /// The engine thread's: the writes pushed and not yet run (the one running included).
    pub fn queued(w: *const Writer) u64 {
        return w.pushed -| w.done.load(.acquire);
    }

    /// The engine thread's: whether `max_queue` writes are queued or running (the next push is refused until one ran).
    pub fn full(w: *const Writer) bool {
        return w.queued() >= max_queue;
    }

    /// Queues `j` (the engine thread): its seq is the next one; refused while the queue is full (wait for room first).
    pub fn push(w: *Writer, j: *Job) !void {
        if (w.full()) return error.SpillQueueFull;
        j.seq = w.pushed + 1;
        w.acquire();
        defer w.release();
        try w.queue.append(w.gpa, j);
        w.pushed = j.seq;
        w.poke();
    }

    pub fn finished(w: *const Writer, j: *const Job) bool {
        return w.done.load(.acquire) >= j.seq;
    }

    /// Waits until `j` ran, at most `limit_ns` (0: no limit); false when the limit passed first.
    pub fn wait(w: *const Writer, j: *const Job, limit_ns: u64) bool {
        const t0 = monoNs();
        while (!w.finished(j)) {
            if (limit_ns > 0 and monoNs() - t0 >= limit_ns) return false;
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 200 * std.time.ns_per_us };
            _ = std.c.nanosleep(&ts, null);
        }
        return true;
    }

    fn poke(w: *Writer) void {
        const b = [1]u8{1};
        _ = std.c.write(w.wake[1], &b, 1);
    }

    fn acquire(w: *Writer) void {
        while (!w.lock.tryLock()) std.atomic.spinLoopHint();
    }

    fn release(w: *Writer) void {
        w.lock.unlock();
    }

    fn loop(w: *Writer) void {
        if (w.hook) |h| h.run(h.ctx);
        while (true) {
            w.acquire();
            const next: ?*Job = if (w.queue.items.len > 0) w.queue.orderedRemove(0) else null;
            const stopping = w.stopping;
            w.release();
            if (next) |j| {
                w.run(j);
                w.done.store(j.seq, .release);
                continue;
            }
            if (stopping) return;
            var b: [64]u8 = undefined;
            const n = std.c.read(w.wake[0], &b, b.len);
            if (n < 0 and std.c.errno(n) != .INTR) return; // (the pipe is the writer's own: it does not fail)
        }
    }

    fn run(w: *Writer, j: *Job) void {
        const t0 = if (w.clock) |c| c.now(c.ctx) else 0;
        j.result = w.write(j);
        if (w.clock) |c| j.ms = (c.now(c.ctx) - t0) / std.time.ns_per_ms;
    }

    /// The job's blocks (and those of its names a failed job before it left unwritten), its slot file, its directory.
    fn write(w: *Writer, j: *Job) !void {
        const sp = w.sp;
        var todo: std.ArrayList(usize) = .empty;
        defer todo.deinit(w.gpa);
        var next: usize = 0;
        for (j.names, 0..) |n, k| {
            const planned = next < j.todo.len and j.todo[next] == k;
            if (planned) next += 1;
            if (planned or w.missing.contains(n)) try todo.append(w.gpa, k);
        }
        try w.missing.ensureUnusedCapacity(w.gpa, @intCast(todo.items.len));
        for (todo.items) |k| w.missing.putAssumeCapacity(j.names[k], {}); // until written: a later job that names it writes it
        try j.wrote.ensureTotalCapacity(w.gpa, todo.items.len);
        if (spill.freeBytes(j.dir) < sp.diskBytes(j.at) + sp.floor) return error.SpillDiskFull;
        if (j.gate) |g| try g.wait(g.ctx);
        const per = sp.blockFile();
        const fit = w.stage.len / per;
        var i: usize = 0;
        while (i < todo.items.len) : (i += fit) {
            const batch = todo.items[i..@min(todo.items.len, i + fit)];
            for (batch, 0..) |k, b| try sp.pull(w.dev, w.stage[b * per + spill.page ..][0 .. per - spill.page], j.base + k * sp.block);
            try w.dev.wait(w.dev.ctx);
            for (batch, 0..) |k, b| {
                try sp.writeBlock(j.dir, j.names[k], w.stage[b * per ..][0..per]);
                _ = w.missing.remove(j.names[k]);
                j.wrote.appendAssumeCapacity(k);
            }
        }
        try sp.writeSlot(j.dir, w.stage, w.dev, j.key, j.at, j.names, j.slot_src);
        spill.syncDir(j.dir);
    }
};

fn monoNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

test {
    _ = @import("spill_writer_test.zig");
}
