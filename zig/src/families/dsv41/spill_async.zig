//! Spill writes off the engine thread on a rank (TF_DS_SPILL_ASYNC=1): queueing, ending and the waits that guard them.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes.zig");
const model = @import("model.zig");
const spill = @import("spill.zig");
const spill_lanes = @import("spill_lanes.zig");
const writer = @import("spill_writer.zig");
const watchdog = @import("watchdog.zig");
const Lanes = lanes.Lanes;
const Snap = lanes.Snap;
const Tier = spill_lanes.Tier;

/// TF_DS_SPILL_ASYNC=1 on one rank: the writer thread, its staging and stream, the writes not yet ended on the tier.
pub const Background = struct {
    w: *writer.Writer,
    host: cuda.HostBuffer,
    stream: cuda.Stream,
    m: *model.Model,
    limit_ns: u64, // TF_DS_WATCHDOG_S: a wait for the writer this long breaks the pair, as a stalled primitive does
    jobs: std.ArrayList(*writer.Job) = .empty, // queued or run, not yet ended on the tier, oldest first
    keys: std.ArrayList(u64) = .empty, // rank 0: ended writes the prompt cache has not heard of yet, and their errors
    fails: std.ArrayList(?anyerror) = .empty,

    pub fn open(gpa: std.mem.Allocator, m: *model.Model, t: *Tier, staging: usize) !*Background {
        const b = try gpa.create(Background);
        errdefer gpa.destroy(b);
        const limit = (watchdog.limitFromEnv() catch 0) * std.time.ns_per_s;
        b.* = .{ .w = undefined, .host = try cuda.HostBuffer.alloc(m.ctx.d, staging), .stream = undefined, .m = m, .limit_ns = limit };
        errdefer b.host.free();
        if (@intFromPtr(b.host.bytes.ptr) % spill.page != 0) return error.StagingNotAligned;
        b.stream = try cuda.Stream.init(m.ctx.d, true);
        errdefer b.stream.deinit();
        const stage: []align(spill.page) u8 = @alignCast(b.host.bytes[0 .. staging / spill.page * spill.page]);
        b.w = try writer.Writer.start(gpa, &t.sp, stage, b.dev(), .{ .ctx = m, .now = nowFn }, .{ .ctx = m, .run = bindFn });
        return b;
    }

    /// Every queued write ended, the thread stopped (before the tier, its staging or the device memory go).
    pub fn close(b: *Background, gpa: std.mem.Allocator, t: *Tier) void {
        drain(t, null);
        b.w.stop();
        b.jobs.deinit(gpa);
        b.keys.deinit(gpa);
        b.fails.deinit(gpa);
        b.stream.deinit();
        b.host.free();
        gpa.destroy(b);
    }

    /// The writer's copies: on its own stream, so that its waits never wait for the engine's work.
    fn dev(b: *Background) spill.Dev {
        const D = struct {
            fn of(ctx: *anyopaque) *Background {
                return @ptrCast(@alignCast(ctx));
            }
            fn toHost(ctx: *anyopaque, dst: []u8, src: u64) anyerror!void {
                const x = of(ctx);
                try x.m.ctx.d.check(x.m.ctx.d.api.cuMemcpyDtoHAsync_v2(dst.ptr, src, dst.len, x.stream.handle), "cuMemcpyDtoHAsync");
            }
            fn toDevice(_: *anyopaque, _: u64, _: []const u8) anyerror!void {
                return error.WriterReadsNothing;
            }
            fn wait(ctx: *anyopaque) anyerror!void {
                try of(ctx).stream.synchronize();
            }
        };
        return .{ .ctx = b, .to_host = D.toHost, .to_device = D.toDevice, .wait = D.wait };
    }

    fn nowFn(ctx: *anyopaque) u64 {
        const m: *model.Model = @ptrCast(@alignCast(ctx));
        return m.now();
    }

    fn bindFn(ctx: *anyopaque) void {
        const m: *model.Model = @ptrCast(@alignCast(ctx));
        m.ctx.makeCurrent() catch |err| std.log.err("dsv41: the spill writer thread has no device context: {s}", .{@errorName(err)});
    }

    /// The newest write of kept state `id` (rank 0's Snap.id) or of state `key` not yet ended on the tier.
    fn pending(b: *const Background, id: ?u64, key: ?u64) ?*writer.Job {
        var i = b.jobs.items.len;
        while (i > 0) {
            i -= 1;
            const j = b.jobs.items[i];
            if ((id != null and j.id == id.?) or (key != null and j.key == key.?)) return j;
        }
        return null;
    }
};

/// An event on the model's stream after the work that made the state: the writer's first read waits for it.
const Fence = struct {
    ev: cuda.Event,
    gpa: std.mem.Allocator,

    fn gate(gpa: std.mem.Allocator, m: *model.Model) !writer.Gate {
        const f = try gpa.create(Fence);
        errdefer gpa.destroy(f);
        f.* = .{ .ev = try cuda.Event.init(m.ctx.d, false), .gpa = gpa };
        errdefer f.ev.deinit();
        try f.ev.record(m.stream);
        return .{ .ctx = f, .wait = wait, .free = free };
    }
    fn of(ctx: *anyopaque) *Fence {
        return @ptrCast(@alignCast(ctx));
    }
    fn wait(ctx: *anyopaque) anyerror!void {
        try of(ctx).ev.synchronize();
    }
    fn free(ctx: *anyopaque) void {
        const f = of(ctx);
        f.ev.deinit();
        f.gpa.destroy(f);
    }
};

/// The writes the writer ran, oldest first, ended on the tier (and rank 0's kept states get their disk copies).
pub fn reap(t: *Tier, ln: ?*Lanes) void {
    const b = t.bg orelse return;
    while (b.jobs.items.len > 0 and b.w.finished(b.jobs.items[0])) end(t, b, ln, b.jobs.orderedRemove(0));
}

fn end(t: *Tier, b: *Background, ln: ?*Lanes, j: *writer.Job) void {
    const gpa = t.sp.gpa;
    defer j.deinit(gpa);
    var failure: ?anyerror = null;
    if (j.result) |_| {
        writer.landed(&t.sp, j) catch |err| {
            failure = err;
        };
    } else |err| failure = err;
    if (failure != null) writer.dropped(&t.sp, j);
    const names = j.attach orelse { // a follower's write: said only when it failed, as before
        if (failure) |err| std.log.err("dsv41: spilling {d} tokens failed: {s}", .{ j.at, @errorName(err) });
        return;
    };
    const ms = std.time.ns_per_ms;
    if (failure == null) std.log.info("dsv41: spilled the kept state of {d} tokens ({d} new blocks, {d} ms) on the writer thread in {d} ms", .{ j.at, j.wrote.items.len, j.engine_ns / ms, j.ms });
    const sn: ?*Snap = if (failure == null and ln != null) snapOf(ln.?, j.id) else null; // (dropped since: none)
    if (sn) |x| spill_lanes.attach(gpa, x, j.key, names) else gpa.free(names);
    b.keys.appendAssumeCapacity(j.key); // (room made when it was queued)
    b.fails.appendAssumeCapacity(failure);
}

fn snapOf(ln: *Lanes, id: u64) ?*Snap {
    for (ln.snaps.items) |x| if (x.id == id) return x;
    return null;
}

/// Waits for queued write `j` to end; one that runs past the watchdog's limit breaks the pair, as a stall does.
fn waitFor(t: *Tier, j: *writer.Job, why: []const u8) void {
    const b = t.bg.?;
    if (b.w.finished(j)) return;
    const t0 = b.m.now();
    if (!b.w.wait(j, b.limit_ns)) {
        var buf: [96]u8 = undefined;
        lanes.pairBroken(b.m, std.fmt.bufPrint(&buf, "watchdog: spill_writer {d} s", .{b.limit_ns / std.time.ns_per_s}) catch "watchdog: spill_writer");
        _ = b.w.wait(j, 0); // (TF_DS_PAIR_BROKEN=log: the wait goes on)
    }
    std.log.info("dsv41: the engine thread waited {d} ms for the spill write of {d} tokens ({s})", .{ (b.m.now() - t0) / std.time.ns_per_ms, j.at, why });
}

/// Room for one more write: the finished ones ended, and with the writer's queue full the oldest write ends first.
fn makeRoom(t: *Tier, ln: ?*Lanes) void {
    const b = t.bg.?;
    reap(t, ln);
    if (!b.w.full() or b.jobs.items.len == 0) return;
    waitFor(t, b.jobs.items[0], "the writer's queue is full");
    reap(t, ln);
}

/// Every queued write ended on the tier.
pub fn drain(t: *Tier, ln: ?*Lanes) void {
    const b = t.bg orelse return;
    if (b.jobs.items.len > 0) waitFor(t, b.jobs.items[b.jobs.items.len - 1], "every queued write ends");
    reap(t, ln);
}

/// Rank 0, before kept state `sn`'s rows go to another stream: every rank's queued write of it ends first.
pub fn rowsGo(ln: *Lanes, sn: *const Snap) !void {
    const t = ln.spill orelse return;
    const b = t.bg orelse return;
    if (try ln.begin(.spill_wait)) { // (always: a follower's write of it may still run when this rank's has ended)
        try ln.wire.int(ln.gpa, u64, sn.id);
        try ln.flush();
    }
    if (b.pending(sn.id, null)) |j| waitFor(t, j, "its rows go to another stream");
    reap(t, ln);
}

/// Rank 0, before kept state `sn`'s slot part is freed (rank 1 waits at its snap_drop frame).
pub fn dropping(ln: *Lanes, sn: *const Snap) void {
    const t = ln.spill orelse return;
    const b = t.bg orelse return;
    if (b.pending(sn.id, null)) |j| waitFor(t, j, "its kept state is dropped");
    reap(t, ln);
}

/// A follower, before kept state `id`'s buffer is freed or its rows go to another stream.
pub fn followHold(t: ?*Tier, id: u64, why: []const u8) void {
    const x = t orelse return;
    const b = x.bg orelse return;
    if (b.pending(id, null)) |j| waitFor(x, j, why);
    reap(x, null);
}

/// State `key`'s queued write ends before the state is read or forgotten.
pub fn settleKey(t: *Tier, ln: ?*Lanes, key: u64, why: []const u8) void {
    const b = t.bg orelse return;
    reap(t, ln);
    if (b.pending(null, key)) |j| waitFor(t, j, why);
    reap(t, ln);
}

/// Snapshots.settled: the keys of rank 0's queued writes that ended since the last call, with their errors.
pub fn settledFn(ptr: *anyopaque, keys: []u64, fails: []?anyerror, wait: bool) usize {
    const ln = spill_lanes.of(ptr);
    const t = ln.spill orelse return 0;
    const b = t.bg orelse return 0;
    if (b.jobs.items.len > 0) ln.m.ctx.makeCurrent() catch {}; // (an ended job frees its event: a lookup may come first)
    if (wait) drain(t, ln) else reap(t, ln);
    const n = @min(keys.len, b.keys.items.len);
    @memcpy(keys[0..n], b.keys.items[0..n]);
    @memcpy(fails[0..n], b.fails.items[0..n]);
    b.keys.replaceRangeAssumeCapacity(0, n, &.{});
    b.fails.replaceRangeAssumeCapacity(0, n, &.{});
    return n;
}

/// writeFn with TF_DS_SPILL_ASYNC=1: the frame, the plan and an event; the writer thread copies and writes.
pub fn queue(ln: *Lanes, t: *Tier, b: *Background, sn: *Snap, dir: [:0]const u8, key: u64, tokens: []const u32) !void {
    const t0 = ln.m.now();
    if (t.sp.dir) |d| if (!std.mem.eql(u8, d, dir)) drain(t, ln); // (another directory opens with nothing queued)
    try t.sp.open(dir);
    reap(t, ln);
    const nm = try spill.names(ln.gpa, tokens, sn.at, model.chunk_rows);
    errdefer ln.gpa.free(nm);
    try b.keys.ensureUnusedCapacity(ln.gpa, b.jobs.items.len + 1); // a record for every queued write and this one
    try b.fails.ensureUnusedCapacity(ln.gpa, b.jobs.items.len + 1);
    if (t.sp.check(key, sn.at, nm)) { // on disk already
        spill_lanes.attach(ln.gpa, sn, key, nm);
        b.keys.appendAssumeCapacity(key);
        b.fails.appendAssumeCapacity(null);
        return;
    }
    makeRoom(t, ln); // (a full queue waits here, in the prompt pass's watchdog window, where the synchronous write ran)
    try b.jobs.ensureUnusedCapacity(ln.gpa, 1);
    const j = try writer.plan(&t.sp, sn.id, key, sn.at, nm, sn.base, sn.buf.ptr);
    errdefer {
        writer.dropped(&t.sp, j);
        j.deinit(ln.gpa);
    }
    j.gate = try Fence.gate(ln.gpa, ln.m);
    if (try ln.begin(.spill_write)) {
        try ln.wire.int(ln.gpa, u64, sn.id);
        try ln.wire.int(ln.gpa, u64, key);
        try ln.wire.int(ln.gpa, u64, sn.at);
        try ln.wire.int(ln.gpa, u64, sn.base);
        try spill_lanes.names(&ln.wire, ln.gpa, nm);
        try ln.flush();
    }
    try b.w.push(j);
    j.attach = nm; // (the writer never reads it)
    b.jobs.appendAssumeCapacity(j);
    j.engine_ns = ln.m.now() - t0;
}

/// A follower's spill_write with TF_DS_SPILL_ASYNC=1: queued as rank 0 queues its own (said only when it fails).
pub fn followWrite(m: *model.Model, t: *Tier, id: u64, key: u64, at: usize, nm: []const spill.Name, base: usize, src: u64) void {
    const b = t.bg.?;
    reap(t, null);
    if (t.sp.check(key, at, nm)) return; // on disk already
    makeRoom(t, null); // (a full queue waits here, in this frame's watchdog window, where the synchronous write ran)
    queueHere(m, t, b, id, key, at, nm, base, src) catch |err| std.log.err("dsv41: spilling {d} tokens failed: {s}", .{ at, @errorName(err) });
}

fn queueHere(m: *model.Model, t: *Tier, b: *Background, id: u64, key: u64, at: usize, nm: []const spill.Name, base: usize, src: u64) !void {
    try b.jobs.ensureUnusedCapacity(t.sp.gpa, 1);
    const j = try writer.plan(&t.sp, id, key, at, nm, base, src);
    errdefer {
        writer.dropped(&t.sp, j);
        j.deinit(t.sp.gpa);
    }
    j.gate = try Fence.gate(t.sp.gpa, m);
    try b.w.push(j);
    b.jobs.appendAssumeCapacity(j);
}
