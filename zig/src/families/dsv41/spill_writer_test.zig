//! The writer thread over a device of host bytes: order, listing only once written, a pending write's names, failures.
const std = @import("std");
const spill = @import("spill.zig");
const writer = @import("spill_writer.zig");
const st = @import("spill_test.zig");
const Fake = st.Fake;
const Rig = st.Rig;
const Writer = writer.Writer;

/// A gate the test opens by hand (the engine's event, not yet reached), or one that fails (a device error).
const Door = struct {
    open: std.atomic.Value(bool) = .init(false),
    fail: bool = false,
    freed: u32 = 0,

    fn gate(d: *Door) writer.Gate {
        return .{ .ctx = d, .wait = wait, .free = free };
    }
    fn of(ctx: *anyopaque) *Door {
        return @ptrCast(@alignCast(ctx));
    }
    fn wait(ctx: *anyopaque) anyerror!void {
        const d = of(ctx);
        while (!d.open.load(.acquire)) {
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 100 * std.time.ns_per_us };
            _ = std.c.nanosleep(&ts, null);
        }
        if (d.fail) return error.CudaFailed;
    }
    fn free(ctx: *anyopaque) void {
        of(ctx).freed += 1;
    }
};

/// A tier, its writer (own staging) and a device of host bytes.
const Bench = struct {
    rig: *Rig,
    stage: []align(spill.page) u8,
    w: *Writer,

    fn init(gpa: std.mem.Allocator, mem: []u8, dir: []const u8, stage_pages: usize) !Bench {
        const rig = try Rig.init(gpa, mem, dir, stage_pages);
        const stage = try gpa.alignedAlloc(u8, .fromByteUnits(spill.page), stage_pages * spill.page);
        return .{ .rig = rig, .stage = stage, .w = try Writer.start(gpa, &rig.sp, stage, rig.f.dev(), null, null) };
    }
    fn deinit(b: *Bench, gpa: std.mem.Allocator) void {
        b.w.stop();
        gpa.free(b.stage);
        b.rig.deinit(gpa);
    }
    /// The job run, then landed or dropped on the tier as the engine thread does it.
    fn settle(b: *Bench, gpa: std.mem.Allocator, j: *writer.Job) !void {
        try std.testing.expect(b.w.wait(j, 10 * std.time.ns_per_s));
        defer j.deinit(gpa);
        if (j.result) |_| try writer.landed(&b.rig.sp, j) else |_| writer.dropped(&b.rig.sp, j);
    }
};

fn fileBytes(gpa: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    var p: [512]u8 = undefined;
    const fd = std.c.open(try std.fmt.bufPrintSentinel(&p, "{s}/{s}", .{ dir, name }, 0), .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.FileNotFound;
    defer _ = std.c.close(fd);
    var out: std.ArrayList(u8) = .empty;
    var chunk: [4096]u8 = undefined;
    while (true) {
        const n = std.c.read(fd, &chunk, chunk.len);
        if (n <= 0) break;
        try out.appendSlice(gpa, chunk[0..@intCast(n)]);
    }
    return out.toOwnedSlice(gpa);
}

fn sameFiles(gpa: std.mem.Allocator, a: []const u8, b: []const u8, names: []const []const u8) !void {
    for (names) |n| {
        const x = try fileBytes(gpa, a, n);
        defer gpa.free(x);
        const y = try fileBytes(gpa, b, n);
        defer gpa.free(y);
        try std.testing.expectEqualSlices(u8, x, y);
    }
}

fn blockName(buf: []u8, n: spill.Name) ![]const u8 {
    return std.fmt.bufPrint(buf, "{x:0>16}{x:0>16}.blk", .{ n[0], n[1] });
}

test "a queued write reads the rows at its gate, lands only once written, and writes the bytes a direct write does" {
    const gpa = std.testing.allocator;
    var d1: [128]u8 = undefined;
    var d2: [128]u8 = undefined;
    const dir = try st.scratch(&d1, "wq");
    defer st.rmDir(dir);
    const ref = try st.scratch(&d2, "wqref");
    defer st.rmDir(ref);
    const mem = try gpa.alloc(u8, 16384);
    defer gpa.free(mem);
    var tokens: [40]u32 = undefined;
    for (&tokens, 0..) |*x, i| x.* = @intCast(i * 3 + 2);
    const nm = try spill.names(gpa, &tokens, 32, 16);
    defer gpa.free(nm);
    var b = try Bench.init(gpa, mem, dir, 2);
    defer b.deinit(gpa);
    b.rig.f.scramble(11);
    var door: Door = .{};
    const j = try writer.plan(&b.rig.sp, 1, 7, 32, nm, 16, Fake.slots);
    j.gate = door.gate();
    try b.w.push(j);
    b.rig.f.scramble(12); // the engine's work before the gate changes the rows: the write must see these
    const want = try b.rig.f.rows(gpa, 16, 32);
    defer gpa.free(want);
    var want_slot: [Fake.slot_bytes]u8 = undefined;
    @memcpy(&want_slot, mem[Fake.slots - Fake.origin ..][0..Fake.slot_bytes]);
    const ts: std.c.timespec = .{ .sec = 0, .nsec = 20 * std.time.ns_per_ms };
    _ = std.c.nanosleep(&ts, null);
    try std.testing.expect(!b.w.finished(j));
    try std.testing.expect(!b.rig.sp.check(7, 32, nm)); // not on the tier while it waits
    try std.testing.expectEqual(@as(usize, 0), st.count(dir, ".slot"));
    door.open.store(true, .release);
    try b.settle(gpa, j);
    try std.testing.expectEqual(@as(u32, 1), door.freed);
    try std.testing.expect(b.rig.sp.check(7, 32, nm));
    const direct = try Rig.init(gpa, mem, ref, 2); // the same state written on the calling thread, from the same bytes
    defer direct.deinit(gpa);
    try direct.sp.write(7, 32, nm, 16, Fake.slots);
    var n0: [64]u8 = undefined;
    var n1: [64]u8 = undefined;
    try sameFiles(gpa, dir, ref, &.{ try blockName(&n0, nm[0]), try blockName(&n1, nm[1]), "0000000000000007.slot" });
    b.rig.f.scramble(13);
    try b.rig.sp.read(7, 32, nm, 32, Fake.slots + Fake.slot_bytes); // back into the extent at 32, the second slot part
    const got = try b.rig.f.rows(gpa, 32, 32);
    defer gpa.free(got);
    try std.testing.expectEqualSlices(u8, want, got);
    try std.testing.expectEqualSlices(u8, &want_slot, mem[Fake.slots + Fake.slot_bytes - Fake.origin ..][0..Fake.slot_bytes]);
}

test "jobs run in the order they were queued" {
    const gpa = std.testing.allocator;
    var d1: [128]u8 = undefined;
    const dir = try st.scratch(&d1, "order");
    defer st.rmDir(dir);
    const mem = try gpa.alloc(u8, 16384);
    defer gpa.free(mem);
    var b = try Bench.init(gpa, mem, dir, 2);
    defer b.deinit(gpa);
    b.rig.f.scramble(21);
    var tokens: [64]u32 = undefined;
    for (&tokens, 0..) |*x, i| x.* = @intCast(i + 9);
    var doors: [3]Door = .{ .{}, .{}, .{} };
    var jobs: [3]*writer.Job = undefined;
    for (&jobs, 0..) |*j, i| {
        const nm = try spill.names(gpa, &tokens, 16 * (i + 1), 16);
        defer gpa.free(nm);
        j.* = try writer.plan(&b.rig.sp, i, 100 + i, 16 * (i + 1), nm, 0, Fake.slots);
        j.*.gate = doors[i].gate();
        try b.w.push(j.*);
    }
    try std.testing.expect(jobs[0].seq < jobs[1].seq and jobs[1].seq < jobs[2].seq);
    doors[2].open.store(true, .release); // the last one's gate first: it still waits for the two before it
    doors[1].open.store(true, .release);
    const ts: std.c.timespec = .{ .sec = 0, .nsec = 20 * std.time.ns_per_ms };
    _ = std.c.nanosleep(&ts, null);
    for (jobs) |j| try std.testing.expect(!b.w.finished(j));
    doors[0].open.store(true, .release);
    try std.testing.expect(b.w.wait(jobs[2], 10 * std.time.ns_per_s));
    for (jobs, 0..) |j, i| {
        try std.testing.expect(b.w.finished(j));
        try std.testing.expectEqualSlices(usize, &.{i}, j.wrote.items); // each its own new block: the rest were reserved
    }
    for (jobs) |j| try b.settle(gpa, j);
    try std.testing.expectEqual(@as(usize, 3), st.count(dir, ".blk"));
    for (0..3) |i| try std.testing.expect(b.rig.sp.has(100 + i));
}

test "a forget while a write is queued keeps the blocks the write names; the write then lands whole" {
    const gpa = std.testing.allocator;
    var d1: [128]u8 = undefined;
    const dir = try st.scratch(&d1, "pend");
    defer st.rmDir(dir);
    const mem = try gpa.alloc(u8, 16384);
    defer gpa.free(mem);
    var b = try Bench.init(gpa, mem, dir, 2);
    defer b.deinit(gpa);
    b.rig.f.scramble(31);
    var tokens: [64]u32 = undefined;
    for (&tokens, 0..) |*x, i| x.* = @intCast(i * 5);
    const short = try spill.names(gpa, &tokens, 32, 16);
    defer gpa.free(short);
    const long = try spill.names(gpa, &tokens, 48, 16);
    defer gpa.free(long);
    try b.rig.sp.write(1, 32, short, 0, Fake.slots); // on disk: blocks 0 and 1
    var door: Door = .{};
    const j = try writer.plan(&b.rig.sp, 2, 2, 48, long, 0, Fake.slots);
    j.gate = door.gate();
    try std.testing.expectEqual(@as(usize, 1), j.todo.len); // block 2 only
    try b.w.push(j);
    b.rig.sp.forget(1); // the cap takes state 1 while state 2's write waits
    try std.testing.expectEqual(@as(usize, 2), st.count(dir, ".blk")); // its blocks stay: the queued write names them
    door.open.store(true, .release);
    try b.settle(gpa, j);
    try std.testing.expect(b.rig.sp.check(2, 48, long));
    try b.rig.sp.read(2, 48, long, 0, Fake.slots + Fake.slot_bytes);
    b.rig.sp.forget(2);
    try std.testing.expectEqual(@as(usize, 0), st.count(dir, ".blk"));
}

test "a failed write gives its names back, lists nothing, and a later write that skipped its blocks writes them" {
    const gpa = std.testing.allocator;
    var d1: [128]u8 = undefined;
    const dir = try st.scratch(&d1, "fail");
    defer st.rmDir(dir);
    const mem = try gpa.alloc(u8, 16384);
    defer gpa.free(mem);
    var b = try Bench.init(gpa, mem, dir, 2);
    defer b.deinit(gpa);
    b.rig.f.scramble(41);
    var tokens: [64]u32 = undefined;
    for (&tokens, 0..) |*x, i| x.* = @intCast(i * 11 + 1);
    const short = try spill.names(gpa, &tokens, 32, 16);
    defer gpa.free(short);
    const long = try spill.names(gpa, &tokens, 48, 16);
    defer gpa.free(long);
    var bad: Door = .{ .fail = true };
    var good: Door = .{};
    const j1 = try writer.plan(&b.rig.sp, 1, 1, 32, short, 0, Fake.slots);
    j1.gate = bad.gate();
    try b.w.push(j1);
    const j2 = try writer.plan(&b.rig.sp, 2, 2, 48, long, 0, Fake.slots); // planned while j1 holds blocks 0 and 1
    j2.gate = good.gate();
    try std.testing.expectEqual(@as(usize, 1), j2.todo.len);
    try b.w.push(j2);
    bad.open.store(true, .release);
    try std.testing.expect(b.w.wait(j1, 10 * std.time.ns_per_s));
    try std.testing.expectError(error.CudaFailed, j1.result);
    try b.settle(gpa, j1);
    try std.testing.expect(!b.rig.sp.has(1));
    try std.testing.expectEqual(@as(usize, 0), st.count(dir, ".slot"));
    good.open.store(true, .release);
    try b.settle(gpa, j2);
    try std.testing.expect(b.rig.sp.check(2, 48, long));
    try std.testing.expectEqual(@as(usize, 3), st.count(dir, ".blk")); // the blocks j1 left unwritten: j2 wrote them
    try b.rig.sp.read(2, 48, long, 0, Fake.slots + Fake.slot_bytes);
    b.rig.sp.forget(2);
    try std.testing.expectEqual(@as(u32, 0), b.rig.sp.refs.count()); // every reservation given back
}

test "a write past the free floor fails before any file; the next one writes the blocks it skipped for it" {
    const gpa = std.testing.allocator;
    var d1: [128]u8 = undefined;
    const dir = try st.scratch(&d1, "full");
    defer st.rmDir(dir);
    const mem = try gpa.alloc(u8, 16384);
    defer gpa.free(mem);
    var b = try Bench.init(gpa, mem, dir, 2);
    defer b.deinit(gpa);
    b.rig.f.scramble(51);
    var tokens: [48]u32 = undefined;
    for (&tokens, 0..) |*x, i| x.* = @intCast(i);
    const short = try spill.names(gpa, &tokens, 32, 16);
    defer gpa.free(short);
    const long = try spill.names(gpa, &tokens, 48, 16);
    defer gpa.free(long);
    const j1 = try writer.plan(&b.rig.sp, 1, 1, 32, short, 0, Fake.slots);
    const j2 = try writer.plan(&b.rig.sp, 2, 2, 48, long, 0, Fake.slots); // block 2 only: j1 holds 0 and 1
    try std.testing.expectEqual(@as(usize, 1), j2.todo.len);
    const floor = b.rig.sp.floor;
    b.rig.sp.floor = std.math.maxInt(u64) / 2; // more than any disk has free (off Linux: than the unbounded figure)
    try b.w.push(j1);
    try std.testing.expect(b.w.wait(j1, 10 * std.time.ns_per_s));
    try std.testing.expectError(error.SpillDiskFull, j1.result);
    try std.testing.expectEqual(@as(usize, 0), st.count(dir, ".blk"));
    try b.settle(gpa, j1);
    b.rig.sp.floor = floor;
    try b.w.push(j2);
    try b.settle(gpa, j2);
    try std.testing.expect(b.rig.sp.check(2, 48, long));
    try std.testing.expectEqual(@as(usize, 3), st.count(dir, ".blk"));
    b.rig.sp.forget(2);
    try std.testing.expectEqual(@as(u32, 0), b.rig.sp.refs.count());
}

test "stop runs the queued jobs before the thread ends" {
    const gpa = std.testing.allocator;
    var d1: [128]u8 = undefined;
    const dir = try st.scratch(&d1, "stop");
    defer st.rmDir(dir);
    const mem = try gpa.alloc(u8, 16384);
    defer gpa.free(mem);
    const rig = try Rig.init(gpa, mem, dir, 2);
    defer rig.deinit(gpa);
    rig.f.scramble(61);
    const stage = try gpa.alignedAlloc(u8, .fromByteUnits(spill.page), 2 * spill.page);
    defer gpa.free(stage);
    const w = try Writer.start(gpa, &rig.sp, stage, rig.f.dev(), null, null);
    var tokens: [32]u32 = undefined;
    for (&tokens, 0..) |*x, i| x.* = @intCast(i + 77);
    const nm = try spill.names(gpa, &tokens, 32, 16);
    defer gpa.free(nm);
    const j = try writer.plan(&rig.sp, 1, 3, 32, nm, 0, Fake.slots);
    defer j.deinit(gpa);
    try w.push(j);
    w.stop();
    try j.result;
    try writer.landed(&rig.sp, j);
    try std.testing.expect(rig.sp.check(3, 32, nm));
}

test "the queue holds max_queue writes; one more is refused until the oldest ran; a stuck writer leaves a wait to time out" {
    const gpa = std.testing.allocator;
    var d1: [128]u8 = undefined;
    const dir = try st.scratch(&d1, "bound");
    defer st.rmDir(dir);
    const mem = try gpa.alloc(u8, 16384);
    defer gpa.free(mem);
    var b = try Bench.init(gpa, mem, dir, 2);
    defer b.deinit(gpa);
    b.rig.f.scramble(71);
    try std.testing.expectEqual(@as(usize, 8), writer.max_queue);
    var doors: [writer.max_queue + 1]Door = @splat(.{});
    defer for (&doors) |*d| d.open.store(true, .release); // (a failed check must not leave the writer on a closed gate)
    var jobs: [writer.max_queue + 1]*writer.Job = undefined;
    for (&jobs, 0..) |*j, i| { // each a block of its own
        var tokens: [16]u32 = undefined;
        for (&tokens, 0..) |*x, k| x.* = @intCast(1000 * (i + 1) + k);
        const nm = try spill.names(gpa, &tokens, 16, 16);
        defer gpa.free(nm);
        j.* = try writer.plan(&b.rig.sp, i, 200 + i, 16, nm, 0, Fake.slots);
        j.*.gate = doors[i].gate();
    }
    for (jobs[0 .. writer.max_queue - 1]) |j| try b.w.push(j);
    try std.testing.expect(!b.w.full()); // one short of the bound
    try b.w.push(jobs[writer.max_queue - 1]);
    try std.testing.expect(b.w.full());
    try std.testing.expectEqual(@as(u64, writer.max_queue), b.w.queued());
    try std.testing.expectError(error.SpillQueueFull, b.w.push(jobs[writer.max_queue])); // the ninth waits for room
    try std.testing.expectEqual(@as(u64, writer.max_queue), b.w.queued()); // and left nothing queued
    try std.testing.expect(!b.w.wait(jobs[0], 30 * std.time.ns_per_ms)); // the writer is stuck on the first gate: the limit passes
    try std.testing.expect(b.w.full());
    doors[0].open.store(true, .release); // the oldest runs (the next gate holds the writer again)
    try std.testing.expect(b.w.wait(jobs[0], 10 * std.time.ns_per_s));
    try std.testing.expect(!b.w.full());
    try b.w.push(jobs[writer.max_queue]);
    try std.testing.expect(b.w.full());
    for (&doors) |*d| d.open.store(true, .release);
    for (jobs) |j| try b.settle(gpa, j);
    try std.testing.expectEqual(@as(u64, 0), b.w.queued());
    for (0..jobs.len) |i| try std.testing.expect(b.rig.sp.has(200 + i));
}
