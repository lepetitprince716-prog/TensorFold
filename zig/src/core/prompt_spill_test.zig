//! Spill mode: every kept state on disk under a hashed index record, read back after an eviction or a restart.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const imprint = @import("prompt_imprint.zig");
const t = @import("prompt_cache_test.zig");
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const Store = pc.Store;
const Imprint = imprint.Imprint;
const Fake = t.Fake;
const fresh = t.fresh;

const rules: pc.Rules = .{ .min_prompt = 0, .min_gap = 1 };

fn scratch(buf: []u8, tag: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "/tmp/tf-spill-{s}-{d}", .{ tag, std.c.getpid() });
}

fn spilling(gpa: std.mem.Allocator, f: *Fake, im: *Imprint, budget: u64) Store {
    var s = Store.init(gpa, f.learned(), rules, budget);
    s.imprint = im;
    s.spill = true;
    return s;
}

test "a conversation's kept state goes to disk as a hashed record; after a restart its next turn resumes there, equal to a fresh pass" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "restart");
    defer rmTree(root);
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 9, 9 }; // history 5: not a shared cut, kept and spilled all the same
    {
        var im = try Imprint.open(gpa, root, 4, 1 << 30);
        defer im.deinit();
        var f: Fake = .{ .gpa = gpa };
        var s = spilling(gpa, &f, &im, 1 << 20);
        defer s.deinit();
        try std.testing.expectEqual(fresh(&t1), f.pass(&s, &t1, try s.begin(a, &t1, 5, &.{}, &.{}, null)));
        try std.testing.expectEqual(@as(usize, 1), im.metas.items.len);
        try std.testing.expect(im.metas.items[0].hashed() and im.metas.items[0].read() == 5);
        try std.testing.expect(im.has(Imprint.keyOf(t1[0..5])));
    }
    var im = try Imprint.open(gpa, root, 4, 1 << 30); // a restarted server: the index read back, nothing in memory
    defer im.deinit();
    try std.testing.expect(im.metas.items[0].hashed() and im.metas.items[0].tokens.len == 0);
    var f: Fake = .{ .gpa = gpa };
    var s = spilling(gpa, &f, &im, 1 << 20);
    defer s.deinit();
    const other = [_]u32{ 1, 2, 4, 4, 5, 9, 9, 9 }; // one token differs below the state: its hash does not match
    var p = try s.begin(a, &other, 5, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 0), p.from);
    try std.testing.expectEqual(fresh(&other), f.pass(&s, &other, p));
    const t2 = t1[0..5].* ++ [_]u32{ 9, 7, 7, 6, 6, 9, 9 }; // the next turn, history 10
    p = try s.begin(a, &t2, 10, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 5), p.from);
    try std.testing.expectEqual(fresh(&t2), f.pass(&s, &t2, p));
    try std.testing.expect(im.has(Imprint.keyOf(t2[0..10]))); // its own state spilled too
}

test "a state the memory budget evicted resumes from disk" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "evict");
    defer rmTree(root);
    var im = try Imprint.open(gpa, root, 4, 1 << 30);
    defer im.deinit();
    var f: Fake = .{ .gpa = gpa };
    var s = spilling(gpa, &f, &im, 110); // one state at 5 (105 bytes) fits in memory
    defer s.deinit();
    const ca = [_]u32{ 1, 2, 3, 4, 5, 9, 9 };
    const cb = [_]u32{ 6, 6, 6, 6, 6, 9, 9 };
    try std.testing.expectEqual(fresh(&ca), f.pass(&s, &ca, try s.begin(a, &ca, 5, &.{}, &.{}, null)));
    try std.testing.expectEqual(fresh(&cb), f.pass(&s, &cb, try s.begin(a, &cb, 5, &.{}, &.{}, null)));
    try std.testing.expect(s.find(&ca, &.{}) == null); // gone from memory
    const ca2 = ca[0..5].* ++ [_]u32{ 9, 8, 8, 9, 9 };
    const p = try s.begin(a, &ca2, 9, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 5), p.from);
    try std.testing.expectEqual(fresh(&ca2), f.pass(&s, &ca2, p));
}

test "below min_spill a conversation's state stays in memory only; a shared cut goes to disk at any length" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "min");
    defer rmTree(root);
    var im = try Imprint.open(gpa, root, 4, 1 << 30);
    defer im.deinit();
    var f: Fake = .{ .gpa = gpa };
    var s = spilling(gpa, &f, &im, 1 << 20);
    defer s.deinit();
    s.min_spill = 6;
    const c1 = [_]u32{ 1, 2, 3, 4, 5, 9, 9 };
    try std.testing.expectEqual(fresh(&c1), f.pass(&s, &c1, try s.begin(a, &c1, 5, &.{}, &.{}, null)));
    try std.testing.expectEqual(@as(usize, 0), im.metas.items.len);
    const c2 = [_]u32{ 3, 3, 3, 3, 3, 9, 1, 2, 9 };
    try std.testing.expectEqual(fresh(&c2), f.pass(&s, &c2, try s.begin(a, &c2, 8, &.{3}, &.{}, null)));
    try std.testing.expect(im.has(Imprint.keyOf(c2[0..3])) and im.has(Imprint.keyOf(c2[0..8])));
}

var answer: pc.Snapshots.Known = .keep;
fn knownFn(_: *anyopaque, _: [:0]const u8, _: []const u64) u8 {
    return @intFromEnum(answer);
}

test "adopt: a family that cannot read the index back clears it, one without a disk tier drops the imprint" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "adopt");
    defer rmTree(root);
    var im = try Imprint.open(gpa, root, 4, 1 << 30);
    defer im.deinit();
    var f: Fake = .{ .gpa = gpa };
    var vt = f.learned().vtable.*;
    vt.known = knownFn;
    var s = Store.init(gpa, .{ .ptr = &f, .vtable = &vt }, rules, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    s.spill = true;
    const c1 = [_]u32{ 1, 2, 3, 4, 5, 9, 9 };
    try std.testing.expectEqual(fresh(&c1), f.pass(&s, &c1, try s.begin(a, &c1, 5, &.{}, &.{}, null)));
    answer = .keep;
    try std.testing.expect(try s.adopt());
    try std.testing.expectEqual(@as(usize, 1), im.metas.items.len);
    answer = .clear;
    try std.testing.expect(try s.adopt());
    try std.testing.expectEqual(@as(usize, 0), im.metas.items.len);
    var path: [512]u8 = undefined;
    try std.testing.expect(std.c.unlink(try Fake.file(&path, im.dir, Imprint.keyOf(c1[0..5]))) != 0); // its file went
    answer = .off;
    try std.testing.expect(!try s.adopt());
    try std.testing.expect(s.imprint == null);
}

test "hashed records match only their own prefix, beside token records, across a reopen" {
    const gpa = std.testing.allocator;
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "hashed");
    defer rmTree(root);
    const long = [_]u32{ 5, 4, 3, 2, 1, 0, 7, 7 };
    {
        var im = try Imprint.open(gpa, root, 9, 1 << 30);
        defer im.deinit();
        try im.addHashed(Imprint.keyOf(long[0..6]), 6, long[0..6], &.{}, 10);
        try im.add(Imprint.keyOf(long[0..3]), 3, long[0..3], &.{}, 10);
        try im.addHashed(Imprint.keyOf(long[0..4]), 4, long[0..4], &.{}, 10);
        try im.remove(Imprint.keyOf(long[0..4])); // the index rewritten with both kinds
    }
    var im = try Imprint.open(gpa, root, 9, 1 << 30);
    defer im.deinit();
    try std.testing.expectEqual(@as(usize, 2), im.metas.items.len);
    try std.testing.expectEqual(@as(u32, 6), im.best(&long, &.{}, false, 0).?.at);
    try std.testing.expectEqual(@as(u32, 6), im.best(&long, &.{}, false, 3).?.at);
    const near = [_]u32{ 5, 4, 3, 2, 1, 1, 7, 7 };
    try std.testing.expectEqual(@as(u32, 3), im.best(&near, &.{}, false, 0).?.at);
    try std.testing.expect(im.best(&long, &.{}, false, 6) == null);
}

test "a state on disk too near the prompt's end for the family's tail is passed over" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "tail");
    defer rmTree(root);
    var im = try Imprint.open(gpa, root, 4, 1 << 30);
    defer im.deinit();
    const tailed: pc.Rules = .{ .min_prompt = 0, .min_gap = 1, .tail = 3 };
    const c1 = [_]u32{ 1, 2, 3, 4, 5, 9, 9, 9 }; // history 5: 5 + 3 <= 8, kept
    {
        var f: Fake = .{ .gpa = gpa };
        var s = Store.init(gpa, f.learned(), tailed, 1 << 20);
        defer s.deinit();
        s.imprint = &im;
        s.spill = true;
        try std.testing.expectEqual(fresh(&c1), f.pass(&s, &c1, try s.begin(a, &c1, 5, &.{}, &.{}, null)));
        try std.testing.expect(im.has(Imprint.keyOf(c1[0..5])));
    }
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.learned(), tailed, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    s.spill = true;
    const near = c1[0..5].* ++ [_]u32{ 7, 7 }; // 5 + 3 > 7: a fresh pass
    var p = try s.begin(a, &near, 0, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 0), p.from);
    try std.testing.expectEqual(fresh(&near), f.pass(&s, &near, p));
    const far = c1[0..5].* ++ [_]u32{ 7, 7, 7 }; // 5 + 3 <= 8: from disk
    p = try s.begin(a, &far, 0, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 5), p.from);
    try std.testing.expectEqual(fresh(&far), f.pass(&s, &far, p));
}

/// A family whose spill writes finish when the test says (settled): the writes it queued, those finished, a failure.
const Later = struct {
    keys: [16]u64 = undefined,
    n: usize = 0, // queued
    done: usize = 0, // finished: the first `done` of them
    told: usize = 0, // the store heard of these
    fail_key: ?u64 = null,
    var it: Later = .{};
    var write: *const fn (*anyopaque, pc.Saved, [:0]const u8, u64, []const u32) anyerror!void = undefined;

    fn vtable(f: *Fake) pc.Snapshots.VTable {
        it = .{};
        var vt = f.learned().vtable.*;
        write = vt.write.?;
        vt.write = writeFn;
        vt.settled = settledFn;
        return vt;
    }
    fn writeFn(ptr: *anyopaque, saved: pc.Saved, dir: [:0]const u8, key: u64, tokens: []const u32) anyerror!void {
        try write(ptr, saved, dir, key, tokens);
        it.keys[it.n] = key;
        it.n += 1;
    }
    fn settledFn(_: *anyopaque, keys: []u64, failures: []?anyerror, wait: bool) usize {
        if (wait) it.done = it.n;
        var k: usize = 0;
        while (it.told < it.done and k < keys.len) : (k += 1) {
            keys[k] = it.keys[it.told];
            failures[k] = if (it.fail_key == it.keys[it.told]) error.SpillWrite else null;
            it.told += 1;
        }
        return k;
    }
};

test "a queued spill write is in the index only once it finished: before, a later turn finds nothing on disk" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "queued");
    defer rmTree(root);
    var im = try Imprint.open(gpa, root, 4, 1 << 30);
    defer im.deinit();
    var f: Fake = .{ .gpa = gpa };
    const vt = Later.vtable(&f);
    var s = Store.init(gpa, .{ .ptr = &f, .vtable = &vt }, rules, 110); // one state at 5 (105 bytes) fits in memory
    defer s.deinit();
    s.imprint = &im;
    s.spill = true;
    const ca = [_]u32{ 1, 2, 3, 4, 5, 9, 9 };
    const cb = [_]u32{ 6, 6, 6, 6, 6, 9, 9 };
    try std.testing.expectEqual(fresh(&ca), f.pass(&s, &ca, try s.begin(a, &ca, 5, &.{}, &.{}, null)));
    try std.testing.expectEqual(@as(usize, 1), s.pending.items.len);
    try std.testing.expectEqual(@as(usize, 0), im.metas.items.len); // queued, not listed
    try std.testing.expectEqual(fresh(&cb), f.pass(&s, &cb, try s.begin(a, &cb, 5, &.{}, &.{}, null))); // evicts ca
    try std.testing.expect(s.find(&ca, &.{}) == null);
    const ca2 = ca[0..5].* ++ [_]u32{ 9, 8, 8, 9, 9 };
    var p = try s.begin(a, &ca2, 9, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 0), p.from); // ca's write has not finished: not on disk for the lookup
    try std.testing.expectEqual(fresh(&ca2), f.pass(&s, &ca2, p));
    Later.it.done = 1; // ca's write finished
    const ca3 = ca[0..5].* ++ [_]u32{ 7, 7, 7, 9, 9 };
    p = try s.begin(a, &ca3, 5, &.{}, &.{}, null);
    try std.testing.expectEqual(@as(u32, 5), p.from); // listed at this lookup, read back
    try std.testing.expectEqual(fresh(&ca3), f.pass(&s, &ca3, p));
    try std.testing.expect(im.has(Imprint.keyOf(ca[0..5])) and !im.has(Imprint.keyOf(cb[0..5])));
}

test "the cap counts queued writes; a failed one is noted and never listed; deinit lists the rest" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const root = try scratch(&buf, "qcap");
    defer rmTree(root);
    var im = try Imprint.open(gpa, root, 4, 2 * 105 + 50); // two states at 5 (105 bytes each) under the cap
    defer im.deinit();
    const c1 = [_]u32{ 1, 1, 1, 1, 1, 9, 9 };
    const c2 = [_]u32{ 2, 2, 2, 2, 2, 9, 9 };
    const c3 = [_]u32{ 3, 3, 3, 3, 3, 9, 9 };
    const c4 = [_]u32{ 4, 4, 4, 4, 4, 9, 9 };
    {
        var f: Fake = .{ .gpa = gpa };
        const vt = Later.vtable(&f);
        var s = Store.init(gpa, .{ .ptr = &f, .vtable = &vt }, rules, 1 << 20);
        defer s.deinit();
        s.imprint = &im;
        s.spill = true;
        _ = f.pass(&s, &c1, try s.begin(a, &c1, 5, &.{}, &.{}, null));
        Later.it.done = 1;
        _ = f.pass(&s, &c2, try s.begin(a, &c2, 5, &.{}, &.{}, null)); // lists c1, queues c2
        try std.testing.expectEqual(@as(usize, 1), im.metas.items.len);
        _ = f.pass(&s, &c3, try s.begin(a, &c3, 5, &.{}, &.{}, null)); // c1 listed and c2 queued fill the cap: c1 goes
        try std.testing.expectEqual(@as(usize, 0), im.metas.items.len);
        try std.testing.expectEqual(@as(usize, 2), s.pending.items.len);
        Later.it.fail_key = Imprint.keyOf(c2[0..5]);
        Later.it.done = 3;
        _ = f.pass(&s, &c4, try s.begin(a, &c4, 5, &.{}, &.{}, null)); // c2 failed, c3 listed, c4 queued
        try std.testing.expect(!im.has(Imprint.keyOf(c2[0..5])) and im.has(Imprint.keyOf(c3[0..5])));
        try std.testing.expectEqual(@as(usize, 1), s.pending.items.len);
    } // deinit: c4 waited for and listed
    try std.testing.expect(im.has(Imprint.keyOf(c4[0..5])));
    try std.testing.expect(im.total() <= im.cap);
    var again = try Imprint.open(gpa, root, 4, 1 << 30);
    defer again.deinit();
    try std.testing.expect(again.has(Imprint.keyOf(c3[0..5])) and again.has(Imprint.keyOf(c4[0..5])));
}

/// A family whose spill writes fail when the test says so, at once (the synchronous path: `write` returns the error).
const Fails = struct {
    var fail: bool = false;
    var write: *const fn (*anyopaque, pc.Saved, [:0]const u8, u64, []const u32) anyerror!void = undefined;

    fn vtable(f: *Fake) pc.Snapshots.VTable {
        fail = false;
        var vt = f.learned().vtable.*;
        write = vt.write.?;
        vt.write = writeFn;
        return vt;
    }
    fn writeFn(ptr: *anyopaque, saved: pc.Saved, dir: [:0]const u8, key: u64, tokens: []const u32) anyerror!void {
        if (fail) return error.SpillWrite;
        try write(ptr, saved, dir, key, tokens);
    }
};

test "a failed rewrite of a state the index lists is said as the synchronous path says it; a good one says nothing" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const ca = [_]u32{ 1, 2, 3, 4, 5, 9, 9 };
    const key = Imprint.keyOf(ca[0..5]);
    var said: [200]u8 = undefined;
    var said_len: usize = 0;
    {
        const root = try scratch(&buf, "rewrite-sync"); // the write returns its error at once
        defer rmTree(root);
        var im = try Imprint.open(gpa, root, 4, 1 << 30);
        defer im.deinit();
        var f: Fake = .{ .gpa = gpa };
        const vt = Fails.vtable(&f);
        var s = Store.init(gpa, .{ .ptr = &f, .vtable = &vt }, rules, 1 << 20);
        defer s.deinit();
        s.imprint = &im;
        s.spill = true;
        _ = f.pass(&s, &ca, try s.begin(a, &ca, 5, &.{}, &.{}, null));
        try std.testing.expect(im.has(key));
        s.forget(&ca, 5); // out of memory, still listed
        f.at = 5;
        Fails.fail = true;
        pc.said = .{};
        try std.testing.expect(s.keep(&ca, 5, null, &.{})); // kept again: the known state's write fails
        try std.testing.expectEqual(@as(usize, 1), pc.said.count);
        said_len = pc.said.last().len;
        @memcpy(said[0..said_len], pc.said.last());
    }
    const root = try scratch(&buf, "rewrite-queued"); // the write is queued and fails later
    defer rmTree(root);
    var im = try Imprint.open(gpa, root, 4, 1 << 30);
    defer im.deinit();
    var f: Fake = .{ .gpa = gpa };
    const vt = Later.vtable(&f);
    var s = Store.init(gpa, .{ .ptr = &f, .vtable = &vt }, rules, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    s.spill = true;
    _ = f.pass(&s, &ca, try s.begin(a, &ca, 5, &.{}, &.{}, null));
    Later.it.done = 1;
    s.settle(false);
    try std.testing.expect(im.has(key) and s.pending.items.len == 0);
    s.forget(&ca, 5);
    f.at = 5;
    pc.said = .{};
    try std.testing.expect(s.keep(&ca, 5, null, &.{})); // the rewrite is queued
    try std.testing.expectEqual(@as(usize, 1), s.pending.items.len); // watched
    try std.testing.expectEqual(@as(u64, 0), s.pending.items[0].bytes); // and adds nothing to the cap: the index counts it
    Later.it.fail_key = key;
    Later.it.done = 2;
    s.settle(false);
    try std.testing.expectEqual(@as(usize, 0), s.pending.items.len);
    try std.testing.expectEqual(@as(usize, 1), pc.said.count);
    try std.testing.expectEqualStrings(said[0..said_len], pc.said.last()); // the same line as the synchronous path's
    try std.testing.expectEqualStrings("prompt cache: learning 5 tokens failed (SpillWrite)", pc.said.last());
    try std.testing.expect(im.has(key) and im.metas.items.len == 1);
    Later.it.fail_key = null; // a rewrite that lands says nothing and lists nothing twice
    s.forget(&ca, 5);
    f.at = 5;
    try std.testing.expect(s.keep(&ca, 5, null, &.{}));
    Later.it.done = 3;
    s.settle(false);
    try std.testing.expectEqual(@as(usize, 1), pc.said.count);
    try std.testing.expect(s.pending.items.len == 0 and im.metas.items.len == 1);
}
