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
