//! The disk tier over a device of host bytes: rows and slot parts read back bit for bit, blocks shared and collected.
const std = @import("std");
const spill = @import("spill.zig");
const Spill = spill.Spill;
const Name = spill.Name;

/// A device of host bytes: address x is mem[x - origin]; three planes over a pool of `cap` positions and slot parts.
pub const Fake = struct {
    mem: []u8,
    copies: usize = 0,
    pub const origin: u64 = 1 << 32;
    const cap = 64; // pool positions
    pub const planes = [_]spill.Plane{ .{ .addr = origin, .ratio = 4, .row = 32 }, .{ .addr = origin + 1024, .ratio = 4, .row = 8 }, .{ .addr = origin + 2048, .ratio = 16, .row = 64 } };
    pub const slots: u64 = origin + 4096; // two slot parts of 5000 bytes
    pub const slot_bytes = 5000;

    pub fn dev(f: *Fake) spill.Dev {
        return .{ .ctx = f, .to_host = toHost, .to_device = toDevice, .wait = wait };
    }
    fn of(ctx: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ctx));
    }
    fn toHost(ctx: *anyopaque, dst: []u8, src: u64) anyerror!void {
        const f = of(ctx);
        f.copies += 1;
        @memcpy(dst, f.mem[src - origin ..][0..dst.len]);
    }
    fn toDevice(ctx: *anyopaque, dst: u64, src: []const u8) anyerror!void {
        const f = of(ctx);
        f.copies += 1;
        @memcpy(f.mem[dst - origin ..][0..src.len], src);
    }
    fn wait(_: *anyopaque) anyerror!void {}

    /// Plane bytes of positions [pos, pos + n).
    pub fn rows(f: *Fake, gpa: std.mem.Allocator, pos: usize, n: usize) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (planes) |p| try out.appendSlice(gpa, f.mem[p.addr - origin + pos / p.ratio * p.row ..][0 .. n / p.ratio * p.row]);
        return out.toOwnedSlice(gpa);
    }

    pub fn scramble(f: *Fake, seed: u64) void {
        var r = std.Random.DefaultPrng.init(seed);
        r.random().bytes(f.mem);
    }
};

pub fn scratch(buf: []u8, tag: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "/tmp/tf-dsspill-{s}-{d}", .{ tag, std.c.getpid() });
}

pub fn rmDir(dir: []const u8) void {
    var buf: [512]u8 = undefined;
    const d = std.c.opendir(std.fmt.bufPrintSentinel(&buf, "{s}", .{dir}, 0) catch return) orelse return;
    while (std.c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (name[0] == '.') continue;
        var p: [600]u8 = undefined;
        _ = std.c.unlink(std.fmt.bufPrintSentinel(&p, "{s}/{s}", .{ dir, name }, 0) catch continue);
    }
    _ = std.c.closedir(d);
    _ = std.c.rmdir(std.fmt.bufPrintSentinel(&buf, "{s}", .{dir}, 0) catch return);
}

pub fn count(dir: []const u8, suffix: []const u8) usize {
    var buf: [512]u8 = undefined;
    const d = std.c.opendir(std.fmt.bufPrintSentinel(&buf, "{s}", .{dir}, 0) catch return 0) orelse return 0;
    defer _ = std.c.closedir(d);
    var n: usize = 0;
    while (std.c.readdir(d)) |ent| {
        if (std.mem.endsWith(u8, std.mem.sliceTo(&ent.name, 0), suffix)) n += 1;
    }
    return n;
}

pub const Rig = struct {
    f: Fake,
    stage: []align(spill.page) u8,
    sp: Spill,

    pub fn init(gpa: std.mem.Allocator, mem: []u8, dir: []const u8, stage_pages: usize) !*Rig {
        const r = try gpa.create(Rig);
        r.f = .{ .mem = mem };
        r.stage = try gpa.alignedAlloc(u8, .fromByteUnits(spill.page), stage_pages * spill.page);
        r.sp = try Spill.init(gpa, &Fake.planes, 16, Fake.slot_bytes, r.f.dev(), r.stage);
        r.sp.direct = false;
        try r.sp.open(dir);
        return r;
    }
    pub fn deinit(r: *Rig, gpa: std.mem.Allocator) void {
        r.sp.deinit();
        gpa.free(r.stage);
        gpa.destroy(r);
    }
};

test "a state's rows and slot part come back bit for bit in another extent, after a reopen" {
    const gpa = std.testing.allocator;
    var dbuf: [128]u8 = undefined;
    const dir = try scratch(&dbuf, "back");
    defer rmDir(dir);
    const mem = try gpa.alloc(u8, 16384);
    defer gpa.free(mem);
    var tokens: [40]u32 = undefined;
    for (&tokens, 0..) |*x, i| x.* = @intCast(i * 7 + 1);
    const nm = try spill.names(gpa, &tokens, 32, 16);
    defer gpa.free(nm);
    try std.testing.expectEqual(@as(usize, 2), nm.len);
    var want_rows: []u8 = undefined;
    var want_slot: [Fake.slot_bytes]u8 = undefined;
    {
        const r = try Rig.init(gpa, mem, dir, 2);
        defer r.deinit(gpa);
        r.f.scramble(1);
        want_rows = try r.f.rows(gpa, 16, 32); // the state's extent starts at 16
        @memcpy(&want_slot, mem[Fake.slots - Fake.origin ..][0..Fake.slot_bytes]);
        try r.sp.write(7, 32, nm, 16, Fake.slots);
        try std.testing.expectEqual(@as(usize, 2), r.sp.wrote_blocks);
        try std.testing.expect(r.sp.check(7, 32, nm));
    }
    defer gpa.free(want_rows);
    const r = try Rig.init(gpa, mem, dir, 4); // reopened (the state from its slot file), two blocks a batch
    defer r.deinit(gpa);
    try std.testing.expect(r.sp.check(7, 32, nm));
    r.f.scramble(2);
    try r.sp.read(7, 32, nm, 32, Fake.slots + Fake.slot_bytes); // into the extent at 32, the second slot part
    const got = try r.f.rows(gpa, 32, 32);
    defer gpa.free(got);
    try std.testing.expectEqualSlices(u8, want_rows, got);
    try std.testing.expectEqualSlices(u8, &want_slot, mem[Fake.slots + Fake.slot_bytes - Fake.origin ..][0..Fake.slot_bytes]);
}

test "a later turn writes only its new blocks; blocks go when no state names them" {
    const gpa = std.testing.allocator;
    var dbuf: [128]u8 = undefined;
    const dir = try scratch(&dbuf, "share");
    defer rmDir(dir);
    const mem = try gpa.alloc(u8, 16384);
    defer gpa.free(mem);
    const r = try Rig.init(gpa, mem, dir, 3);
    defer r.deinit(gpa);
    r.f.scramble(3);
    var tokens: [64]u32 = undefined;
    for (&tokens, 0..) |*x, i| x.* = @intCast(i);
    const n1 = try spill.names(gpa, &tokens, 32, 16);
    defer gpa.free(n1);
    const n2 = try spill.names(gpa, &tokens, 48, 16);
    defer gpa.free(n2);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(n1), std.mem.sliceAsBytes(n2[0..2])); // a prefix's names
    try r.sp.write(1, 32, n1, 0, Fake.slots);
    try r.sp.write(2, 48, n2, 0, Fake.slots);
    try std.testing.expectEqual(@as(usize, 1), r.sp.wrote_blocks); // the third block only
    try std.testing.expectEqual(@as(usize, 3), count(dir, ".blk"));
    var other = tokens;
    other[3] += 1; // a different first block: every name differs
    const n3 = try spill.names(gpa, &other, 32, 16);
    defer gpa.free(n3);
    try std.testing.expect(n3[0][0] != n1[0][0] and n3[1][0] != n1[1][0] and n3[1][1] != n1[1][1]);
    r.sp.forget(1);
    try std.testing.expectEqual(@as(usize, 3), count(dir, ".blk"));
    try std.testing.expect(r.sp.check(2, 48, n2));
    r.sp.forget(2);
    try std.testing.expectEqual(@as(usize, 0), count(dir, ".blk"));
    try std.testing.expectEqual(@as(usize, 0), count(dir, ".slot"));
}

test "a damaged block fails its read and is dropped; the states that name it no longer check" {
    const gpa = std.testing.allocator;
    var dbuf: [128]u8 = undefined;
    const dir = try scratch(&dbuf, "damage");
    defer rmDir(dir);
    const mem = try gpa.alloc(u8, 16384);
    defer gpa.free(mem);
    const r = try Rig.init(gpa, mem, dir, 2);
    defer r.deinit(gpa);
    r.f.scramble(4);
    var tokens: [48]u32 = undefined;
    for (&tokens, 0..) |*x, i| x.* = @intCast(i + 100);
    const nm = try spill.names(gpa, &tokens, 48, 16);
    defer gpa.free(nm);
    const short = nm[0..2];
    try r.sp.write(5, 48, nm, 0, Fake.slots);
    try r.sp.write(6, 32, short, 0, Fake.slots);
    var p: [512]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&p, "{s}/{x:0>16}{x:0>16}.blk", .{ dir, nm[1][0], nm[1][1] }, 0);
    const fd = std.c.open(path, .{ .ACCMODE = .WRONLY }, @as(std.c.mode_t, 0));
    try std.testing.expect(fd >= 0);
    const bad = [_]u8{0xff};
    try std.testing.expectEqual(@as(isize, 1), std.c.pwrite(fd, &bad, 1, spill.page + 3));
    _ = std.c.close(fd);
    try std.testing.expectError(error.SpillBadBlock, r.sp.read(5, 48, nm, 0, null));
    try std.testing.expect(!r.sp.check(5, 48, nm) and !r.sp.check(6, 32, short));
    try std.testing.expectError(error.SpillMissing, r.sp.read(6, 32, short, 0, null));
    try r.sp.write(6, 32, short, 0, Fake.slots); // a later keep writes it again
    try std.testing.expect(r.sp.check(6, 32, short));
    try r.sp.read(6, 32, short, 0, Fake.slots);
}

test "keepOnly forgets the states rank 0's index lost; a reopen collects stray blocks and temporaries" {
    const gpa = std.testing.allocator;
    var dbuf: [128]u8 = undefined;
    const dir = try scratch(&dbuf, "sync");
    defer rmDir(dir);
    const mem = try gpa.alloc(u8, 16384);
    defer gpa.free(mem);
    var tokens: [64]u32 = undefined;
    for (&tokens, 0..) |*x, i| x.* = @intCast(i ^ 0x55);
    const a = try spill.names(gpa, &tokens, 16, 16);
    defer gpa.free(a);
    const b = try spill.names(gpa, tokens[16..], 16, 16);
    defer gpa.free(b);
    {
        const r = try Rig.init(gpa, mem, dir, 2);
        defer r.deinit(gpa);
        r.f.scramble(5);
        try r.sp.write(1, 16, a, 0, Fake.slots);
        try r.sp.write(2, 16, b, 16, Fake.slots);
        try r.sp.keepOnly(&.{2});
        try std.testing.expect(!r.sp.has(1) and r.sp.has(2));
        try std.testing.expectEqual(@as(usize, 1), count(dir, ".blk"));
        var p: [512]u8 = undefined;
        for ([_][]const u8{ "0123456789abcdef0123456789abcdef.blk", "0000000000000009.slot.part" }) |name| {
            const fd = std.c.open(try std.fmt.bufPrintSentinel(&p, "{s}/{s}", .{ dir, name }, 0), .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(std.c.mode_t, 0o600));
            _ = std.c.close(fd);
        }
    }
    const r = try Rig.init(gpa, mem, dir, 2);
    defer r.deinit(gpa);
    try std.testing.expect(r.sp.has(2) and !r.sp.has(1));
    try std.testing.expectEqual(@as(usize, 1), count(dir, ".blk"));
    try std.testing.expectEqual(@as(usize, 0), count(dir, ".part"));
    try std.testing.expectEqual(@as(u64, 4096 + 8192 + 2 * 8192), r.sp.diskBytes(32)); // head page, slot pages, blocks
}

test "the tier refuses planes a block does not hold whole rows of, and staging smaller than a block file" {
    const gpa = std.testing.allocator;
    var f: Fake = .{ .mem = &.{} };
    const stage = try gpa.alignedAlloc(u8, .fromByteUnits(spill.page), spill.page);
    defer gpa.free(stage);
    try std.testing.expectError(error.BlockNotRowAligned, Spill.init(gpa, &.{.{ .addr = 0, .ratio = 3, .row = 8 }}, 16, 10, f.dev(), stage));
    try std.testing.expectError(error.StagingTooSmall, Spill.init(gpa, &Fake.planes, 16, 10, f.dev(), stage));
}

test "an identity follows a file's bytes, or its sample, and the names in a directory" {
    const gpa = std.testing.allocator;
    var dbuf: [128]u8 = undefined;
    const dir = try scratch(&dbuf, "ident");
    defer rmDir(dir);
    var mbuf: [128]u8 = undefined;
    _ = std.c.mkdir(try std.fmt.bufPrintSentinel(&mbuf, "{s}", .{dir}, 0), 0o700);
    var p: [512]u8 = undefined;
    const f = try std.fmt.bufPrintSentinel(&p, "{s}/a.cubin", .{dir}, 0);
    const put = struct {
        fn put(path: [:0]const u8, bytes: []const u8) void {
            const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
            _ = std.c.write(fd, bytes.ptr, bytes.len);
            _ = std.c.close(fd);
        }
    }.put;
    put(f, "one");
    const of = struct {
        fn of(a: std.mem.Allocator, d: []const u8, whole: bool) !u64 {
            var id: spill.Identity = .{};
            id.word(2);
            try id.tree(a, d, whole);
            return id.final();
        }
    }.of;
    const h1 = try of(gpa, dir, true);
    try std.testing.expectEqual(h1, try of(gpa, dir, true));
    put(f, "two");
    try std.testing.expect(h1 != try of(gpa, dir, true));
    const s1 = try of(gpa, dir, false);
    put(try std.fmt.bufPrintSentinel(&p, "{s}/b.cubin", .{dir}, 0), "x");
    try std.testing.expect(s1 != try of(gpa, dir, false));
}

test "a follower's directory is made and stamped; other identities go oldest first past the cap" {
    const gpa = std.testing.allocator;
    var rbuf: [128]u8 = undefined;
    const root = try scratch(&rbuf, "prep");
    const old = try spill.prepare(gpa, root, 1, 1 << 30);
    defer gpa.free(old);
    var p: [512]u8 = undefined;
    const fd = std.c.open(try std.fmt.bufPrintSentinel(&p, "{s}/x.blk", .{old}, 0), .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(std.c.mode_t, 0o600));
    _ = std.c.write(fd, "0123456789", 10);
    _ = std.c.close(fd);
    const mid = try spill.prepare(gpa, root, 2, 1 << 30);
    defer gpa.free(mid);
    const now = try spill.prepare(gpa, root, 3, 16 + 8); // room for the stamps, not for identity 1's 10 bytes
    defer gpa.free(now);
    var b: [128]u8 = undefined;
    try std.testing.expect(std.c.opendir(try std.fmt.bufPrintSentinel(&b, "{s}", .{old}, 0)) == null);
    const keep = std.c.opendir(try std.fmt.bufPrintSentinel(&b, "{s}", .{mid}, 0));
    try std.testing.expect(keep != null);
    _ = std.c.closedir(keep.?);
    for ([_][]const u8{ mid, now }) |d| rmDir(d);
    _ = std.c.unlink(try std.fmt.bufPrintSentinel(&p, "{s}/clock", .{root}, 0));
    rmDir(root);
}
