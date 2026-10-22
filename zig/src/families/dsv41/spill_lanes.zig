//! The prompt cache's spill mode on the lanes: rank 0's Snapshots functions, a pass's disk load, the followers' frames.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes.zig");
const model = @import("model.zig");
const prompt = @import("prompt.zig");
const spill = @import("spill.zig");
const link = @import("link.zig");
const bgw = @import("spill_async.zig");
const writer = @import("spill_writer.zig");
const Lanes = lanes.Lanes;
const Snap = lanes.Snap;
const Reader = lanes.Reader;

/// What the environment asks of a rank's tier (TF_DS_SPILL_DIR on every rank; off when unset or empty).
pub const Settings = struct {
    root: []const u8,
    cap: u64, // TF_DS_SPILL_GIB (default 200): every identity's files under `root` together
    min_tokens: u32, // TF_DS_SPILL_MIN_TOKENS (default 8192): shorter conversation states stay in memory
    staging: usize, // TF_DS_SPILL_STAGING_MIB (default 32): pinned host memory for the copies
    background: bool = false, // TF_DS_SPILL_ASYNC=1: writes on a writer thread with staging of its own (default off)

    pub fn read() !?Settings {
        const root = env("TF_DS_SPILL_DIR") orelse return null;
        if (root.len == 0) return null;
        const gib = if (env("TF_DS_SPILL_GIB")) |v| std.fmt.parseFloat(f64, v) catch return error.BadSpillGib else 200;
        const min = if (env("TF_DS_SPILL_MIN_TOKENS")) |v| std.fmt.parseInt(u32, v, 10) catch return error.BadSpillMin else 8192;
        const mib = if (env("TF_DS_SPILL_STAGING_MIB")) |v| std.fmt.parseInt(usize, v, 10) catch return error.BadSpillStaging else 32;
        if (gib <= 0 or mib == 0) return null;
        const knob = env("TF_DS_SPILL_ASYNC");
        const bg = writer.knobOn(knob);
        if (knob) |v| std.log.info("dsv41: TF_DS_SPILL_ASYNC={s} is read as {s} (only 1 runs spill writes on a writer thread)", .{ v, onOff(bg) });
        return .{ .root = root, .cap = @intFromFloat(gib * (1 << 30)), .min_tokens = min, .staging = mib << 20, .background = bg };
    }

    fn env(name: [:0]const u8) ?[]const u8 {
        return std.mem.span(std.c.getenv(name) orelse return null);
    }
};

fn onOff(on: bool) []const u8 {
    return if (on) "on" else "off";
}

/// One rank's tier: its files (spill.zig) through its pinned staging, on its model's stream.
pub const Tier = struct {
    sp: spill.Spill,
    host: cuda.HostBuffer,
    settings: Settings,
    identity: u64,
    bg: ?*bgw.Background = null, // TF_DS_SPILL_ASYNC=1 (null: the engine thread writes, as before)
    off: bool = false, // a follower's: the ranks differ on TF_DS_SPILL_ASYNC (spill_sync), so it answers as if it had no tier

    pub fn open(gpa: std.mem.Allocator, m: *model.Model, s: Settings, id: u64) !*Tier {
        const t = try gpa.create(Tier);
        errdefer gpa.destroy(t);
        t.settings = s;
        t.identity = id;
        t.host = try cuda.HostBuffer.alloc(m.ctx.d, s.staging);
        errdefer t.host.free();
        if (@intFromPtr(t.host.bytes.ptr) % spill.page != 0) return error.StagingNotAligned;
        const stage: []align(spill.page) u8 = @alignCast(t.host.bytes[0 .. s.staging / spill.page * spill.page]);
        const planes = try m.spillPlanes(gpa);
        defer gpa.free(planes);
        t.sp = try spill.Spill.init(gpa, planes, model.chunk_rows, m.snapBytes(), m.spillDev(), stage);
        errdefer t.sp.deinit();
        std.log.info("dsv41: kept prompt states spill to {s} ({d} MiB a block of {d} tokens, {d} MiB a slot part)", .{ s.root, t.sp.blockBytes() >> 20, model.chunk_rows, m.snapBytes() >> 20 });
        t.bg = null;
        t.off = false;
        if (s.background) {
            t.bg = try bgw.Background.open(gpa, m, t, s.staging);
            std.log.info("dsv41: spill writes run on a writer thread (TF_DS_SPILL_ASYNC=1)", .{});
        }
        return t;
    }

    pub fn close(t: *Tier, gpa: std.mem.Allocator) void {
        if (t.bg) |b| b.close(gpa, t);
        t.sp.deinit();
        t.host.free();
        gpa.destroy(t);
    }
};

/// What sets the bits of this rank's kept states: binary, kernels, checkpoint, weights, tables, the pass's rules.
pub fn identity(gpa: std.mem.Allocator, m: *const model.Model, o: model.Options) !u64 {
    var id: spill.Identity = .{};
    for ([_]u64{ 1, m.rank, m.world, model.chunk_rows, @intFromBool(prompt.replay_floor), m.cfg.window, m.snapBytes(), @intFromBool(m.drafting()) }) |w| id.word(w);
    id.content("/proc/self/exe");
    var buf: [1200]u8 = undefined;
    id.content(try std.fmt.bufPrint(&buf, "{s}/config.json", .{o.model_dir}));
    id.content(try std.fmt.bufPrint(&buf, "{s}/engram.json", .{o.kit_dir}));
    if (o.token_map) |p| id.content(p);
    try id.tree(gpa, try std.fmt.bufPrint(&buf, "{s}/cubins", .{o.kit_dir}), true);
    try id.tree(gpa, o.aot_dir orelse try std.fmt.bufPrint(&buf, "{s}/aot", .{o.kit_dir}), true);
    try id.tree(gpa, o.kit_dir, false); // the RoPE tables
    try id.tree(gpa, o.cache_dir, false); // the rank caches
    if (o.engram_dir) |d| try id.tree(gpa, d, false);
    if (o.vision_bias) |p| id.sample(p);
    return id.final();
}

pub fn of(ptr: *anyopaque) *Lanes {
    return @ptrCast(@alignCast(ptr));
}

pub fn names(w: *lanes.Wire, gpa: std.mem.Allocator, nm: []const spill.Name) !void {
    try w.int(gpa, u32, @intCast(nm.len));
    for (nm) |n| {
        try w.int(gpa, u64, n[0]);
        try w.int(gpa, u64, n[1]);
    }
}

fn readNames(r: *Reader, gpa: std.mem.Allocator) ![]spill.Name {
    const n = try r.int(u32);
    if (n > 1 << 20) return error.BadFrame;
    const out = try gpa.alloc(spill.Name, n);
    errdefer gpa.free(out);
    for (out) |*x| x.* = .{ try r.int(u64), try r.int(u64) };
    return out;
}

/// A kept state's disk copy noted (its names now the snap's).
pub fn attach(gpa: std.mem.Allocator, sn: *Snap, key: u64, nm: []spill.Name) void {
    if (sn.disk) |d| gpa.free(d.names);
    sn.disk = .{ .key = key, .names = nm };
}

/// Snapshots.write: a kept state's new blocks and slot part to this rank's disk, the followers told to do the same.
pub fn writeFn(ptr: *anyopaque, saved: *anyopaque, dir: [:0]const u8, key: u64, tokens: []const u32) anyerror!void {
    const ln = of(ptr);
    const t = ln.spill orelse return error.NoSpill;
    const sn: *Snap = @ptrCast(@alignCast(saved));
    if (sn.stale or !sn.slot) return error.RowsGone; // (the store writes right after the save)
    if (t.bg) |b| return bgw.queue(ln, t, b, sn, dir, key, tokens);
    try t.sp.open(dir);
    const nm = try spill.names(ln.gpa, tokens, sn.at, model.chunk_rows);
    errdefer ln.gpa.free(nm);
    if (!t.sp.check(key, sn.at, nm)) {
        if (try ln.begin(.spill_write)) {
            try ln.wire.int(ln.gpa, u64, sn.id);
            try ln.wire.int(ln.gpa, u64, key);
            try ln.wire.int(ln.gpa, u64, sn.at);
            try ln.wire.int(ln.gpa, u64, sn.base);
            try names(&ln.wire, ln.gpa, nm);
            try ln.flush();
        }
        const t0 = ln.m.now();
        try t.sp.write(key, sn.at, nm, sn.base, sn.buf.ptr);
        std.log.info("dsv41: spilled the kept state of {d} tokens ({d} new blocks, {d} ms)", .{ sn.at, t.sp.wrote_blocks, (ln.m.now() - t0) / std.time.ns_per_ms });
    }
    attach(ln.gpa, sn, key, nm);
}

/// Snapshots.read: a state on disk as a kept state whose rows and slot part a prompt pass loads (load) when it resumes.
pub fn readFn(ptr: *anyopaque, dir: [:0]const u8, key: u64, at: u32, tokens: []const u32) anyerror!*anyopaque {
    const ln = of(ptr);
    const t = ln.spill orelse return error.NoSpill;
    try ln.m.ctx.makeCurrent(); // (the lane thread may look up before its first backend call)
    try t.sp.open(dir);
    if (t.bg != null) bgw.settleKey(t, ln, key, "it is read back");
    if (at % model.chunk_rows != 0) return error.NotABlockEnd;
    const nm = try spill.names(ln.gpa, tokens, at, model.chunk_rows);
    errdefer ln.gpa.free(nm);
    if (!t.sp.check(key, at, nm)) return error.SpillMissing;
    const sn = try ln.gpa.create(Snap);
    errdefer ln.gpa.destroy(sn);
    sn.* = .{ .id = ln.next_snap, .at = at, .base = 0, .buf = try cuda.DeviceBuffer.alloc(ln.m.ctx.d, ln.m.snapBytes()), .stale = true, .slot = false, .disk = .{ .key = key, .names = nm } };
    errdefer sn.buf.free();
    try ln.snaps.append(ln.gpa, sn);
    ln.next_snap += 1;
    return sn;
}

/// Snapshots.forget: state `key` gone from every rank's disk.
pub fn forgetFn(ptr: *anyopaque, dir: [:0]const u8, key: u64) void {
    const ln = of(ptr);
    const t = ln.spill orelse return;
    t.sp.open(dir) catch return;
    forget(ln, t, key);
}

fn forget(ln: *Lanes, t: *Tier, key: u64) void {
    if (t.bg != null) bgw.settleKey(t, ln, key, "it is forgotten");
    t.sp.forget(key);
    for (ln.snaps.items) |sn| if (sn.disk) |d| if (d.key == key) {
        ln.gpa.free(d.names);
        sn.disk = null;
    };
    if (ln.begin(.spill_forget) catch false) {
        ln.wire.int(ln.gpa, u64, key) catch {};
        ln.flush() catch |err| std.log.err("dsv41: rank 1 kept a forgotten spilled state: {s}", .{@errorName(err)});
    }
}

pub fn diskBytesFn(ptr: *anyopaque, at: u32) u64 {
    const t = of(ptr).spill orelse return 0;
    return t.sp.diskBytes(at);
}

/// Snapshots.known: every rank keeps the index's states only; changed followers clear them, one without a tier: off.
pub fn knownFn(ptr: *anyopaque, dir: [:0]const u8, keys: []const u64) u8 {
    const ln = of(ptr);
    const t = ln.spill orelse return 2;
    t.sp.open(dir) catch return 2;
    bgw.drain(t, ln);
    t.sp.keepOnly(keys) catch {};
    var peers = std.hash.Wyhash.init(0);
    var differs = false;
    ln.busyStartup("spill_sync"); // waits on every follower, which may still be loading: TF_DS_WATCHDOG_START_S, not _S
    defer ln.idle();
    if (ln.begin(.spill_sync) catch return 2) {
        writer.putSync(&ln.wire, ln.gpa, keys, t.bg != null) catch return 2;
        ln.flush() catch return 2;
        for (ln.m.followers(), 1..) |l, rank| {
            var buf: [16]u8 = undefined;
            var r: Reader = .{ .b = l.?.recv(&buf) catch return 2 };
            const on = r.int(u8) catch 0;
            const id = r.int(u64) catch 0;
            switch (writer.answerOf(on)) {
                .no_tier => {
                    std.log.err("dsv41: rank {d} has no TF_DS_SPILL_DIR: kept prompt states stay in memory", .{rank});
                    return 2;
                },
                .differs => { // (every follower's answer is read: none is left on its link)
                    std.log.err("dsv41: TF_DS_SPILL_ASYNC differs: rank 0 {s}, rank {d} {s}; the disk tier is off on both ranks", .{ onOff(t.bg != null), rank, onOff(t.bg == null) });
                    differs = true;
                },
                .same => {},
            }
            peers.update(std.mem.asBytes(&id));
        }
    }
    if (differs) { // the writes of one rank would race the other's rows: no disk tier is the exact outcome
        ln.spill = null;
        return 2;
    }
    const now = peers.final();
    var path: [1200]u8 = undefined;
    const file = std.fmt.bufPrintSentinel(&path, "{s}/peers", .{dir}, 0) catch return 2;
    const before = readWord(file);
    if (before == now) return 0;
    writeWord(file, now);
    return if (keys.len == 0) 0 else 1; // states written beside other followers' files: none reads back
}

/// A pass resumes `sn` from every rank's disk into the extent at `base`; a failure forgets it everywhere (from 0).
pub fn load(ln: *Lanes, sn: *Snap, base: usize) !void {
    const t = ln.spill orelse return error.NoSpill;
    const d = sn.disk orelse return error.NoDiskCopy;
    bgw.settleKey(t, ln, d.key, "a pass loads it");
    if (try ln.begin(.spill_load)) {
        try ln.wire.int(ln.gpa, u64, sn.id);
        try ln.wire.int(ln.gpa, u64, d.key);
        try ln.wire.int(ln.gpa, u64, sn.at);
        try ln.wire.int(ln.gpa, u64, base);
        try ln.wire.int(ln.gpa, u8, @intFromBool(!sn.slot));
        try names(&ln.wire, ln.gpa, d.names);
        try ln.flush();
    }
    const t0 = ln.m.now();
    const own = t.sp.read(d.key, sn.at, d.names, base, if (sn.slot) null else sn.buf.ptr);
    var peers = true;
    if (ln.peer != null) for (ln.m.followers()) |l| {
        var buf: [8]u8 = undefined;
        const b = try l.?.recv(&buf);
        peers = peers and b.len == 1 and b[0] == 1;
    };
    if (own) |_| {
        if (peers) {
            sn.slot = true;
            std.log.info("dsv41: resumed {d} tokens from disk ({d} MiB a rank in {d} ms)", .{ sn.at, t.sp.read_bytes >> 20, (ln.m.now() - t0) / std.time.ns_per_ms });
            return;
        }
    } else |_| {}
    std.log.warn("dsv41: the spilled state of {d} tokens did not read back ({s}); forgotten, the prompt runs from 0", .{ sn.at, if (own) |_| "a follower's copy" else |err| @errorName(err) });
    forget(ln, t, d.key);
    return error.SpillLoad;
}

/// A follower's spill frame (`kind`), its kept states' buffers in `kept`, answers on `peer`.
pub fn follow(gpa: std.mem.Allocator, m: *model.Model, tier: ?*Tier, kind: lanes.Kind, r: *Reader, kept: *std.AutoHashMapUnmanaged(u64, cuda.DeviceBuffer), peer: link.Link) !void {
    const t: ?*Tier = if (tier) |x| (if (x.off) null else x) else null; // (a tier turned off answers as none does)
    switch (kind) {
        .spill_sync => {
            const n = try r.int(u32);
            const keys = try gpa.alloc(u64, n);
            defer gpa.free(keys);
            for (keys) |*k| k.* = try r.int(u64);
            const theirs = writer.readMode(r);
            const mine = if (t) |x| x.bg != null else false;
            const say = writer.answer(t != null, mine, theirs);
            var w: lanes.Wire = .{};
            defer w.deinit(gpa);
            try writer.putAnswer(&w, gpa, say, if (t) |x| x.identity else 0);
            if (say == .differs) {
                if (!@import("builtin").is_test) std.log.err("dsv41: TF_DS_SPILL_ASYNC differs: rank 0 {s}, this rank {s}; the disk tier is off on both ranks", .{ onOff(theirs), onOff(mine) });
                tier.?.off = true;
            }
            if (t) |x| bgw.drain(x, null);
            if (t) |x| x.sp.keepOnly(keys) catch |err| std.log.err("dsv41: spill sync: {s}", .{@errorName(err)});
            try peer.send(w.buf.items);
        },
        .spill_write => {
            const id = try r.int(u64);
            const key = try r.int(u64);
            const at: usize = @intCast(try r.int(u64));
            const base: usize = @intCast(try r.int(u64));
            const nm = try readNames(r, gpa);
            defer gpa.free(nm);
            const x = t orelse return;
            const b = kept.get(id) orelse return;
            if (x.bg != null) return bgw.followWrite(m, x, id, key, at, nm, base, b.ptr);
            x.sp.write(key, at, nm, base, b.ptr) catch |err| std.log.err("dsv41: spilling {d} tokens failed: {s}", .{ at, @errorName(err) });
        },
        .spill_load => {
            const id = try r.int(u64);
            const key = try r.int(u64);
            const at: usize = @intCast(try r.int(u64));
            const base: usize = @intCast(try r.int(u64));
            const slot = try r.int(u8) != 0;
            const nm = try readNames(r, gpa);
            defer gpa.free(nm);
            if (t) |x| bgw.settleKey(x, null, key, "a pass loads it");
            const ok = loadHere(gpa, m, t, kept, id, key, at, base, slot, nm);
            try peer.send(&.{@intFromBool(ok)});
        },
        .spill_forget => {
            const key = try r.int(u64);
            if (t) |x| bgw.settleKey(x, null, key, "it is forgotten");
            if (t) |x| x.sp.forget(key);
        },
        .spill_wait => bgw.followHold(t, try r.int(u64), "its rows go to another stream"),
        else => return error.BadFrame,
    }
}

fn loadHere(gpa: std.mem.Allocator, m: *model.Model, t: ?*Tier, kept: *std.AutoHashMapUnmanaged(u64, cuda.DeviceBuffer), id: u64, key: u64, at: usize, base: usize, slot: bool, nm: []const spill.Name) bool {
    const x = t orelse return false;
    if (!slot and !kept.contains(id)) return false; // the slot part rank 0 counts on is not here
    var fresh: ?cuda.DeviceBuffer = null;
    if (slot and !kept.contains(id)) fresh = cuda.DeviceBuffer.alloc(m.ctx.d, m.snapBytes()) catch return false;
    const dst: ?u64 = if (!slot) null else if (fresh) |b| b.ptr else kept.get(id).?.ptr;
    x.sp.read(key, at, nm, base, dst) catch |err| {
        std.log.err("dsv41: reading the spilled state of {d} tokens failed: {s}", .{ at, @errorName(err) });
        if (fresh) |*b| b.free();
        return false;
    };
    if (fresh) |b| kept.put(gpa, id, b) catch {
        var c = b;
        c.free();
        return false;
    };
    return true;
}

fn readWord(path: [:0]const u8) u64 {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return 0;
    defer _ = std.c.close(fd);
    var w: u64 = 0;
    return if (std.c.read(fd, std.mem.asBytes(&w), 8) == 8) w else 0;
}

fn writeWord(path: [:0]const u8, w: u64) void {
    const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return;
    defer _ = std.c.close(fd);
    _ = std.c.write(fd, std.mem.asBytes(&w), 8);
    _ = std.c.fsync(fd);
}

/// A follower with tier `t` answers the spill_sync frame in `w` over a socket pair; the reply's bytes, in `out`.
fn syncReply(gpa: std.mem.Allocator, t: *Tier, w: *lanes.Wire, out: []u8) ![]u8 {
    var fds: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, &fds) != 0) return error.SocketPair;
    defer for (fds) |fd| {
        _ = std.c.close(fd);
    };
    var kept: std.AutoHashMapUnmanaged(u64, cuda.DeviceBuffer) = .empty;
    defer kept.deinit(gpa);
    var m: *model.Model = undefined; // (a spill_sync frame never reads the model)
    _ = &m;
    var r: Reader = .{ .b = w.buf.items };
    try follow(gpa, m, t, .spill_sync, &r, &kept, .{ .fd = fds[0] });
    return (link.Link{ .fd = fds[1] }).recv(out);
}

test "spill_sync: a follower whose writer mode is rank 0's answers as it always did; one whose mode differs says so and is off" {
    const gpa = std.testing.allocator;
    const st = @import("spill_test.zig");
    var d1: [128]u8 = undefined;
    const dir = try st.scratch(&d1, "follow-sync");
    defer st.rmDir(dir);
    const mem = try gpa.alloc(u8, 16384);
    defer gpa.free(mem);
    var fake: st.Fake = .{ .mem = mem };
    const stage = try gpa.alignedAlloc(u8, .fromByteUnits(spill.page), 2 * spill.page);
    defer gpa.free(stage);
    var writing: bgw.Background = .{ .w = undefined, .host = undefined, .stream = undefined, .m = undefined, .limit_ns = 0 }; // (nothing queued)
    for ([_][2]bool{ .{ false, false }, .{ true, true }, .{ false, true }, .{ true, false } }) |modes| { // rank 0's, the follower's
        var t: Tier = .{ .sp = try spill.Spill.init(gpa, &st.Fake.planes, 16, st.Fake.slot_bytes, fake.dev(), stage), .host = undefined, .settings = undefined, .identity = 0x1122334455667788 };
        defer t.sp.deinit();
        try t.sp.open(dir);
        t.bg = if (modes[1]) &writing else null;
        var w: lanes.Wire = .{};
        defer w.deinit(gpa);
        try writer.putSync(&w, gpa, &.{ 7, 9 }, modes[0]);
        var buf: [16]u8 = undefined;
        const reply = try syncReply(gpa, &t, &w, &buf);
        try std.testing.expectEqual(@as(usize, 9), reply.len);
        try std.testing.expectEqual(@as(u8, if (modes[0] == modes[1]) 1 else 2), reply[0]); // 1: the byte it always sent
        try std.testing.expectEqual(@as(u64, 0x1122334455667788), std.mem.readInt(u64, reply[1..9], .little));
        try std.testing.expectEqual(modes[0] != modes[1], t.off);
        if (t.off) { // a tier turned off answers as none does
            var again: lanes.Wire = .{};
            defer again.deinit(gpa);
            try writer.putSync(&again, gpa, &.{}, modes[0]);
            const none = try syncReply(gpa, &t, &again, &buf);
            try std.testing.expectEqual(@as(u8, 0), none[0]);
        }
    }
}
