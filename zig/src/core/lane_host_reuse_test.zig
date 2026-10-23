//! LaneHost with the prompt cache: a conversation's next turn resumes from its kept prompt state (the fake backend).
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const LaneHost = api.LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Reason = api.Reason;

test "a lane host resumes a conversation from its kept prompt state and reports the cached tokens" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const Snaps = struct {
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + 4 * @as(u64, at);
        }
        fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            return f.save(@ptrCast(@alignCast(owner.?)), at);
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
            return error.BackendRestores; // lane-core backends restore inside their own prompt pass
        }
        fn drop(ptr: *anyopaque, saved: pc.Saved) void {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            f.drop(saved);
        }
    };
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        cached: ?u32 = null,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => |c| b.cached = c,
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const e = host.engine();
    const t1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    var b1: Box = .{};
    defer b1.tokens.deinit(gpa);
    const r1: Request = .{ .prompt = &t1, .max_tokens = 6, .history_len = 8 };
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b1.wait());
    try std.testing.expectEqual(@as(?u32, 0), b1.cached);
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    var t2: std.ArrayList(u32) = .empty; // the next turn: the history, the reply, a tool result, a new generation prompt
    defer t2.deinit(gpa);
    try t2.appendSlice(gpa, &t1);
    try t2.appendSlice(gpa, b1.tokens.items[0..4]);
    try t2.appendSlice(gpa, &.{ 7, 7, 5, 3 });
    var b2: Box = .{};
    defer b2.tokens.deinit(gpa);
    const r2: Request = .{ .prompt = t2.items, .max_tokens = 12, .history_len = @intCast(t2.items.len - 2) };
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expectEqual(@as(?u32, 8), b2.cached);
    var history: std.ArrayList(u32) = .empty; // the fake's tokens read the whole history: any slip in it changes them
    defer history.deinit(gpa);
    try history.appendSlice(gpa, t2.items);
    for (b2.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(u64, 1), store.counts.hits);
    try std.testing.expectEqual(@as(usize, 2), store.entries.items.len);
}

test "a turn's mark evicting the state it resumed from: the hit is reported first and the entry freed once" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const Snaps = struct { // a budget of one state: each keep records the hits the store had counted by then
        f: *lanes.fake.Fake,
        store: ?*pc.Store = null,
        hits_at_save: [4]u64 = @splat(0),
        saves: usize = 0,
        live: usize = 0,
        fn of(ptr: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ptr));
        }
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + 4 * @as(u64, at);
        }
        fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
            const x = of(ptr);
            x.hits_at_save[x.saves] = x.store.?.counts.hits;
            x.saves += 1;
            x.live += 1;
            return x.f.save(@ptrCast(@alignCast(owner.?)), at);
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
            return error.BackendRestores;
        }
        fn drop(ptr: *anyopaque, saved: pc.Saved) void {
            of(ptr).live -= 1;
            of(ptr).f.drop(saved);
        }
    };
    var snaps: Snaps = .{ .f = &target };
    var store = pc.Store.init(gpa, .{ .ptr = &snaps, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 150);
    defer store.deinit();
    snaps.store = &store;
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        cached: ?u32 = null,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => |c| b.cached = c,
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const e = host.engine();
    const t1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    var b1: Box = .{};
    defer b1.tokens.deinit(gpa);
    const r1: Request = .{ .prompt = &t1, .max_tokens = 6, .history_len = 8 };
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b1.wait());
    var t2: std.ArrayList(u32) = .empty; // 18 tokens: its history's state (128 bytes) and turn 1's (96) pass the budget together
    defer t2.deinit(gpa);
    try t2.appendSlice(gpa, &t1);
    try t2.appendSlice(gpa, b1.tokens.items[0..4]);
    try t2.appendSlice(gpa, &.{ 7, 7, 5, 3 });
    var b2: Box = .{};
    defer b2.tokens.deinit(gpa);
    const r2: Request = .{ .prompt = t2.items, .max_tokens = 12, .history_len = @intCast(t2.items.len - 2) };
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expectEqual(@as(?u32, 8), b2.cached);
    try std.testing.expectEqual(@as(usize, 2), snaps.saves);
    try std.testing.expectEqual(@as(u64, 1), snaps.hits_at_save[1]); // the resume was reported before the keep evicted it
    try std.testing.expectEqual(@as(u64, 1), store.counts.evicted);
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    try std.testing.expectEqual(@as(u32, 16), store.entries.items[0].at);
    try std.testing.expectEqual(@as(usize, 1), snaps.live);
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, t2.items);
    for (b2.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
}

test "a prompt with an image resumes when its image is the same, and not past the image when it differs" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const Snaps = struct {
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + 4 * @as(u64, at);
        }
        fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            return f.save(@ptrCast(@alignCast(owner.?)), at);
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
            return error.BackendRestores;
        }
        fn drop(ptr: *anyopaque, saved: pc.Saved) void {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            f.drop(saved);
        }
    };
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        cached: ?u32 = null,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => |c| b.cached = c,
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const e = host.engine();
    const image = [_]api.Image{.{ .at = 4, .tokens = 3, .bytes = "pixels" }};
    // an image span (positions 4 to 6, each the image token 99) that the key spells by the image's digest
    const t1 = [_]u32{ 3, 1, 4, 1, 99, 99, 99, 5, 9, 2 };
    const k1 = [_]u32{ 3, 1, 4, 1, 0x8000_0001, 0x8000_0002, 0x8000_0003, 5, 9, 2 };
    var b1: Box = .{};
    defer b1.tokens.deinit(gpa);
    const r1: Request = .{ .prompt = &t1, .images = &image, .cache_key = &k1, .max_tokens = 6, .history_len = 8 };
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b1.wait());
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    var t2: std.ArrayList(u32) = .empty; // the next turn: the history, the reply, a new user message
    defer t2.deinit(gpa);
    try t2.appendSlice(gpa, &t1);
    try t2.appendSlice(gpa, b1.tokens.items[0..4]);
    try t2.appendSlice(gpa, &.{ 7, 7, 5, 3 });
    var k2: std.ArrayList(u32) = .empty;
    defer k2.deinit(gpa);
    try k2.appendSlice(gpa, &k1);
    try k2.appendSlice(gpa, t2.items[k1.len..]);
    var b2: Box = .{};
    defer b2.tokens.deinit(gpa);
    const r2: Request = .{ .prompt = t2.items, .images = &image, .cache_key = k2.items, .max_tokens = 4, .history_len = @intCast(t2.items.len - 2) };
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expectEqual(@as(?u32, 8), b2.cached); // the same image: resumed past it
    // the same tokens with another image at the span: its key differs from position 4, so nothing is resumed
    var k3: std.ArrayList(u32) = .empty;
    defer k3.deinit(gpa);
    try k3.appendSlice(gpa, k2.items);
    k3.items[4] = 0x8000_0011;
    k3.items[5] = 0x8000_0012;
    k3.items[6] = 0x8000_0013;
    var b3: Box = .{};
    defer b3.tokens.deinit(gpa);
    const r3: Request = .{ .prompt = t2.items, .images = &image, .cache_key = k3.items, .max_tokens = 4, .history_len = @intCast(t2.items.len - 2) };
    try e.submit(3, &r3, .{ .ctx = &b3, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b3.wait());
    try std.testing.expectEqual(@as(?u32, 0), b3.cached);
}

/// A fake snapshot vtable over lanes.fake.Fake (states at any position), a sink that records one reply, and step 1 of
/// every step-ahead test on the production tail (DeepSeek's 128-row window) and a 16-row grid.
const AheadKit = struct {
    /// Step 1's prompt (160 tokens), kept at its history 32 (a chunk end the tail before its end, as dsv41 keeps); its
    /// 144 reply tokens make 304, so the pass keeps at 176 (304 - 128 down to 16s), 16 tokens into the reply.
    const t1 = blk: {
        var a: [160]u32 = undefined;
        for (&a, 0..) |*x, i| x.* = @intCast((i * 31 + 3) % 97);
        break :blk a;
    };
    const history = 32;
    const reply = 144;
    const at = 176;
    const warm_len = 304;
    const grid: LaneHost.Grid = .{ .chunk = 16, .tail = 128 };
    // (min_gap 32: the stable mark at step 1's prompt end, 160, is no state of its own next to 176, as on dsv41, where
    // it floors onto a kept point)
    const rules: pc.Rules = .{ .min_gap = 32, .min_prompt = 0, .tail = 128 };

    const Snaps = struct {
        fn bytes(_: *anyopaque, n: u32) u64 {
            return 64 + 4 * @as(u64, n);
        }
        fn save(ptr: *anyopaque, owner: ?*anyopaque, n: u32) anyerror!pc.Saved {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            return f.save(@ptrCast(@alignCast(owner.?)), n);
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
            return error.BackendRestores;
        }
        fn drop(ptr: *anyopaque, saved: pc.Saved) void {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            f.drop(saved);
        }
    };
    const vtable: pc.Snapshots.VTable = .{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop };

    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        cached: ?u32 = null,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => |c| b.cached = c,
                .tokens => |t| b.tokens.appendSlice(std.testing.allocator, t) catch {},
                .finished => |f| b.done = f.reason,
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };

    const r1: Request = .{ .prompt = &t1, .max_tokens = reply, .history_len = history };

    /// The next step: step 1's prompt, its reply as the client sends it back (the first token changed when `changed`),
    /// a tool result (308 tokens). The caller frees it.
    fn next(gpa: std.mem.Allocator, got: []const u32, changed: bool) ![]u32 {
        var p: std.ArrayList(u32) = .empty;
        errdefer p.deinit(gpa);
        try p.appendSlice(gpa, &t1);
        try p.appendSlice(gpa, got);
        if (changed) p.items[t1.len] = (p.items[t1.len] + 1) % lanes.fake.vocab;
        try p.appendSlice(gpa, &.{ 7, 7, 5, 3 });
        return p.toOwnedSlice(gpa);
    }

    fn expectFake(prompt: []const u32, got: []const u32) !void {
        var h: std.ArrayList(u32) = .empty;
        defer h.deinit(std.testing.allocator);
        try h.appendSlice(std.testing.allocator, prompt);
        for (got) |t| {
            try std.testing.expectEqual(lanes.fake.next(h.items, null, h.items.len), t);
            try h.append(std.testing.allocator, t);
        }
    }

    /// The store's entry at `n` tokens that is a prefix of `prompt`.
    fn entry(store: *pc.Store, prompt: []const u32, n: u32) ?*pc.Entry {
        for (store.entries.items) |e| if (e.at == n and std.mem.startsWith(u32, prompt, e.tokens)) return e;
        return null;
    }

    fn counts(host: *LaneHost) LaneHost.AheadCounts {
        host.mutex.lockUncancelable(std.testing.io);
        defer host.mutex.unlock(std.testing.io);
        return host.ahead_counts;
    }
};

test "step-ahead: a reply's own tokens are read while the client works, and its next step resumes past them with the same tokens" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &AheadKit.vtable }, AheadKit.rules, 1 << 20);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    host.ahead_on = true;
    host.ahead_grid = AheadKit.grid;
    var warm: AheadKit.Box = .{};
    defer warm.tokens.deinit(gpa);
    host.ahead_sink = .{ .ctx = &warm, .event = AheadKit.Box.event };
    try host.start();
    defer host.stop();
    const e = host.engine();
    var b1: AheadKit.Box = .{};
    defer b1.tokens.deinit(gpa);
    try e.submit(1, &AheadKit.r1, .{ .ctx = &b1, .event = AheadKit.Box.event });
    try std.testing.expectEqual(Reason.length, b1.wait());
    try std.testing.expectEqual(@as(?u32, 0), b1.cached);
    // the step-ahead pass: one token past a pass that resumed at 32, its state at 176 provisional
    try std.testing.expectEqual(Reason.length, warm.wait());
    try std.testing.expectEqual(@as(?u32, AheadKit.history), warm.cached);
    const t2 = try AheadKit.next(gpa, b1.tokens.items, false);
    defer gpa.free(t2);
    const kept = AheadKit.entry(&store, t2, AheadKit.at) orelse return error.NotKept;
    try std.testing.expect(kept.provisional);
    try std.testing.expect(!AheadKit.entry(&store, t2, AheadKit.history).?.provisional);
    // step 2: the history, the whole reply, a tool result: resumes at 176, not 32, and reads the history exactly
    var b2: AheadKit.Box = .{};
    defer b2.tokens.deinit(gpa);
    const r2: Request = .{ .prompt = t2, .max_tokens = 12, .history_len = @intCast(t2.len - 2), .drafts = true };
    try e.submit(2, &r2, .{ .ctx = &b2, .event = AheadKit.Box.event });
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expectEqual(@as(?u32, AheadKit.at), b2.cached);
    try AheadKit.expectFake(t2, b2.tokens.items);
    try std.testing.expectEqual(@as(u64, 1), AheadKit.counts(&host).resumed);
}

test "step-ahead: nothing runs when off, or when the reply moves no kept state a chunk on" {
    const gpa = std.testing.allocator;
    for ([_]bool{ false, true }) |on| {
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        var target: lanes.fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer core.deinit();
        var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &AheadKit.vtable }, AheadKit.rules, 1 << 20);
        defer store.deinit();
        var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
        host.cache = &store;
        host.ahead_on = on;
        // 160 + 2 tokens, less the tail, reach no chunk end past the kept 32: no pass when on either
        host.ahead_grid = AheadKit.grid;
        try host.start();
        const e = host.engine();
        var b1: AheadKit.Box = .{};
        defer b1.tokens.deinit(gpa);
        const r1: Request = .{ .prompt = &AheadKit.t1, .max_tokens = 2, .history_len = AheadKit.history };
        try e.submit(1, &r1, .{ .ctx = &b1, .event = AheadKit.Box.event });
        try std.testing.expectEqual(Reason.length, b1.wait());
        host.stop();
        try std.testing.expectEqual(@as(u64, 0), host.ahead_counts.queued);
        try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    }
}

test "step-ahead: a reply the client sends back changed resumes where it would without the pass, and the pass's state goes first" {
    const gpa = std.testing.allocator;
    var got: [2]?u32 = undefined;
    for ([_]bool{ false, true }, 0..) |on, k| {
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        var target: lanes.fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer core.deinit();
        // room for step 1's state at 32 (192 bytes) and the pass's at 176 (768), not for one more of 192
        var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &AheadKit.vtable }, AheadKit.rules, 1100);
        defer store.deinit();
        var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
        host.cache = &store;
        host.ahead_on = on;
        host.ahead_grid = AheadKit.grid;
        var warm: AheadKit.Box = .{};
        defer warm.tokens.deinit(gpa);
        host.ahead_sink = .{ .ctx = &warm, .event = AheadKit.Box.event };
        try host.start();
        var stopped = false;
        defer if (!stopped) host.stop();
        const e = host.engine();
        var b1: AheadKit.Box = .{};
        defer b1.tokens.deinit(gpa);
        try e.submit(1, &AheadKit.r1, .{ .ctx = &b1, .event = AheadKit.Box.event });
        try std.testing.expectEqual(Reason.length, b1.wait());
        const sent = try AheadKit.next(gpa, b1.tokens.items, false); // as generated: what the pass read
        defer gpa.free(sent);
        if (on) {
            try std.testing.expectEqual(Reason.length, warm.wait());
            try std.testing.expect(AheadKit.entry(&store, sent, AheadKit.at).?.provisional);
        }
        // another conversation keeps a state: the budget evicts one, the pass's (it extends step 1's state, which
        // would have gone first as superseded)
        var q: [40]u32 = undefined;
        for (&q, 0..) |*x, i| x.* = @intCast((i * 13 + 5) % 97);
        var bq: AheadKit.Box = .{};
        defer bq.tokens.deinit(gpa);
        const rq: Request = .{ .prompt = &q, .max_tokens = 2, .history_len = AheadKit.history };
        try e.submit(2, &rq, .{ .ctx = &bq, .event = AheadKit.Box.event });
        try std.testing.expectEqual(Reason.length, bq.wait());
        try std.testing.expect(AheadKit.entry(&store, sent, AheadKit.history) != null);
        try std.testing.expect(AheadKit.entry(&store, sent, AheadKit.at) == null);
        try std.testing.expectEqual(@as(u64, if (on) 1 else 0), store.counts.evicted);
        // the client sends step 1's reply back with its first token changed (re-rendered): no use for the pass's state
        const p2 = try AheadKit.next(gpa, b1.tokens.items, true);
        defer gpa.free(p2);
        var b2: AheadKit.Box = .{};
        defer b2.tokens.deinit(gpa);
        const r2: Request = .{ .prompt = p2, .max_tokens = 4, .history_len = AheadKit.at };
        try e.submit(3, &r2, .{ .ctx = &b2, .event = AheadKit.Box.event });
        try std.testing.expectEqual(Reason.length, b2.wait());
        try AheadKit.expectFake(p2, b2.tokens.items);
        got[k] = b2.cached;
        host.stop();
        stopped = true;
        try std.testing.expectEqual(@as(u64, if (on) 1 else 0), host.ahead_counts.ran);
        try std.testing.expectEqual(@as(u64, 0), host.ahead_counts.resumed);
    }
    try std.testing.expectEqual(@as(?u32, AheadKit.history), got[0]);
    try std.testing.expectEqual(got[0], got[1]);
}

/// A step-ahead pass in pieces with cut-in on (TF_FILL_CHUNKS=1, TF_FILL_CUT_IN), the fake checking the cancel before
/// each chunk past the first as dsv41 does, held in its second chunk (its second piece, run between rounds) until a
/// second request is in the queue: the chunks in the order they ran, each request's reply and where it resumed.
const AheadCut = struct {
    mutex: std.Io.Mutex = .init,
    order: std.ArrayList([2]usize) = .empty, // (prompt length, chunk) as the fake ran them
    held: bool = false, // the step-ahead pass's second chunk runs
    sent: bool = false, // the second request is queued

    fn hook(ctx: *anyopaque, s: *lanes.stream.Stream, chunk: usize) void {
        const w: *AheadCut = @ptrCast(@alignCast(ctx));
        w.mutex.lockUncancelable(std.testing.io);
        w.order.append(std.testing.allocator, .{ s.prompt_len, chunk }) catch {};
        const hold = s.prompt_len == AheadKit.warm_len and chunk == 1;
        if (hold) w.held = true;
        w.mutex.unlock(std.testing.io);
        if (hold) w.until(.sent) catch {};
    }

    fn until(w: *AheadCut, comptime what: enum { held, sent }) !void {
        for (0..20000) |_| {
            w.mutex.lockUncancelable(std.testing.io);
            const ok = if (what == .held) w.held else w.sent;
            w.mutex.unlock(std.testing.io);
            if (ok) return;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }

    const Result = struct { order: [][2]usize, warm: Reason, cached: ?u32, counts: LaneHost.AheadCounts, lanes_left: usize, cancel_stops: usize };

    /// Step 1, its step-ahead pass held in its second chunk, then `second` (built from step 1's reply) and, once it
    /// ended, `third` if any; every reply checked as the fake's. The caller frees `order`.
    fn run(comptime second: fn (gpa: std.mem.Allocator, reply: []const u32) anyerror![]u32, third: bool) !Result {
        const gpa = std.testing.allocator;
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        var w: AheadCut = .{};
        errdefer w.order.deinit(gpa);
        var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 6, .prefill_hook = hook, .prefill_hook_ctx = &w, .cancel_before = true };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer core.deinit();
        core.fill_chunks = 1;
        core.fill_rounds = 2;
        core.cut_in = 512; // every request below is short enough to cut in
        var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &AheadKit.vtable }, AheadKit.rules, 1 << 20);
        defer store.deinit();
        var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
        host.cache = &store;
        host.ahead_on = true;
        host.ahead_grid = AheadKit.grid;
        var warm: AheadKit.Box = .{};
        defer warm.tokens.deinit(gpa);
        host.ahead_sink = .{ .ctx = &warm, .event = AheadKit.Box.event };
        try host.start();
        var stopped = false;
        defer if (!stopped) host.stop();
        const e = host.engine();
        var b1: AheadKit.Box = .{};
        defer b1.tokens.deinit(gpa);
        try e.submit(1, &AheadKit.r1, .{ .ctx = &b1, .event = AheadKit.Box.event });
        try std.testing.expectEqual(Reason.length, b1.wait());
        try w.until(.held);
        const p2 = try second(gpa, b1.tokens.items);
        defer gpa.free(p2);
        var b2: AheadKit.Box = .{};
        defer b2.tokens.deinit(gpa);
        // one reply token, its history mark the pass's point: no step-ahead pass after it
        const r2: Request = .{ .prompt = p2, .max_tokens = if (p2.len > 8) 1 else 2, .history_len = if (p2.len > 8) AheadKit.at else 0 };
        try e.submit(2, &r2, .{ .ctx = &b2, .event = AheadKit.Box.event });
        w.mutex.lockUncancelable(std.testing.io);
        w.sent = true;
        w.mutex.unlock(std.testing.io);
        const warm_done = warm.wait();
        try std.testing.expectEqual(Reason.length, b2.wait());
        try AheadKit.expectFake(p2, b2.tokens.items);
        var cached = b2.cached;
        if (third) { // the step after it: step 1's history, its reply and a tool result
            const p3 = try leads(gpa, b1.tokens.items);
            defer gpa.free(p3);
            var b3: AheadKit.Box = .{};
            defer b3.tokens.deinit(gpa);
            const r3: Request = .{ .prompt = p3, .max_tokens = 1, .history_len = AheadKit.at };
            try e.submit(3, &r3, .{ .ctx = &b3, .event = AheadKit.Box.event });
            try std.testing.expectEqual(Reason.length, b3.wait());
            try AheadKit.expectFake(p3, b3.tokens.items);
            cached = b3.cached;
        }
        host.stop();
        stopped = true;
        return .{ .order = try w.order.toOwnedSlice(gpa), .warm = warm_done, .cached = cached, .counts = host.ahead_counts, .lanes_left = target.lanes.count(), .cancel_stops = target.cancel_stops };
    }

    /// The next step the step-ahead pass leads to.
    fn leads(gpa: std.mem.Allocator, reply: []const u32) anyerror![]u32 {
        return AheadKit.next(gpa, reply, false);
    }

    /// Another conversation's short prompt.
    fn other(gpa: std.mem.Allocator, _: []const u32) anyerror![]u32 {
        return gpa.dupe(u32, &.{ 8, 6, 7 });
    }
};

test "step-ahead and cut-in: a short arrival it does not lead to cancels the filling pass between its pieces, never pauses it" {
    const got = try AheadCut.run(AheadCut.other, true);
    defer std.testing.allocator.free(got.order);
    try std.testing.expectEqual(Reason.cancelled, got.warm);
    // step 1's six chunks, the step-ahead pass's first two (its first piece in its admission, its second between
    // rounds), then the arrival's six; the pass never resumes (a cut-in would have paused it and run its other four after)
    try std.testing.expectEqual(@as(usize, 6 + 2 + 6 + 6), got.order.len);
    for (got.order[6..8], 0..) |o, i| try std.testing.expectEqual([2]usize{ AheadKit.warm_len, i }, o);
    for (got.order[8..14], 0..) |o, i| try std.testing.expectEqual([2]usize{ 3, i }, o);
    for (got.order) |o| try std.testing.expect(o[0] != AheadKit.warm_len or o[1] < 2);
    // the host's takeCancels discarded it between pieces: no chunk's own check ended it
    try std.testing.expectEqual(@as(usize, 0), got.cancel_stops);
    // nothing kept at 176: the step it led to resumes where it did before, at step 1's history
    try std.testing.expectEqual(@as(?u32, AheadKit.history), got.cached);
    try std.testing.expectEqual(@as(u64, 1), got.counts.cancelled);
    try std.testing.expectEqual(@as(u64, 0), got.counts.ran);
    try std.testing.expectEqual(@as(usize, 0), got.lanes_left);
}

test "step-ahead and cut-in: a short arrival it leads to waits for the whole pass and resumes at its kept point" {
    const got = try AheadCut.run(AheadCut.leads, false);
    defer std.testing.allocator.free(got.order);
    try std.testing.expectEqual(Reason.length, got.warm);
    // step 1's six chunks, the step-ahead pass's six, then the arrival's: short as it is, it did not cut in
    try std.testing.expectEqual(@as(usize, 18), got.order.len);
    for (got.order[6..12], 0..) |o, i| try std.testing.expectEqual([2]usize{ AheadKit.warm_len, i }, o);
    for (got.order[12..18], 0..) |o, i| try std.testing.expectEqual([2]usize{ AheadKit.warm_len + 4, i }, o);
    try std.testing.expectEqual(@as(?u32, AheadKit.at), got.cached);
    try std.testing.expectEqual(@as(u64, 1), got.counts.ran);
    try std.testing.expectEqual(@as(u64, 1), got.counts.resumed);
    try std.testing.expectEqual(@as(u64, 0), got.counts.cancelled);
    try std.testing.expectEqual(@as(usize, 0), got.lanes_left);
}

/// The spill tier over the fake: kept states written to an in-memory disk, each write finished when the test says
/// (the async writer's `settled`), read back as new states.
const AheadDisk = struct {
    f: *lanes.fake.Fake,
    mutex: std.Io.Mutex = .init,
    files: std.AutoHashMapUnmanaged(u64, []u32) = .empty,
    keys: std.ArrayList(u64) = .empty, // writes in the order they were queued
    done: usize = 0, // the first `done` of them finished
    told: usize = 0, // the store heard of these

    fn of(ptr: *anyopaque) *AheadDisk {
        return @ptrCast(@alignCast(ptr));
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        return of(ptr).f.save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        of(ptr).f.drop(saved);
    }
    fn write(ptr: *anyopaque, saved: pc.Saved, _: [:0]const u8, key: u64, _: []const u32) anyerror!void {
        const d = of(ptr);
        const kept: *std.ArrayList(u32) = @ptrCast(@alignCast(saved));
        d.mutex.lockUncancelable(std.testing.io);
        defer d.mutex.unlock(std.testing.io);
        const copy = try std.testing.allocator.dupe(u32, kept.items);
        const got = try d.files.getOrPut(std.testing.allocator, key);
        if (got.found_existing) std.testing.allocator.free(got.value_ptr.*);
        got.value_ptr.* = copy;
        try d.keys.append(std.testing.allocator, key);
    }
    fn read(ptr: *anyopaque, _: [:0]const u8, key: u64, at: u32, _: []const u32) anyerror!pc.Saved {
        const d = of(ptr);
        d.mutex.lockUncancelable(std.testing.io);
        defer d.mutex.unlock(std.testing.io);
        const file = d.files.get(key) orelse return error.FileNotFound;
        if (file.len != at) return error.BadState;
        const kept = try std.testing.allocator.create(std.ArrayList(u32));
        kept.* = .empty;
        try kept.appendSlice(std.testing.allocator, file);
        return kept;
    }
    fn forget(ptr: *anyopaque, _: [:0]const u8, key: u64) void {
        const d = of(ptr);
        d.mutex.lockUncancelable(std.testing.io);
        defer d.mutex.unlock(std.testing.io);
        if (d.files.fetchRemove(key)) |kv| std.testing.allocator.free(kv.value);
    }
    fn diskBytes(_: *anyopaque, at: u32) u64 {
        return 64 + 4 * @as(u64, at);
    }
    fn settled(ptr: *anyopaque, keys: []u64, failures: []?anyerror, wait: bool) usize {
        const d = of(ptr);
        d.mutex.lockUncancelable(std.testing.io);
        defer d.mutex.unlock(std.testing.io);
        if (wait) d.done = d.keys.items.len;
        var k: usize = 0;
        while (d.told < d.done and k < keys.len) : (k += 1) {
            keys[k] = d.keys.items[d.told];
            failures[k] = null;
            d.told += 1;
        }
        return k;
    }
    fn finishAll(d: *AheadDisk) void {
        d.mutex.lockUncancelable(std.testing.io);
        defer d.mutex.unlock(std.testing.io);
        d.done = d.keys.items.len;
    }
    fn deinit(d: *AheadDisk) void {
        var it = d.files.valueIterator();
        while (it.next()) |v| std.testing.allocator.free(v.*);
        d.files.deinit(std.testing.allocator);
        d.keys.deinit(std.testing.allocator);
    }
    const vtable: pc.Snapshots.VTable = .{ .bytes = AheadKit.Snaps.bytes, .save = save, .restore = AheadKit.Snaps.restore, .drop = drop, .write = write, .read = read, .forget = forget, .disk_bytes = diskBytes, .settled = settled };
};

test "step-ahead and the spill tier: its kept state goes through the queued writer as any kept state, listed once written, and serves the next step after a restart" {
    const gpa = std.testing.allocator;
    const imprint = @import("prompt_imprint.zig");
    var buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&buf, "/tmp/tf-ahead-spill-{d}", .{std.c.getpid()});
    defer @import("prompt_imprint_test.zig").rmTree(root);
    var im = try imprint.Imprint.open(gpa, root, 4, 1 << 30);
    defer im.deinit();
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var disk: AheadDisk = .{ .f = &target };
    defer disk.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var b1: AheadKit.Box = .{};
    defer b1.tokens.deinit(gpa);
    {
        var store = pc.Store.init(gpa, .{ .ptr = &disk, .vtable = &AheadDisk.vtable }, AheadKit.rules, 1 << 20);
        defer store.deinit(); // (lists the writes still queued, as a server's close does)
        store.imprint = &im;
        store.spill = true;
        var warm: AheadKit.Box = .{};
        defer warm.tokens.deinit(gpa);
        var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
        defer host.stop();
        host.cache = &store;
        host.ahead_on = true;
        host.ahead_grid = AheadKit.grid;
        host.ahead_sink = .{ .ctx = &warm, .event = AheadKit.Box.event };
        try host.start();
        try host.engine().submit(1, &AheadKit.r1, .{ .ctx = &b1, .event = AheadKit.Box.event });
        try std.testing.expectEqual(Reason.length, b1.wait());
        try std.testing.expectEqual(Reason.length, warm.wait());
        // step 1's state at 32 and the pass's at 176, both queued to the writer and neither listed yet
        try std.testing.expectEqual(@as(usize, 2), store.pending.items.len);
        try std.testing.expectEqual(@as(usize, 0), im.metas.items.len);
        disk.finishAll();
    }
    var warm_tokens: [AheadKit.at]u32 = undefined;
    @memcpy(warm_tokens[0..AheadKit.t1.len], &AheadKit.t1);
    @memcpy(warm_tokens[AheadKit.t1.len..], b1.tokens.items[0 .. AheadKit.at - AheadKit.t1.len]);
    try std.testing.expect(im.has(imprint.Imprint.keyOf(AheadKit.t1[0..AheadKit.history])) and im.has(imprint.Imprint.keyOf(&warm_tokens)));
    // a restart: nothing in memory, the index on disk; the step after step 1 reads the pass's state back
    var store = pc.Store.init(gpa, .{ .ptr = &disk, .vtable = &AheadDisk.vtable }, AheadKit.rules, 1 << 20);
    defer store.deinit();
    store.imprint = &im;
    store.spill = true;
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    defer host.stop();
    host.cache = &store;
    try host.start();
    const p2 = try AheadCut.leads(gpa, b1.tokens.items);
    defer gpa.free(p2);
    var b2: AheadKit.Box = .{};
    defer b2.tokens.deinit(gpa);
    const r2: Request = .{ .prompt = p2, .max_tokens = 4, .history_len = AheadKit.at };
    try host.engine().submit(2, &r2, .{ .ctx = &b2, .event = AheadKit.Box.event });
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expectEqual(@as(?u32, AheadKit.at), b2.cached);
    try AheadKit.expectFake(p2, b2.tokens.items);
}
