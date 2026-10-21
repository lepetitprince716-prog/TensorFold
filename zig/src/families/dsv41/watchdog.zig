//! A rank's watchdog: a primitive with no progress for the limit, or a peer that hung up, ends its pair (lanes.zig).
const std = @import("std");
const posix = std.posix;

/// Seconds a primitive may run without a tick when TF_DS_WATCHDOG_S is unset.
pub const default_limit_s: u64 = 120;

/// Seconds the start-up exchange may wait for the other rank when TF_DS_WATCHDOG_START_S is unset (a cold model load).
pub const default_start_s: u64 = 600;

/// What a watchdog is set to: the quiet limits in seconds (0: unwatched) and the links whose hangup goes to `hangup`.
pub const Config = struct { limit_s: u64, start_s: u64 = default_start_s, peers: []const std.c.fd_t = &.{}, hangup: ?Hangup = null };

/// A primitive that stopped ticking: its name and the seconds since its last progress.
pub const Stall = struct { name: []const u8, seconds: u64 };

/// What the watchdog thread calls when a primitive stalls.
pub const Fire = *const fn (ctx: *anyopaque, stall: Stall) void;

/// What the watchdog thread calls, once, when a peer's link has hung up.
pub const Hangup = *const fn (ctx: *anyopaque) void;

/// The most links a watchdog watches (rank 0 of the four-node split has three).
pub const max_peers = 4;

/// Where the watchdog reads time: nanoseconds on a monotonic clock (a test sets its own by hand).
pub const Clock = struct {
    ctx: *anyopaque,
    now: *const fn (ctx: *anyopaque) u64,
};

/// TF_DS_WATCHDOG_S as the limit in seconds: unset or empty is the default, 0 turns the watchdog off.
pub fn limitFromText(text: ?[]const u8) error{BadWatchdog}!u64 {
    const t = text orelse return default_limit_s;
    if (t.len == 0) return default_limit_s;
    return std.fmt.parseInt(u64, t, 10) catch error.BadWatchdog;
}

pub fn limitFromEnv() error{BadWatchdog}!u64 {
    const v = std.c.getenv("TF_DS_WATCHDOG_S") orelse return limitFromText(null);
    return limitFromText(std.mem.span(v));
}

/// TF_DS_WATCHDOG_START_S as the start-up window's limit in seconds: unset is the default, 0 leaves it unwatched.
pub fn startLimitFromText(text: ?[]const u8) error{BadWatchdogStart}!u64 {
    const t = text orelse return default_start_s;
    if (t.len == 0) return default_start_s;
    return std.fmt.parseInt(u64, t, 10) catch error.BadWatchdogStart;
}

pub fn startLimitFromEnv() error{BadWatchdogStart}!u64 {
    const v = std.c.getenv("TF_DS_WATCHDOG_START_S") orelse return startLimitFromText(null);
    return startLimitFromText(std.mem.span(v));
}

pub const Watchdog = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    clock: Clock,
    limit_ns: u64,
    start_ns: u64 = 0,
    fire_ctx: *anyopaque,
    fire: Fire,
    /// Guards active, reported, name, since and window_ns: the primitive's thread writes, the watchdog's thread reads.
    lock: std.atomic.Value(bool) = .init(false),
    active: bool = false,
    reported: bool = false,
    name: []const u8 = "",
    since: u64 = 0,
    window_ns: u64 = 0,
    peers: [max_peers]std.c.fd_t = undefined,
    npeers: usize = 0,
    hangup: ?Hangup = null,
    hung: bool = false, // the thread's own: a hangup is said once
    stopped: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    sleep_ms: u64 = 100,
    every: u32 = 10,

    /// Starts the watchdog thread: `fire` runs there once a primitive has made no progress for its window's limit.
    pub fn start(gpa: std.mem.Allocator, io: std.Io, c: Config, fire_ctx: *anyopaque, fire: Fire) !*Watchdog {
        const ns = std.time.ns_per_s;
        var w: Watchdog = .{ .gpa = gpa, .io = io, .clock = undefined, .limit_ns = c.limit_s *| ns, .start_ns = c.start_s *| ns, .fire_ctx = fire_ctx, .fire = fire, .hangup = c.hangup };
        w.npeers = @min(c.peers.len, max_peers);
        @memcpy(w.peers[0..w.npeers], c.peers[0..w.npeers]);
        return spawn(w);
    }

    /// A watchdog set up by hand, with its thread: it sleeps `sleep_ms` at a time and checks every `every` sleeps.
    fn spawn(template: Watchdog) !*Watchdog {
        const w = try template.gpa.create(Watchdog);
        errdefer template.gpa.destroy(w);
        w.* = template;
        w.clock = .{ .ctx = w, .now = awake };
        w.thread = try std.Thread.spawn(.{}, loop, .{w});
        return w;
    }

    /// Stops the thread and frees the watchdog. Call before the model whose rings it may abort goes away.
    pub fn stop(w: *Watchdog) void {
        w.stopped.store(true, .release);
        if (w.thread) |t| t.join();
        w.gpa.destroy(w);
    }

    /// A primitive starts: its window opens now, with the primitive limit.
    pub fn busy(w: *Watchdog, name: []const u8) void {
        w.open(name, w.limit_ns);
    }

    /// The start-up exchange starts: a window with its own limit, for a peer that may still be loading (0: unwatched).
    pub fn busyStartup(w: *Watchdog, name: []const u8) void {
        w.open(name, w.start_ns);
    }

    fn open(w: *Watchdog, name: []const u8, limit_ns: u64) void {
        const at = w.clock.now(w.clock.ctx);
        w.acquire();
        defer w.release();
        w.active = true;
        w.reported = false;
        w.name = name;
        w.since = at;
        w.window_ns = limit_ns;
    }

    /// The primitive made progress (a prompt chunk, a round): its window restarts. Idle: nothing to restart.
    pub fn tick(w: *Watchdog) void {
        const at = w.clock.now(w.clock.ctx);
        w.acquire();
        defer w.release();
        if (!w.active) return;
        w.reported = false;
        w.since = at;
    }

    /// The primitive ended, or the rank waits for its next frame: nothing can stall.
    pub fn idle(w: *Watchdog) void {
        w.acquire();
        defer w.release();
        w.active = false;
    }

    /// The busy primitive that has not ticked for its window's limit, reported once until it ticks again; null: none.
    pub fn check(w: *Watchdog) ?Stall {
        if (w.limit_ns == 0) return null;
        const at = w.clock.now(w.clock.ctx);
        w.acquire();
        defer w.release();
        if (!w.active or w.reported or w.window_ns == 0) return null;
        const quiet = at -| w.since;
        if (quiet < w.window_ns) return null;
        w.reported = true;
        return .{ .name = w.name, .seconds = quiet / std.time.ns_per_s };
    }

    fn acquire(w: *Watchdog) void {
        while (w.lock.swap(true, .acquire)) std.atomic.spinLoopHint();
    }

    fn release(w: *Watchdog) void {
        w.lock.store(false, .release);
    }

    fn awake(ctx: *anyopaque) u64 {
        const w: *Watchdog = @ptrCast(@alignCast(ctx));
        return @intCast(std.Io.Clock.awake.now(w.io).toNanoseconds());
    }

    fn loop(w: *Watchdog) void {
        var slices: u32 = 0;
        while (!w.stopped.load(.acquire)) {
            std.Io.sleep(w.io, .fromMilliseconds(@intCast(w.sleep_ms)), .awake) catch {};
            slices += 1;
            if (slices < w.every) continue;
            slices = 0; // a check every second by default, the sleeps short so that stop returns at once
            if (w.stopped.load(.acquire)) break;
            if (w.check()) |stall| w.fire(w.fire_ctx, stall);
            if (!w.hung and hungUp(w.peers[0..w.npeers])) {
                w.hung = true; // said once: the link stays hung up, and with TF_DS_PAIR_BROKEN=log the callback returns
                if (w.hangup) |gone| gone(w.fire_ctx);
            }
        }
    }
};

/// What poll asks for: the peer's close (Linux), or readable (EOF is that; elsewhere POLLRDHUP does not exist).
const hangup_events: i16 = if (@hasDecl(posix.POLL, "RDHUP")) posix.POLL.RDHUP else posix.POLL.IN;

/// Whether a peer closed one of `fds` with nothing left to read (poll, then a peek that takes nothing off the stream).
pub fn hungUp(fds: []const std.c.fd_t) bool {
    for (fds) |fd| {
        var p = [_]posix.pollfd{.{ .fd = fd, .events = hangup_events, .revents = 0 }};
        const ready = posix.poll(&p, 0) catch continue;
        if (ready == 0 or p[0].revents & (hangup_events | posix.POLL.HUP | posix.POLL.ERR) == 0) continue;
        var probe: [1]u8 = undefined;
        const rc = posix.system.recvfrom(fd, &probe, 1, posix.MSG.PEEK | posix.MSG.DONTWAIT, null, null);
        switch (posix.errno(rc)) {
            .SUCCESS => if (rc == 0) return true,
            .AGAIN, .INTR => {},
            else => return true,
        }
    }
    return false;
}

const testing = std.testing;
const sec = std.time.ns_per_s;

/// A clock the test sets by hand.
const Fake = struct {
    at: u64 = 0,

    fn now(ctx: *anyopaque) u64 {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        return f.at;
    }

    fn clock(f: *Fake) Clock {
        return .{ .ctx = f, .now = now };
    }
};

fn never(_: *anyopaque, _: Stall) void {}

/// A watchdog on the fake clock, no thread.
fn watcher(f: *Fake, limit_s: u64) Watchdog {
    return .{ .gpa = testing.allocator, .io = testing.io, .clock = f.clock(), .limit_ns = limit_s * sec, .start_ns = default_start_s * sec, .fire_ctx = f, .fire = never };
}

test "a primitive with no tick fires at the limit and not a second before, once" {
    var f: Fake = .{ .at = 1000 * sec };
    var w = watcher(&f, 120);
    w.busy("verify");
    f.at += 119 * sec;
    try testing.expect(w.check() == null);
    f.at += sec;
    const stall = w.check().?;
    try testing.expectEqualStrings("verify", stall.name);
    try testing.expectEqual(@as(u64, 120), stall.seconds);
    f.at += 600 * sec;
    try testing.expect(w.check() == null); // said once
    w.tick(); // progress after all: the next stall is reported again
    f.at += 119 * sec;
    try testing.expect(w.check() == null);
    f.at += sec;
    try testing.expectEqual(@as(u64, 120), w.check().?.seconds);
}

test "ticks keep a primitive alive however long it runs, and the limit counts from the last one" {
    var f: Fake = .{};
    var w = watcher(&f, 120);
    w.busy("fill");
    for (0..1000) |_| { // a prompt pass of a day and more, a chunk every 100 s
        f.at += 100 * sec;
        try testing.expect(w.check() == null);
        w.tick();
    }
    f.at += 119 * sec;
    try testing.expect(w.check() == null);
    f.at += sec;
    try testing.expectEqualStrings("fill", w.check().?.name);
}

test "idle never fires, a tick while idle arms nothing, and the next busy opens a fresh window" {
    var f: Fake = .{};
    var w = watcher(&f, 120);
    try testing.expect(w.check() == null); // never busy
    w.busy("verify");
    w.idle(); // the primitive ended
    f.at += 100000 * sec;
    try testing.expect(w.check() == null);
    w.tick();
    f.at += 100000 * sec;
    try testing.expect(w.check() == null);
    w.busy("pass");
    f.at += 119 * sec;
    try testing.expect(w.check() == null);
    f.at += sec;
    try testing.expectEqualStrings("pass", w.check().?.name);
}

test "a primitive that ends in time leaves nothing for the next one to inherit" {
    var f: Fake = .{};
    var w = watcher(&f, 120);
    w.busy("verify");
    f.at += 100 * sec;
    w.idle();
    f.at += 50 * sec; // 150 s since the first busy, 50 s since it ended
    w.busy("pass");
    f.at += 119 * sec;
    try testing.expect(w.check() == null);
    f.at += sec;
    try testing.expectEqualStrings("pass", w.check().?.name);
}

test "a limit of 0 is off" {
    var f: Fake = .{};
    var w = watcher(&f, 0);
    w.busy("verify");
    f.at += 100000 * sec;
    try testing.expect(w.check() == null);
}

test "the start-up window has its own limit, and the next ordinary window is back to the primitive limit" {
    var f: Fake = .{};
    var w = watcher(&f, 120);
    w.busyStartup("spill_sync");
    f.at += 599 * sec; // past the primitive limit, inside the start-up one
    try testing.expect(w.check() == null);
    f.at += sec;
    const stall = w.check().?;
    try testing.expectEqualStrings("spill_sync", stall.name);
    try testing.expectEqual(@as(u64, 600), stall.seconds);
    w.idle();
    w.busy("verify");
    f.at += 119 * sec;
    try testing.expect(w.check() == null);
    f.at += sec;
    try testing.expectEqual(@as(u64, 120), w.check().?.seconds);
    w.busyStartup("spill_sync"); // a tick restarts the start-up window at its own limit too
    f.at += 599 * sec;
    w.tick();
    f.at += 599 * sec;
    try testing.expect(w.check() == null);
}

test "a start-up limit of 0 leaves that window unwatched and the others as they are" {
    var f: Fake = .{};
    var w = watcher(&f, 120);
    w.start_ns = 0;
    w.busyStartup("spill_sync");
    f.at += 100000 * sec;
    try testing.expect(w.check() == null);
    w.busy("verify");
    f.at += 120 * sec;
    try testing.expectEqualStrings("verify", w.check().?.name);
    var off = watcher(&f, 0); // the watchdog off is off for the start-up window too
    off.busyStartup("spill_sync");
    f.at += 100000 * sec;
    try testing.expect(off.check() == null);
}

test "the start-up limit's text: unset or empty is 600 s, 0 is unwatched, anything else but whole seconds is refused" {
    try testing.expectEqual(@as(u64, 600), try startLimitFromText(null));
    try testing.expectEqual(@as(u64, 600), try startLimitFromText(""));
    try testing.expectEqual(@as(u64, 0), try startLimitFromText("0"));
    try testing.expectEqual(@as(u64, 1800), try startLimitFromText("1800"));
    for ([_][]const u8{ "x", "-3", "1.5", "10m", " 5" }) |bad| try testing.expectError(error.BadWatchdogStart, startLimitFromText(bad));
}

test "the limit's text: unset or empty is 120 s, 0 is off, anything else but whole seconds is refused" {
    try testing.expectEqual(@as(u64, 120), try limitFromText(null));
    try testing.expectEqual(@as(u64, 120), try limitFromText(""));
    try testing.expectEqual(@as(u64, 0), try limitFromText("0"));
    try testing.expectEqual(@as(u64, 45), try limitFromText("45"));
    for ([_][]const u8{ "x", "-3", "1.5", "12s", " 5" }) |bad| try testing.expectError(error.BadWatchdog, limitFromText(bad));
}

test "the thread starts and stops with nothing stalled" {
    var ctx: u8 = 0;
    const w = try Watchdog.start(testing.allocator, testing.io, .{ .limit_s = 3600 }, &ctx, never);
    w.busy("verify");
    w.tick();
    w.idle();
    Watchdog.stop(w);
}

/// What the thread's fire calls leave behind, for the test that runs it on the real clock.
const Fires = struct {
    n: std.atomic.Value(u32) = .init(0),
    last: []const u8 = "",

    fn fire(ctx: *anyopaque, stall: Stall) void {
        const f: *Fires = @ptrCast(@alignCast(ctx));
        f.last = stall.name;
        _ = f.n.fetchAdd(1, .release);
    }

    fn wait(f: *const Fires, want: u32) !void {
        for (0..1000) |_| { // up to 5 s
            if (f.n.load(.acquire) >= want) return;
            std.Io.sleep(testing.io, .fromMilliseconds(5), .awake) catch {};
        }
        return error.Timeout;
    }

    fn settle(f: *const Fires, want: u32) !void {
        std.Io.sleep(testing.io, .fromMilliseconds(120), .awake) catch {};
        try testing.expectEqual(want, f.n.load(.acquire));
    }
};

test "the thread fires once for a stalled primitive, again after a tick, and not while idle (the real clock)" {
    var fires: Fires = .{};
    const w = try Watchdog.spawn(.{ .gpa = testing.allocator, .io = testing.io, .clock = undefined, .limit_ns = 40 * std.time.ns_per_ms, .fire_ctx = &fires, .fire = Fires.fire, .sleep_ms = 5, .every = 2 });
    w.busy("verify");
    try fires.wait(1);
    try testing.expectEqualStrings("verify", fires.last);
    try fires.settle(1); // the stall is said once
    w.tick();
    try fires.wait(2);
    w.idle();
    try fires.settle(2);
    Watchdog.stop(w);
}

/// A connected pair of stream sockets: a socketpair, or loopback TCP as the ranks' link is (a close is a FIN there).
const Kind = enum { unix, tcp };

fn socketPair(kind: Kind) ![2]std.c.fd_t {
    var pair: [2]std.c.fd_t = undefined;
    if (kind == .unix) {
        if (std.c.socketpair(posix.AF.UNIX, posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
        return pair;
    }
    const listener = std.c.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (listener < 0) return error.Socket;
    defer _ = std.c.close(listener);
    var a: std.c.sockaddr.in = .{ .port = 0, .addr = @bitCast([4]u8{ 127, 0, 0, 1 }) };
    var len: std.c.socklen_t = @sizeOf(std.c.sockaddr.in);
    if (std.c.bind(listener, @ptrCast(&a), len) != 0 or std.c.listen(listener, 1) != 0) return error.Listen;
    if (std.c.getsockname(listener, @ptrCast(&a), &len) != 0) return error.Listen;
    pair[0] = std.c.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (pair[0] < 0) return error.Socket;
    errdefer _ = std.c.close(pair[0]);
    if (std.c.connect(pair[0], @ptrCast(&a), len) != 0) return error.Connect;
    pair[1] = std.c.accept(listener, null, null);
    if (pair[1] < 0) return error.Accept;
    return pair;
}

test "hungUp: an open idle peer is not a hangup; a closed one is once nothing is left to read (socketpair and TCP)" {
    for ([_]Kind{ .unix, .tcp }) |kind| {
        const p = try socketPair(kind);
        defer _ = std.c.close(p[0]);
        const q = try socketPair(kind);
        defer _ = std.c.close(q[0]);
        defer _ = std.c.close(q[1]);
        try testing.expect(!hungUp(&.{}));
        try testing.expect(!hungUp(&.{p[0]})); // open and idle
        _ = std.c.write(p[1], "done", 4);
        try testing.expect(!hungUp(&.{p[0]})); // open, a frame waiting
        _ = std.c.close(p[1]);
        try testing.expect(!hungUp(&.{p[0]})); // closed with the frame unread: its reader goes first
        var buf: [4]u8 = undefined;
        try testing.expectEqual(@as(isize, 4), std.c.read(p[0], &buf, 4));
        try testing.expect(hungUp(&.{p[0]})); // read: nothing left
        try testing.expect(hungUp(&.{ q[0], p[0] })); // any of the links
        try testing.expect(!hungUp(&.{q[0]}));
    }
}

/// What the hangup calls leave behind, for the tests that run the thread on the real clock.
const Hangups = struct {
    n: std.atomic.Value(u32) = .init(0),

    fn gone(ctx: *anyopaque) void {
        const h: *Hangups = @ptrCast(@alignCast(ctx));
        _ = h.n.fetchAdd(1, .release);
    }

    fn wait(h: *const Hangups, want: u32) !void {
        for (0..1000) |_| { // up to 5 s
            if (h.n.load(.acquire) >= want) return;
            std.Io.sleep(testing.io, .fromMilliseconds(5), .awake) catch {};
        }
        return error.Timeout;
    }

    fn settle(h: *const Hangups, want: u32) !void {
        std.Io.sleep(testing.io, .fromMilliseconds(120), .awake) catch {};
        try testing.expectEqual(want, h.n.load(.acquire));
    }
};

/// A watchdog on the real clock, paced for a test, that watches the link `fd`.
fn watching(fd: std.c.fd_t, h: *Hangups, sleep_ms: u64, every: u32) !*Watchdog {
    var t: Watchdog = .{ .gpa = testing.allocator, .io = testing.io, .clock = undefined, .limit_ns = 3600 * sec, .fire_ctx = h, .fire = never, .hangup = Hangups.gone, .sleep_ms = sleep_ms, .every = every };
    t.peers[0] = fd;
    t.npeers = 1;
    return Watchdog.spawn(t);
}

test "the thread calls the hangup once for a peer that closed, never for an open idle one, and waits for unread frames" {
    const idle = try socketPair(.unix);
    defer _ = std.c.close(idle[0]);
    defer _ = std.c.close(idle[1]);
    const shut = try socketPair(.unix);
    defer _ = std.c.close(shut[0]);
    var quiet: Hangups = .{};
    var loud: Hangups = .{};
    const a = try watching(idle[0], &quiet, 5, 2);
    defer Watchdog.stop(a);
    const b = try watching(shut[0], &loud, 5, 2);
    defer Watchdog.stop(b);
    try loud.settle(0); // a dozen polls with both peers open
    _ = std.c.write(shut[1], "done", 4);
    _ = std.c.close(shut[1]);
    try loud.settle(0); // closed with a frame unread: the thread leaves it to its reader
    var buf: [4]u8 = undefined;
    _ = std.c.read(shut[0], &buf, 4);
    try loud.wait(1);
    try loud.settle(1); // said once
    try quiet.settle(0);
}

test "at its own pace the thread sees a peer's close within about a second (loopback TCP, the link's kind)" {
    const p = try socketPair(.tcp);
    defer _ = std.c.close(p[0]);
    var h: Hangups = .{};
    const w = try Watchdog.start(testing.allocator, testing.io, .{ .limit_s = 3600, .peers = &.{p[0]}, .hangup = Hangups.gone }, &h, never);
    defer Watchdog.stop(w);
    const t0 = std.Io.Timestamp.now(testing.io, .awake).nanoseconds;
    _ = std.c.close(p[1]);
    try h.wait(1);
    const waited_ms = @divTrunc(std.Io.Timestamp.now(testing.io, .awake).nanoseconds - t0, std.time.ns_per_ms);
    try testing.expect(waited_ms <= 2500); // one period, 1 s, and a margin for a busy machine
}
