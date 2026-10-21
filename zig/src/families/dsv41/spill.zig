//! One rank's disk tier of kept prompt states: slot parts, compressed rows in blocks named by tokens, O_DIRECT.
const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

/// A block's name: two independent 64-bit hash chains over the tokens up to the block's end.
pub const Name = [2]u64;

/// Offsets, lengths and buffers of a direct file read or write (the NVMe's logical block).
pub const page = 4096;
const VERSION: u32 = 1;
const BLOCK_MAGIC: u32 = 0x4b425344; // "DSBK"
const SLOT_MAGIC: u32 = 0x50535344; // "DSSP"

/// A plane the blocks carry: rows of `row` bytes from device address `addr`, one row every `ratio` positions.
pub const Plane = struct { addr: u64, ratio: u32, row: u32 };

/// Copies on the rank's stream between pinned staging and the device, and the wait that ends them.
pub const Dev = struct {
    ctx: *anyopaque,
    to_host: *const fn (ctx: *anyopaque, dst: []u8, src: u64) anyerror!void,
    to_device: *const fn (ctx: *anyopaque, dst: u64, src: []const u8) anyerror!void,
    wait: *const fn (ctx: *anyopaque) anyerror!void,
};

const BlockHead = extern struct { magic: u32, version: u32, name: Name, bytes: u64, sum: u64 };
const SlotHead = extern struct { magic: u32, version: u32, key: u64, at: u64, blocks: u64, slot: u64, sum: u64 };

/// The names of the whole blocks below `at`: block k's chains run over tokens [0, (k + 1) * block).
pub fn names(gpa: Allocator, tokens: []const u32, at: usize, block: usize) ![]Name {
    const n = @min(at, tokens.len) / block;
    const out = try gpa.alloc(Name, n);
    var a: u64 = 0x5b10c;
    var b: u64 = 0xb10c5;
    for (out, 0..) |*x, k| {
        const bytes = std.mem.sliceAsBytes(tokens[k * block ..][0..block]);
        a = std.hash.Wyhash.hash(a, bytes);
        b = std.hash.XxHash3.hash(b, bytes);
        x.* = .{ a, b };
    }
    return out;
}

pub const Spill = struct {
    gpa: Allocator,
    planes: []const Plane,
    block: usize, // positions a block holds (the prompt chunk)
    slot_bytes: usize,
    dev: Dev,
    stage: []align(page) u8, // pinned, a whole block file at least
    direct: bool = builtin.os.tag == .linux, // O_DIRECT (the staging is page aligned)
    floor: u64 = 50 << 30, // free bytes a write leaves on the file system at least
    dir: ?[:0]u8 = null,
    states: std.AutoHashMapUnmanaged(u64, []Name) = .empty,
    refs: std.AutoHashMapUnmanaged(Name, u32) = .empty,
    bad: std.AutoHashMapUnmanaged(Name, void) = .empty, // named blocks whose file failed its read (gone until rewritten)
    // what the last write and read moved (for their log lines)
    wrote_blocks: usize = 0,
    read_bytes: u64 = 0,

    /// A tier over `planes` (copied); `open` names its directory. The staging must hold a block file.
    pub fn init(gpa: Allocator, planes: []const Plane, block: usize, slot_bytes: usize, dev: Dev, stage: []align(page) u8) !Spill {
        for (planes) |p| if (p.ratio == 0 or block % p.ratio != 0) return error.BlockNotRowAligned;
        var sp: Spill = .{ .gpa = gpa, .planes = try gpa.dupe(Plane, planes), .block = block, .slot_bytes = slot_bytes, .dev = dev, .stage = stage };
        if (stage.len < sp.blockFile() or stage.len % page != 0) {
            gpa.free(sp.planes);
            return error.StagingTooSmall;
        }
        return sp;
    }

    pub fn deinit(sp: *Spill) void {
        var it = sp.states.valueIterator();
        while (it.next()) |v| sp.gpa.free(v.*);
        sp.states.deinit(sp.gpa);
        sp.refs.deinit(sp.gpa);
        sp.bad.deinit(sp.gpa);
        if (sp.dir) |d| sp.gpa.free(d);
        sp.gpa.free(sp.planes);
    }

    /// A block's payload: every plane's rows of `block` positions.
    pub fn blockBytes(sp: *const Spill) usize {
        var n: usize = 0;
        for (sp.planes) |p| n += sp.block / p.ratio * p.row;
        return n;
    }

    fn blockFile(sp: *const Spill) usize {
        return page + std.mem.alignForward(usize, sp.blockBytes(), page);
    }

    fn slotHead(blocks: usize) usize {
        return std.mem.alignForward(usize, @sizeOf(SlotHead) + blocks * @sizeOf(Name), page);
    }

    /// The disk a state at `at` takes at most (its blocks counted whole, as if no other state shared them).
    pub fn diskBytes(sp: *const Spill, at: usize) u64 {
        const n = at / sp.block;
        return slotHead(n) + std.mem.alignForward(usize, sp.slot_bytes, page) + n * sp.blockFile();
    }

    pub fn has(sp: *const Spill, key: u64) bool {
        return sp.states.contains(key);
    }

    /// The tier in `dir` (made when missing): every slot file's state read back, temporaries and unnamed blocks gone.
    pub fn open(sp: *Spill, dir: []const u8) !void {
        if (sp.dir) |d| if (std.mem.eql(u8, d, dir)) return;
        const d = try sp.gpa.dupeSentinel(u8, dir, 0);
        if (sp.dir) |old| sp.gpa.free(old);
        sp.dir = d;
        _ = std.c.mkdir(d, 0o700); // (the imprint made it on rank 0)
        const h = std.c.opendir(d) orelse return error.SpillDir;
        defer _ = std.c.closedir(h);
        var blocks: std.ArrayList(Name) = .empty;
        defer blocks.deinit(sp.gpa);
        while (std.c.readdir(h)) |ent| {
            const name = std.mem.sliceTo(&ent.name, 0);
            if (std.mem.endsWith(u8, name, ".part")) {
                sp.unlinkName(name);
            } else if (std.mem.endsWith(u8, name, ".slot") and name.len == 21) {
                const key = std.fmt.parseInt(u64, name[0..16], 16) catch continue;
                sp.loadSlot(key) catch sp.unlinkName(name);
            } else if (std.mem.endsWith(u8, name, ".blk") and name.len == 36) {
                const a = std.fmt.parseInt(u64, name[0..16], 16) catch continue;
                const b = std.fmt.parseInt(u64, name[16..32], 16) catch continue;
                try blocks.append(sp.gpa, .{ a, b });
            }
        }
        for (blocks.items) |n| if (!sp.refs.contains(n)) {
            var buf: [1200]u8 = undefined;
            _ = std.c.unlink(sp.blockPath(&buf, n) catch continue);
        };
    }

    /// A slot file's header and names into the state table (a buffered read: a page or two).
    fn loadSlot(sp: *Spill, key: u64) !void {
        var buf: [1200]u8 = undefined;
        const fd = std.c.open(try sp.slotPath(&buf, key, false), .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.SpillRead;
        defer _ = std.c.close(fd);
        var head: SlotHead = undefined;
        if (!readAt(fd, std.mem.asBytes(&head), 0) or head.magic != SLOT_MAGIC or head.version != VERSION or head.key != key) return error.SpillHeader;
        if (head.slot != sp.slot_bytes or head.blocks != head.at / sp.block or head.blocks > 1 << 20) return error.SpillHeader;
        const nm = try sp.gpa.alloc(Name, head.blocks);
        errdefer sp.gpa.free(nm);
        if (!readAt(fd, std.mem.sliceAsBytes(nm), @sizeOf(SlotHead))) return error.SpillHeader;
        try sp.register(key, nm);
    }

    fn register(sp: *Spill, key: u64, nm: []Name) !void {
        try sp.states.ensureUnusedCapacity(sp.gpa, 1);
        try sp.refs.ensureUnusedCapacity(sp.gpa, @intCast(nm.len));
        for (nm) |n| {
            const r = sp.refs.getOrPutAssumeCapacity(n);
            r.value_ptr.* = if (r.found_existing) r.value_ptr.* + 1 else 1;
        }
        if (sp.states.fetchRemove(key)) |old| sp.release(old.value); // after the new names: blocks both name stay
        sp.states.putAssumeCapacity(key, nm);
    }

    /// A state's names given back: blocks no other state names are deleted.
    fn release(sp: *Spill, nm: []Name) void {
        for (nm) |n| if (sp.refs.getPtr(n)) |r| {
            r.* -= 1;
            if (r.* > 0) continue;
            _ = sp.refs.remove(n);
            _ = sp.bad.remove(n);
            var buf: [1200]u8 = undefined;
            _ = std.c.unlink(sp.blockPath(&buf, n) catch continue);
        };
        sp.gpa.free(nm);
    }

    /// State `key` gone from this rank's disk (its slot file, and its blocks no other state names).
    pub fn forget(sp: *Spill, key: u64) void {
        const kv = sp.states.fetchRemove(key) orelse return;
        var buf: [1200]u8 = undefined;
        if (sp.slotPath(&buf, key, false)) |p| {
            _ = std.c.unlink(p);
        } else |_| {}
        sp.release(kv.value);
    }

    /// Every state but `keys` forgotten (the index rank 0 holds).
    pub fn keepOnly(sp: *Spill, keys: []const u64) !void {
        var gone: std.ArrayList(u64) = .empty;
        defer gone.deinit(sp.gpa);
        var it = sp.states.keyIterator();
        while (it.next()) |k| if (std.mem.indexOfScalar(u64, keys, k.*) == null) try gone.append(sp.gpa, k.*);
        for (gone.items) |k| sp.forget(k);
    }

    /// Whether state `key` of `at` positions with these names is on this rank's disk, every block it names too.
    pub fn check(sp: *const Spill, key: u64, at: usize, nm: []const Name) bool {
        const have = sp.states.get(key) orelse return false;
        if (at / sp.block != have.len or !std.mem.eql(u8, std.mem.sliceAsBytes(have), std.mem.sliceAsBytes(nm))) return false;
        for (nm) |n| if (!sp.refs.contains(n) or sp.bad.contains(n)) return false;
        return true;
    }

    /// State `key`: its missing blocks from the extent at `base`, then its slot part from device `slot_src`.
    pub fn write(sp: *Spill, key: u64, at: usize, nm: []const Name, base: usize, slot_src: u64) !void {
        const dir = sp.dir orelse return error.SpillClosed;
        if (at / sp.block != nm.len or at % sp.block != 0) return error.NotABlockEnd;
        if (sp.check(key, at, nm)) return; // on disk already
        if (freeBytes(dir) < sp.diskBytes(at) + sp.floor) return error.SpillDiskFull;
        const owned = try sp.gpa.dupe(Name, nm);
        errdefer sp.gpa.free(owned);
        const per = sp.blockFile();
        const fit = sp.stage.len / per;
        var todo: std.ArrayList(usize) = .empty;
        defer todo.deinit(sp.gpa);
        for (nm, 0..) |n, k| if (!sp.refs.contains(n) or sp.bad.contains(n)) try todo.append(sp.gpa, k);
        sp.wrote_blocks = todo.items.len;
        var i: usize = 0;
        while (i < todo.items.len) : (i += fit) {
            const batch = todo.items[i..@min(todo.items.len, i + fit)];
            for (batch, 0..) |k, j| try sp.pull(sp.stage[j * per + page ..][0 .. per - page], base + k * sp.block);
            try sp.dev.wait(sp.dev.ctx);
            for (batch, 0..) |k, j| {
                try sp.writeBlock(nm[k], sp.stage[j * per ..][0..per]);
                _ = sp.bad.remove(nm[k]);
            }
        }
        try sp.writeSlot(key, at, nm, slot_src);
        syncDir(dir);
        try sp.register(key, owned);
    }

    /// The rows of the block at `pos` (every plane's, in order) into `out` (the rest zero).
    fn pull(sp: *Spill, out: []u8, pos: usize) !void {
        var off: usize = 0;
        for (sp.planes) |p| {
            const n = sp.block / p.ratio * p.row;
            try sp.dev.to_host(sp.dev.ctx, out[off..][0..n], p.addr + pos / p.ratio * p.row);
            off += n;
        }
        @memset(out[off..], 0);
    }

    fn push(sp: *Spill, rows: []const u8, pos: usize) !void {
        var off: usize = 0;
        for (sp.planes) |p| {
            const n = sp.block / p.ratio * p.row;
            try sp.dev.to_device(sp.dev.ctx, p.addr + pos / p.ratio * p.row, rows[off..][0..n]);
            off += n;
        }
    }

    fn writeBlock(sp: *Spill, n: Name, file: []u8) !void {
        const len = sp.blockBytes();
        const head: BlockHead = .{ .magic = BLOCK_MAGIC, .version = VERSION, .name = n, .bytes = len, .sum = std.hash.XxHash3.hash(0, file[page..][0..len]) };
        @memset(file[0..page], 0);
        @memcpy(file[0..@sizeOf(BlockHead)], std.mem.asBytes(&head));
        var tmp: [1200]u8 = undefined;
        var fin: [1200]u8 = undefined;
        const part = try std.fmt.bufPrintSentinel(&tmp, "{s}.part", .{try sp.blockPath(&fin, n)}, 0);
        const fd = try sp.create(part);
        defer _ = std.c.close(fd);
        errdefer _ = std.c.unlink(part);
        try pwriteAll(fd, file, 0);
        if (std.c.fsync(fd) != 0) return error.SpillWrite;
        if (std.c.rename(part, try sp.blockPath(&fin, n)) != 0) return error.SpillWrite;
    }

    fn writeSlot(sp: *Spill, key: u64, at: usize, nm: []const Name, src: u64) !void {
        var tmp: [1200]u8 = undefined;
        var fin: [1200]u8 = undefined;
        const part = try sp.slotPath(&tmp, key, true);
        const fd = try sp.create(part);
        defer _ = std.c.close(fd);
        errdefer _ = std.c.unlink(part);
        const head_len = slotHead(nm.len);
        if (head_len > sp.stage.len) return error.StagingTooSmall;
        var sum = std.hash.XxHash3.init(0);
        var off: usize = 0;
        while (off < sp.slot_bytes) {
            const n = @min(sp.stage.len, sp.slot_bytes - off);
            try sp.dev.to_host(sp.dev.ctx, sp.stage[0..n], src + off);
            try sp.dev.wait(sp.dev.ctx);
            sum.update(sp.stage[0..n]);
            const padded = std.mem.alignForward(usize, n, page);
            @memset(sp.stage[n..padded], 0);
            try pwriteAll(fd, sp.stage[0..padded], head_len + off);
            off += n;
        }
        const head: SlotHead = .{ .magic = SLOT_MAGIC, .version = VERSION, .key = key, .at = at, .blocks = nm.len, .slot = sp.slot_bytes, .sum = sum.final() };
        const region = sp.stage[0..head_len];
        @memset(region, 0);
        @memcpy(region[0..@sizeOf(SlotHead)], std.mem.asBytes(&head));
        @memcpy(region[@sizeOf(SlotHead)..][0 .. nm.len * @sizeOf(Name)], std.mem.sliceAsBytes(nm));
        try pwriteAll(fd, region, 0);
        if (std.c.fsync(fd) != 0) return error.SpillWrite;
        if (std.c.rename(part, try sp.slotPath(&fin, key, false)) != 0) return error.SpillWrite;
    }

    /// State `key` back: blocks into the extent at `base`, slot part into `slot_dst` (null: held); a bad file fails it.
    pub fn read(sp: *Spill, key: u64, at: usize, nm: []const Name, base: usize, slot_dst: ?u64) !void {
        if (!sp.check(key, at, nm)) return error.SpillMissing;
        sp.read_bytes = 0;
        const per = sp.blockFile();
        const fit = sp.stage.len / per;
        var k: usize = 0;
        while (k < nm.len) : (k += fit) {
            const end = @min(nm.len, k + fit);
            for (k..end) |b| sp.readBlock(nm[b], sp.stage[(b - k) * per ..][0..per]) catch |err| {
                sp.dropBlock(nm[b]);
                return err;
            };
            for (k..end) |b| try sp.push(sp.stage[(b - k) * per + page ..][0..sp.blockBytes()], base + b * sp.block);
            try sp.dev.wait(sp.dev.ctx);
            sp.read_bytes += (end - k) * per;
        }
        if (slot_dst) |dst| sp.readSlot(key, at, nm.len, dst) catch |err| {
            sp.forget(key);
            return err;
        };
    }

    fn readBlock(sp: *Spill, n: Name, file: []u8) !void {
        var buf: [1200]u8 = undefined;
        const fd = try sp.openRead(try sp.blockPath(&buf, n));
        defer _ = std.c.close(fd);
        if (!readAt(fd, file, 0)) return error.SpillRead;
        const head = std.mem.bytesToValue(BlockHead, file[0..@sizeOf(BlockHead)]);
        const len = sp.blockBytes();
        if (head.magic != BLOCK_MAGIC or head.version != VERSION or head.name[0] != n[0] or head.name[1] != n[1] or head.bytes != len) return error.SpillBadBlock;
        if (std.hash.XxHash3.hash(0, file[page..][0..len]) != head.sum) return error.SpillBadBlock;
    }

    /// A block that does not read back: its file gone and every state naming it unable to check until it is rewritten.
    fn dropBlock(sp: *Spill, n: Name) void {
        sp.bad.put(sp.gpa, n, {}) catch {
            _ = sp.refs.remove(n); // out of memory: unnamed instead (a later release skips it)
        };
        var buf: [1200]u8 = undefined;
        _ = std.c.unlink(sp.blockPath(&buf, n) catch return);
    }

    fn readSlot(sp: *Spill, key: u64, at: usize, blocks: usize, dst: u64) !void {
        var buf: [1200]u8 = undefined;
        const fd = try sp.openRead(try sp.slotPath(&buf, key, false));
        defer _ = std.c.close(fd);
        const head_len = slotHead(blocks);
        if (head_len > sp.stage.len) return error.StagingTooSmall;
        if (!readAt(fd, sp.stage[0..head_len], 0)) return error.SpillRead;
        const head = std.mem.bytesToValue(SlotHead, sp.stage[0..@sizeOf(SlotHead)]);
        if (head.magic != SLOT_MAGIC or head.version != VERSION or head.key != key or head.at != at or head.blocks != blocks or head.slot != sp.slot_bytes) return error.SpillHeader;
        var sum = std.hash.XxHash3.init(0);
        var off: usize = 0;
        while (off < sp.slot_bytes) {
            const n = @min(sp.stage.len, sp.slot_bytes - off);
            if (!readAt(fd, sp.stage[0..std.mem.alignForward(usize, n, page)], head_len + off)) return error.SpillRead;
            sum.update(sp.stage[0..n]);
            try sp.dev.to_device(sp.dev.ctx, dst + off, sp.stage[0..n]);
            try sp.dev.wait(sp.dev.ctx);
            off += n;
        }
        sp.read_bytes += head_len + std.mem.alignForward(usize, sp.slot_bytes, page);
        if (sum.final() != head.sum) return error.SpillBadSlot;
    }

    fn blockPath(sp: *const Spill, buf: []u8, n: Name) ![:0]const u8 {
        return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}{x:0>16}.blk", .{ sp.dir orelse return error.SpillClosed, n[0], n[1] }, 0);
    }

    fn slotPath(sp: *const Spill, buf: []u8, key: u64, part: bool) ![:0]const u8 {
        return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}.slot{s}", .{ sp.dir orelse return error.SpillClosed, key, if (part) ".part" else "" }, 0);
    }

    fn unlinkName(sp: *const Spill, name: []const u8) void {
        var buf: [1200]u8 = undefined;
        _ = std.c.unlink(std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ sp.dir.?, name }, 0) catch return);
    }

    fn create(sp: *const Spill, path: [:0]const u8) !std.c.fd_t {
        if (builtin.os.tag == .linux and sp.direct) {
            const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true, .DIRECT = true }, @as(std.c.mode_t, 0o600));
            if (fd >= 0) return fd;
        }
        const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
        return if (fd < 0) error.SpillWrite else fd;
    }

    fn openRead(sp: *const Spill, path: [:0]const u8) !std.c.fd_t {
        if (builtin.os.tag == .linux and sp.direct) {
            const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true, .DIRECT = true }, @as(std.c.mode_t, 0));
            if (fd >= 0) return fd;
        }
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        return if (fd < 0) error.SpillMissing else fd;
    }
};

fn pwriteAll(fd: std.c.fd_t, bytes: []const u8, at: usize) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.pwrite(fd, bytes.ptr + done, bytes.len - done, @intCast(at + done));
        if (n < 0 and std.c.errno(n) == .INTR) continue;
        if (n <= 0) return error.SpillWrite;
        done += @intCast(n);
    }
}

fn readAt(fd: std.c.fd_t, dest: []u8, at: usize) bool {
    var done: usize = 0;
    while (done < dest.len) {
        const n = std.c.pread(fd, dest.ptr + done, dest.len - done, @intCast(at + done));
        if (n < 0 and std.c.errno(n) == .INTR) continue;
        if (n <= 0) return false;
        done += @intCast(n);
    }
    return true;
}

fn syncDir(dir: [:0]const u8) void {
    const fd = std.c.open(dir, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return;
    _ = std.c.fsync(fd);
    _ = std.c.close(fd);
}

extern "c" fn statvfs(path: [*:0]const u8, buf: *[16]u64) c_int;

/// Bytes free to an unprivileged writer on `dir`'s file system (Linux; elsewhere unbounded).
fn freeBytes(dir: [:0]const u8) u64 {
    if (builtin.os.tag != .linux) return std.math.maxInt(u64) / 2;
    var v: [16]u64 = @splat(0); // glibc's struct statvfs on 64-bit Linux: f_bsize, f_frsize, f_blocks, f_bfree, f_bavail
    if (statvfs(dir.ptr, &v) != 0) return 0;
    return v[4] *| v[1];
}

/// A rank's identity for its tier's directory: everything that sets the bits of a kept state, hashed.
pub const Identity = struct {
    h: std.hash.Wyhash = .init(0xd5_41_5b),

    pub fn word(id: *Identity, x: u64) void {
        id.h.update(std.mem.asBytes(&x));
    }

    pub fn text(id: *Identity, s: []const u8) void {
        id.word(s.len);
        id.h.update(s);
    }

    /// A file's bytes (a missing one counts as its name only).
    pub fn content(id: *Identity, path: []const u8) void {
        id.text(path);
        var buf: [1200]u8 = undefined;
        const fd = std.c.open(std.fmt.bufPrintSentinel(&buf, "{s}", .{path}, 0) catch return, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return id.word(0);
        defer _ = std.c.close(fd);
        var chunk: [1 << 16]u8 = undefined;
        while (true) {
            const n = std.c.read(fd, &chunk, chunk.len);
            if (n <= 0) break;
            id.h.update(chunk[0..@intCast(n)]);
        }
    }

    /// A large file (weights, tables): its size and 16 spans of 64 KiB spread over it, not its whole bytes.
    pub fn sample(id: *Identity, path: []const u8) void {
        id.text(path);
        var buf: [1200]u8 = undefined;
        const fd = std.c.open(std.fmt.bufPrintSentinel(&buf, "{s}", .{path}, 0) catch return, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return id.word(0);
        defer _ = std.c.close(fd);
        const end = std.c.lseek(fd, 0, std.c.SEEK.END);
        const size: u64 = if (end > 0) @intCast(end) else 0;
        id.word(size);
        var span: [1 << 16]u8 = undefined;
        for (0..16) |i| {
            const at = (size -| span.len) / 15 * i;
            const n = std.c.pread(fd, &span, span.len, @intCast(at));
            if (n > 0) id.h.update(span[0..@intCast(n)]);
        }
    }

    /// Every regular file directly in `dir`, by name order: its bytes (`whole`) or its sample.
    pub fn tree(id: *Identity, gpa: Allocator, dir: []const u8, whole: bool) !void {
        var buf: [1200]u8 = undefined;
        const d = std.c.opendir(try std.fmt.bufPrintSentinel(&buf, "{s}", .{dir}, 0)) orelse return id.text(dir);
        var list: std.ArrayList([]u8) = .empty;
        defer {
            for (list.items) |x| gpa.free(x);
            list.deinit(gpa);
        }
        while (std.c.readdir(d)) |ent| {
            if (ent.type != std.c.DT.REG) continue;
            try list.append(gpa, try gpa.dupe(u8, std.mem.sliceTo(&ent.name, 0)));
        }
        _ = std.c.closedir(d);
        std.mem.sort([]u8, list.items, {}, struct {
            fn lt(_: void, a: []u8, b: []u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lt);
        for (list.items) |name| {
            const p = try std.fmt.bufPrint(&buf, "{s}/{s}", .{ dir, name });
            if (whole) id.content(p) else id.sample(p);
        }
    }

    pub fn final(id: *Identity) u64 {
        return id.h.final();
    }
};

test {
    _ = @import("spill_test.zig");
}
