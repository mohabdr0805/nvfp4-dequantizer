const std = @import("std");

pub const Dtype = enum { U8, F8_E4M3, F32, BF16 };

pub const SafetensorsFile = struct {
    const Self = @This();
    gpa: std.mem.Allocator,
    file: std.Io.File,
    data: []const u8,
    parsed: std.json.Parsed(std.json.Value),
    offset: u64,

    pub fn deinit(self: *const Self, io: std.Io) void {
        self.file.close(io);
        self.parsed.deinit();
        self.gpa.free(self.data);
    }
};

pub fn open_tensor(io: std.Io, gpa: std.mem.Allocator, filename: []const u8) ![]const u8 {
    const file: std.Io.File = try std.Io.Dir.openFile(.cwd(), io, filename, .{ .mode = .read_only });
    defer file.close(io);

    var buffer: [4096]u8 = undefined;
    var reader = file.reader(io, &buffer);

    const n = (try file.stat(io)).size;

    const t = try reader.interface.readAlloc(gpa, n);

    return t;
}

pub fn open(io: std.Io, gpa: std.mem.Allocator, filename: []const u8) !SafetensorsFile {
    const file: std.Io.File = try std.Io.Dir.openFile(.cwd(), io, filename, .{ .mode = .read_only });
    var buffer: [4096]u8 = undefined;
    var reader = file.reader(io, &buffer);
    const n = try reader.interface.takeInt(u64, .little);
    const t = try reader.interface.readAlloc(gpa, n);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, t, .{});

    const safetensors_file = SafetensorsFile{
        .gpa = gpa,
        .file = file,
        .data = t,
        .parsed = parsed,
        .offset = 8 + n,
    };

    return safetensors_file;
}
