const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const io = init.io; // Zig te fournit l'Io et l'allocateur
    const gpa = init.gpa;

    const path: []const u8 = "reference/entete-reelle.safetensors";

    const file: std.Io.File = try std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only });
    defer file.close(io);

    var buffer: [4096]u8 = undefined;

    var reader = file.reader(io, &buffer);

    const n = try reader.interface.takeInt(u64, .little);

    std.debug.print("n = {d}\n", .{n});

    const t = try reader.interface.readAlloc(gpa, n);
    defer gpa.free(t);

    std.debug.print("t {s}", .{t[0..200]});
}
