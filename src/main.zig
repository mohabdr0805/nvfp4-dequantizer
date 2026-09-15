const std = @import("std");
const nvp4 = @import("nvfp4.zig");
const safetensors = @import("safetensors.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;
    const arena = init.arena.allocator();

    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) {
        std.debug.print("Erreur : Veuillez passer au moins un argument.\n", .{});
        return;
    }

    const safetensor_file = try safetensors.open(io, gpa, args[1]);
    defer safetensor_file.deinit(io);

    std.debug.print("count : {d}\n", .{safetensor_file.offset});

    const values = safetensor_file.parsed.value;

    const count = values.object.count();

    const obj = values.object;

    var compteurs = std.enums.EnumArray(safetensors.Dtype, u64).initFill(0);

    for (obj.values()) |v| {
        if (v.object.get("dtype")) |d| {
            if (std.meta.stringToEnum(safetensors.Dtype, d.string)) |ds| {
                compteurs.getPtr(ds).* += 1;
            } else std.debug.print("Type non reconnu : {s}\n", .{d.string});
        }
    }

    var it = compteurs.iterator();
    while (it.next()) |e|
        std.debug.print("{s:<9} {d:>3}\n", .{ @tagName(e.key), e.value.* });

    std.debug.print("total : {d}", .{count});
}
