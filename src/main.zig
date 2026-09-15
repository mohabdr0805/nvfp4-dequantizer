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

    const object_map = values.object;

    var counts = std.enums.EnumArray(safetensors.Dtype, u64).initFill(0);

    for (object_map.values()) |v| {
        if (v.object.get("dtype")) |d| {
            if (std.meta.stringToEnum(safetensors.Dtype, d.string)) |ds| {
                counts.getPtr(ds).* += 1;
            } else std.debug.print("Type non reconnu : {s}\n", .{d.string});
        }
    }

    //for (object_map.keys(), object_map.values()) |k, v| {
    //    if (v.object.get("dtype")) |d| {
    //        std.debug.print("{s} -> {f}\n", .{ k, std.json.fmt(v, .{}) });
    //        _ = d;
    //    }
    //}

    var it = counts.iterator();
    while (it.next()) |e|
        std.debug.print("{s:<9} {d:>3}\n", .{ @tagName(e.key), e.value.* });

    std.debug.print("total : {d}\n", .{count});

    const out = try safetensors.layout(gpa, object_map);
    defer gpa.free(out);

    std.debug.print("out len : {d}\n", .{out.len});

    const end_file = out[out.len - 1].end;
    std.debug.print("size from out : {d}\n", .{end_file});

    const file: std.Io.File = try std.Io.Dir.createFile(.cwd(), io, args[2], .{});
    var buf: [64 * 1024]u8 = undefined;
    var writer = file.writer(io, &buf);

    try safetensors.writeHeader(gpa, &writer.interface, out);
    try writer.flush();
    //defer safetensor_file.deinit(io);

}
