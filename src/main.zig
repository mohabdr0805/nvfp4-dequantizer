const std = @import("std");
const nvp4 = @import("nvfp4.zig");
const safetensors = @import("safetensors.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;
    const arena = init.arena.allocator();

    // Default values
    var n_workers: u64 = 1;
    var mode = safetensors.Mode.full;

    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 3) {
        std.debug.print("Error : Please provide input and output file.\n", .{});
        return;
    }

    if (args.len > 3) {
        n_workers = try std.fmt.parseInt(u32, args[3], 10);
    }
    if (args.len > 4) {
        mode = std.meta.stringToEnum(safetensors.Mode, args[4]).?;
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

    var it = counts.iterator();
    while (it.next()) |e|
        std.debug.print("{s:<9} {d:>3}\n", .{ @tagName(e.key), e.value.* });

    std.debug.print("total :  {d}\n", .{count});

    const out = try safetensors.layout(gpa, object_map);
    defer gpa.free(out);

    const file: std.Io.File = try std.Io.Dir.createFile(.cwd(), io, args[2], .{});
    var buf: [8 * 1024 * 1024]u8 = undefined;

    var writer = file.writer(io, &buf);
    const sink = &writer.interface;

    const t0 = std.Io.Clock.awake.now(io);
    if (mode == .write) {
        try safetensors.writeOnly(gpa, sink, out);
    } else {
        const write_offset = try safetensors.writeHeader(gpa, sink, out);
        try safetensors.writeDecode(io, gpa, args[1], args[2], write_offset, out, n_workers, mode);
    }
    try sink.flush();

    const t1 = std.Io.Clock.awake.now(io);
    const us = std.Io.Timestamp.durationTo(t0, t1).toMicroseconds();
    const s = @as(f64, @floatFromInt(us)) / 1e6;

    const end_file = out[out.len - 1].out_end;
    std.debug.print("temps : {d:.2} s   debit : {d:.2} Go/s\n", .{
        s,
        @as(f64, @floatFromInt(end_file)) / 1e9 / s,
    });
}
