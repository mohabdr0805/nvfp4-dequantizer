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

    std.debug.print("{s}\n", .{t[0..200]});

    // A TOI : parser t, compter les entrees, afficher dtype et shape
    //          de "model.layers.0.self_attn.q_proj.weight"
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, t, .{});
    defer parsed.deinit();

    const values = parsed.value;

    //var it = values.object.iterator();
    const count = values.object.count();

    const v_tensor = values.object.get("model.layers.0.self_attn.q_proj.weight").?;
    std.debug.print("tensors : {f}\n", .{std.json.fmt(v_tensor, .{ .whitespace = .indent_2 })});

    const tensor = v_tensor.object;

    const v_dtype = tensor.get("dtype").?;
    const dtype = v_dtype.string;

    const shape = tensor.get("shape").?.array.items;
    const data_offset = tensor.get("data_offsets").?.array.items;

    std.debug.print("count {d}, dtype {s}, shape ({d},{d}), data_offset({d},{d})", .{ count, dtype, shape[0].integer, shape[1].integer, data_offset[0].integer, data_offset[1].integer });
}
