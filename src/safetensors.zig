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

const OutputTensor = struct {
    name: []const u8,
    dtype: Dtype,
    rank: u8,
    shape: [4]u64,
    in_start: u64,
    in_end: u64,
    offset_global: ?[2]u64,
    offset_partial: ?[2]u64,
    out_start: u64,
    out_end: u64,
};

pub fn readWholeFile(io: std.Io, gpa: std.mem.Allocator, filename: []const u8) ![]const u8 {
    const file: std.Io.File = try std.Io.Dir.openFile(.cwd(), io, filename, .{ .mode = .write_only });
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

fn readPair(obj: std.json.Value, key: []const u8) [2]u64 {
    var t: [2]u64 = .{ 1, 1 };
    for (0..2) |i| t[i] = @intCast(obj.object.get(key).?.array.items[i].integer);
    return t;
}

pub fn layout(gpa: std.mem.Allocator, object_map: std.json.ObjectMap) ![]OutputTensor {
    var res: std.ArrayList(OutputTensor) = .empty;

    var cursor: u64 = 0;
    for (object_map.keys(), object_map.values()) |k, v| {
        if (!std.mem.endsWith(u8, k, "weight_scale") and !std.mem.endsWith(u8, k, "weight_scale_2")) {
            if (v.object.get("dtype")) |d| {
                const in_offsets = readPair(v, "data_offsets");
                var dtype = std.meta.stringToEnum(Dtype, d.string).?;
                const rank: u8 = @intCast(v.object.get("shape").?.array.items.len);
                var shape = blk: {
                    var t: [4]u64 = .{ 1, 1, 1, 1 };
                    for (0..v.object.get("shape").?.array.items.len) |i| t[i] = @intCast(v.object.get("shape").?.array.items[i].integer);
                    break :blk t;
                };
                var offset_global: ?[2]u64 = null;
                var offset_partial: ?[2]u64 = null;
                if (dtype == .U8) {
                    dtype = .F32;
                    shape[rank - 1] = shape[rank - 1] * 2;
                    var buffer: [100]u8 = undefined;
                    var name = try std.fmt.bufPrint(&buffer, "{s}_scale", .{k});
                    offset_partial = readPair(object_map.get(name).?, "data_offsets");
                    name = try std.fmt.bufPrint(&buffer, "{s}_scale_2", .{k});
                    offset_global = readPair(object_map.get(name).?, "data_offsets");
                }

                var elem_size: u64 = 0;
                switch (dtype) {
                    .F32 => elem_size = 4,
                    .BF16 => elem_size = 2,
                    .F8_E4M3 => elem_size = 1,
                    .U8 => {
                        std.debug.print("a U8 value went through condition", .{});
                    },
                }

                var shape_size: u64 = 1;
                for (0..rank) |i| {
                    shape_size = shape_size * shape[i];
                }

                const out_end = cursor + shape_size * elem_size;
                const obj = OutputTensor{
                    .name = k,
                    .dtype = dtype,
                    .rank = rank,
                    .shape = shape,
                    .in_start = in_offsets[0],
                    .in_end = in_offsets[1],
                    .offset_global = offset_global,
                    .offset_partial = offset_partial,
                    .out_start = cursor,
                    .out_end = out_end,
                };

                cursor += obj.out_end - obj.out_start;
                try res.append(gpa, obj);
            }
        }
    }

    return res.toOwnedSlice(gpa);
}

pub fn writeHeader(gpa: std.mem.Allocator, writer: *std.Io.Writer, tensors_layout: []OutputTensor) !void {
    var acc = std.Io.Writer.Allocating.init(gpa);
    defer acc.deinit();
    var js: std.json.Stringify = .{ .writer = &acc.writer };

    try js.beginObject();
    for (tensors_layout) |tensor| {
        try js.objectField(tensor.name);
        try js.beginObject();
        try js.objectField("dtype");
        try js.write(tensor.dtype);
        try js.objectField("shape");
        try js.write(tensor.shape[0..tensor.rank]);
        try js.objectField("data_offsets");
        try js.write(.{ tensor.out_start, tensor.out_end });
        try js.endObject();
    }
    try js.endObject();

    try writer.writeInt(u64, acc.written().len, .little);
    try writer.writeAll(acc.written());

    //try writer.interface.writeInt(u64, , endian: Endian)
    //const t = try writer.interface.writeAll(buf);
    //defer gpa.free(t);
}
