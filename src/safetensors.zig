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
    start: u64,
    end: u64,
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

pub fn layout(gpa: std.mem.Allocator, object_map: std.json.ObjectMap) ![]OutputTensor {
    var res: std.ArrayList(OutputTensor) = .empty;

    var cursor: u64 = 0;
    for (object_map.keys(), object_map.values()) |k, v| {
        if (!std.mem.endsWith(u8, k, "weight_scale") and !std.mem.endsWith(u8, k, "weight_scale_2")) {
            if (v.object.get("dtype")) |d| {
                var obj = OutputTensor{
                    .name = k,
                    .dtype = std.meta.stringToEnum(Dtype, d.string).?,
                    .rank = @intCast(v.object.get("shape").?.array.items.len),
                    .shape = blk: {
                        var t: [4]u64 = .{ 1, 1, 1, 1 };
                        for (0..v.object.get("shape").?.array.items.len) |i| t[i] = @intCast(v.object.get("shape").?.array.items[i].integer);
                        break :blk t;
                    },
                    .start = cursor,
                    .end = undefined,
                };
                if (obj.dtype == .U8) {
                    obj.dtype = .F32;
                    obj.shape[obj.rank - 1] = obj.shape[obj.rank - 1] * 2;
                }

                var elem_size: u64 = 0;
                switch (obj.dtype) {
                    .F32 => elem_size = 4,
                    .BF16 => elem_size = 2,
                    .F8_E4M3 => elem_size = 1,
                    .U8 => {
                        std.debug.print("a U8 value went through condition", .{});
                    },
                }

                var shape_size: u64 = 1;
                for (0..obj.rank) |i| {
                    shape_size = shape_size * obj.shape[i];
                }

                obj.end = cursor + shape_size * elem_size;
                cursor += obj.end - obj.start;

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
        try js.write(.{ tensor.start, tensor.end });
        try js.endObject();
    }
    try js.endObject();

    try writer.writeInt(u64, acc.written().len, .little);
    try writer.writeAll(acc.written());

    //try writer.interface.writeInt(u64, , endian: Endian)
    //const t = try writer.interface.writeAll(buf);
    //defer gpa.free(t);
}
