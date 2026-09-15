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

const Sortie = struct {
    nom: []const u8,
    dtype: Dtype,
    rank: u8,
    shape: [4]u64,
    debut: u64,
    fin: u64,
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

pub fn plan(gpa: std.mem.Allocator, object_map: std.json.ObjectMap) ![]Sortie {
    var res: std.ArrayList(Sortie) = .empty;

    var cursor: u64 = 0;
    for (object_map.keys(), object_map.values()) |k, v| {
        if (!std.mem.endsWith(u8, k, "weight_scale") and !std.mem.endsWith(u8, k, "weight_scale_2")) {
            if (v.object.get("dtype")) |d| {
                var obj = Sortie{
                    .nom = k,
                    .dtype = std.meta.stringToEnum(Dtype, d.string).?,
                    .rank = @intCast(v.object.get("shape").?.array.items.len),
                    .shape = blk: {
                        var t: [4]u64 = .{ 1, 1, 1, 1 };
                        for (0..v.object.get("shape").?.array.items.len) |i| t[i] = @intCast(v.object.get("shape").?.array.items[i].integer);
                        break :blk t;
                    },
                    .debut = cursor,
                    .fin = undefined,
                };
                if (obj.dtype == .U8) {
                    obj.dtype = .F32;
                    obj.shape[obj.rank - 1] = obj.shape[obj.rank - 1] * 2;
                }

                var factor: u64 = 0;
                switch (obj.dtype) {
                    .F32 => factor = 4,
                    .BF16 => factor = 2,
                    .F8_E4M3 => factor = 1,
                    .U8 => {
                        std.debug.print("a U8 value went through condition", .{});
                    },
                }

                var shape_size: u64 = 1;
                for (0..obj.rank) |i| {
                    shape_size = shape_size * obj.shape[i];
                }

                obj.fin = cursor + shape_size * factor;
                cursor += obj.fin - obj.debut;

                try res.append(gpa, obj);
            }
        }
    }

    return res.toOwnedSlice(gpa);
}

//fn write(){
//        try js.objectField(obj.nom);
//    try js.beginObject();
//    try js.objectField("dtype");
//    try js.write(obj.dtype);
//    try js.objectField("shape");
//    try js.write(obj.shape);
//    try js.objectField("data_offsets");
//    try js.write(.{ obj.debut, obj.fin });
//    try js.endObject();
//
//}
