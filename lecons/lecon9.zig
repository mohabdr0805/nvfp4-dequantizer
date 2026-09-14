const std = @import("std");

fn open_tensor(io: std.Io, gpa: std.mem.Allocator, filename: []const u8) ![]const u8 {
    const file: std.Io.File = try std.Io.Dir.openFile(.cwd(), io, filename, .{ .mode = .read_only });
    defer file.close(io);

    var buffer: [4096]u8 = undefined;
    var reader = file.reader(io, &buffer);

    const n = (try file.stat(io)).size;

    const t = try reader.interface.readAlloc(gpa, n);

    return t;
}

fn decodeE2M1(code: u4) f32 {
    const sign = code >> 3 != 0;
    const decimal = code & 1;
    const f_decimal: f32 = @floatFromInt(decimal);

    const ee = (code >> 1) & 3;

    var res: f32 = 0;

    if (ee == 0) {
        res = f_decimal * 0.5;
    } else {
        const shift: f32 = @floatFromInt(@as(u4, 1) << @intCast(ee - 1));
        res = (1 + f_decimal / 2) * shift;
    }

    res = if (sign) -res else res;

    return res;
}

// E4M3 (variante FN de l'OCP) : 1 bit de signe, 4 bits d'exposant (biais 7),
// 3 bits de mantisse. Pas d'infini. NaN = exposant 15 et mantisse 7.
fn decodeE4M3(octet: u8) f32 {
    const sign = octet >> 7 != 0; //last bit
    const decimal = octet & 7; //first 3 bits
    const f_decimal: f32 = @floatFromInt(decimal);
    const ee: i8 = @intCast((octet >> 3) & 15);

    var res: f32 = 0;

    if (ee == 0) {
        const shift: f32 = std.math.ldexp(@as(f32, 1.0), -6);
        res = (f_decimal / 8) * shift;
    } else if (ee == 15 and decimal == 7) {
        return std.math.nan(f32);
    } else {
        const shift: f32 = std.math.ldexp(@as(f32, 1.0), ee - 7);
        res = (1 + f_decimal / 8) * shift;
    }

    res = if (sign) -res else res;

    return res;
}

// CONVENTION DE NIBBLES : element[0] = nibble BAS, element[1] = nibble HAUT.
// Choix, pas deduction : les octets ne disent pas lequel porte l'indice pair.
// A valider contre une reference a l'etape 3d.
fn depaquete(octet: u8) [2]u4 {
    const faible: u4 = @intCast(octet & 15);
    const fort: u4 = @intCast(octet >> 4);
    return .{ faible, fort };
}

// Un bloc NVFP4 : 16 elements E2M1 dans 8 octets, 1 octet d'echelle E4M3,
// et l'echelle FP32 commune a tout le tenseur.
//     valeur = globale * decodeE4M3(echelle) * decodeE2M1(element)
fn decodeBloc(globale: f32, echelle: u8, paquet: [8]u8, sortie: *[16]f32) void {
    const echelle_decode = decodeE4M3(echelle);
    const echelle_totale = globale * echelle_decode;

    for (paquet, 0..) |p, i| {
        const d_p = depaquete(p);

        sortie[2 * i] = echelle_totale * decodeE2M1(d_p[0]);
        sortie[2 * i + 1] = echelle_totale * decodeE2M1(d_p[1]);
    }
}

const TABLE: [16]f32 = blk: {
    var t: [16]f32 = undefined;
    for (0..16) |i| t[i] = decodeE2M1(i);
    break :blk t;
};

//decode NVFP4 mais en prenans le tableau en comptime
fn decodeBloc_inline(globale: f32, echelle: u8, paquet: [8]u8, sortie: *[16]f32) void {
    const echelle_decode = decodeE4M3(echelle);
    const echelle_totale = globale * echelle_decode;

    for (paquet, 0..) |p, i| {
        const d_p = depaquete(p);

        sortie[2 * i] = echelle_totale * TABLE[d_p[0]];
        sortie[2 * i + 1] = echelle_totale * TABLE[d_p[1]];
    }
}

// A lancer depuis zig-lab, en RELEASE :
//     zig run -OReleaseFast lecons/lecon9.zig
//
// Les donnees viennent de fetch_bench.py :
//     bench/q_proj_packed.bin   8 388 608 o  -> 16 777 216 elements
//     bench/q_proj_scales.bin   1 048 576 o  -> 1 048 576 blocs
//     bench/q_proj_global.bin           4 o
pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;

    // 1. lire les trois fichiers de bench/
    // 2. allouer la sortie : 16 777 216 f32 = 67 Mo
    // 3. chronometrer le decodage de tous les blocs
    // 4. refaire la mesure plusieurs fois, garder la meilleure
    // 5. afficher : temps, valeurs/s, et debit en Go/s
    const n = 16777216;
    const bloc = 1048576;

    var out = try gpa.alloc(f32, n);
    defer gpa.free(out);

    const f_global = try open_tensor(io, gpa, "bench/q_proj_global.bin");
    defer gpa.free(f_global);
    const f_partial = try open_tensor(io, gpa, "bench/q_proj_scales.bin");
    defer gpa.free(f_partial);
    const f_fp4 = try open_tensor(io, gpa, "bench/q_proj_packed.bin");
    defer gpa.free(f_fp4);

    std.debug.print("lens : {d}, {d}, {d}\n", .{ f_global.len, f_partial.len, f_fp4.len });

    const global: f32 = @bitCast(f_global[0..4].*);
    std.debug.print("global = {d}\n", .{global});

    const n_iter = 40;
    var times: [n_iter]i64 = undefined;

    for (0..n_iter) |iter| {
        const start = std.Io.Clock.awake.now(init.io);
        for (0..bloc) |i| {
            const partial: u8 = f_partial[i];
            const fp4: [8]u8 = f_fp4[8 * i ..][0..8].*;
            decodeBloc_inline(global, partial, fp4, out[16 * i ..][0..16]);
        }
        const end = std.Io.Clock.awake.now(init.io);
        const duration = std.Io.Timestamp.durationTo(start, end);
        times[iter] = duration.toMicroseconds();
    }

    const best = std.mem.min(i64, &times);
    std.debug.print("Time Elapsed: {any} mirosecs, best = {d}\n", .{ times, best });

    const o_total = (@as(f64, @floatFromInt(n)) / 2 + bloc + 4 + n * 4) / 1000;
    const debit: f64 = o_total / @as(f64, @floatFromInt(best));
    std.debug.print("Debit max = {d} Go/s\n", .{debit});

    std.mem.doNotOptimizeAway(out);
}
