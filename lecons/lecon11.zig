const std = @import("std");

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

fn decodeE4M3(byte: u8) f32 {
    const sign = byte >> 7 != 0;
    const decimal = byte & 7;
    const f_decimal: f32 = @floatFromInt(decimal);
    const ee: i8 = @intCast((byte >> 3) & 15);

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

fn unpack(byte: u8) [2]u4 {
    const low: u4 = @intCast(byte & 15);
    const high: u4 = @intCast(byte >> 4);
    return .{ low, high };
}

const E2M1_VALUES: [16]f32 = blk: {
    var t: [16]f32 = undefined;
    for (0..16) |i| t[i] = decodeE2M1(i);
    break :blk t;
};

fn decodeBloc(globale: f32, echelle: u8, paquet: [8]u8, sortie: *[16]f32) void {
    const echelle_decode = decodeE4M3(echelle);
    const echelle_totale = globale * echelle_decode;

    for (paquet, 0..) |p, i| {
        const d_p = unpack(p);

        sortie[2 * i] = echelle_totale * decodeE2M1(d_p[0]);
        sortie[2 * i + 1] = echelle_totale * decodeE2M1(d_p[1]);
    }
}

// La reference : la version en service, celle qu'il faut egaler bit pour bit.
fn decodeBlockTable(global_scale: f32, scale: u8, bytes: [8]u8, out: *[16]f32) void {
    const block_scale = decodeE4M3(scale);
    const total_scale = global_scale * block_scale;

    for (bytes, 0..) |p, i| {
        const d_p = unpack(p);

        out[2 * i] = total_scale * E2M1_VALUES[d_p[0]];
        out[2 * i + 1] = total_scale * E2M1_VALUES[d_p[1]];
    }
}

// Ce que l'assembleur de la version ci-dessus contient vraiment :
//   14 vpinsrb, 16 shr, 11 and, 27 mov   -> le depaquetage des quartets, en scalaire
//    2 vgatherdps                        -> la table, deja vectorisee par LLVM
//    2 vmulps, 5 vmulss                  -> la multiplication
// Le cout est le depaquetage, pas la table. D'ou deux variantes a ecrire et a comparer.

// VARIANTE 1 — A ECRIRE. On garde la table, on vectorise le depaquetage :
// les 8 octets contigus -> un @Vector(16, u4 ou u8) par & 15, >> 4 et entrelacement,
// puis les 16 lectures de table et la multiplication.
fn decodeBlockGather(global_scale: f32, scale: u8, bytes: [8]u8, out: *[16]f32) void {
    const block_scale = decodeE4M3(scale);
    const total_scale = global_scale * block_scale;

    const broadcast = @shuffle(u8, bytes, undefined, @Vector(16, u8){ 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7 });
    const mask: @Vector(16, bool) = .{ true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false };

    const low = broadcast & @as(@Vector(16, u8), @splat(15));
    const high = broadcast >> @as(@Vector(16, u8), @splat(4));

    const tmp: @Vector(16, u8) = @select(u8, mask, low, high);
    var res = @as(@Vector(16, f32), @splat(0));

    inline for (0..16) |i| {
        res[i] = E2M1_VALUES[tmp[i]];
    }

    res = res * @as(@Vector(16, f32), @splat(total_scale));

    out.* = res;
}

// VARIANTE 2 — A ECRIRE. Pas de table du tout : le motif f32 est reconstruit
// arithmetiquement depuis les bits du code, @select traite e == 0, @bitCast rend
// des f32. Meme depaquetage vectoriel que la variante 1.
fn decodeBlockSimd(global_scale: f32, scale: u8, bytes: [8]u8, out: *[16]f32) void {
    const block_scale = decodeE4M3(scale);
    const total_scale = global_scale * block_scale;

    const broadcast = @shuffle(u8, bytes, undefined, @Vector(16, u8){ 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7 });
    const mask: @Vector(16, bool) = .{ true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false };

    const low = broadcast & @as(@Vector(16, u8), @splat(15));
    const high = broadcast >> @as(@Vector(16, u8), @splat(4));

    const tmp: @Vector(16, u32) = @select(u8, mask, low, high);

    var sign: @Vector(16, u32) = (tmp >> @as(@Vector(16, u8), @splat(3)));
    const decimal: @Vector(16, u32) = tmp & @as(@Vector(16, u8), @splat(1));

    const ee: @Vector(16, u32) = (tmp >> @as(@Vector(16, u8), @splat(1))) & @as(@Vector(16, u8), @splat(3));

    sign = sign << @as(@Vector(16, u8), @splat(31));
    const n_decimal = decimal << @as(@Vector(16, u8), @splat(22));

    const ee_zero_decimal = ee + (@as(@Vector(16, u32), @splat(63)) << @as(@Vector(16, u32), @splat(24)));
    const ee_zero = @select(u32, decimal == @as(@Vector(16, u32), @splat(0)), ee, ee_zero_decimal);

    const ee_nonzero = (ee + @as(@Vector(16, u8), @splat(126))) << @as(@Vector(16, u8), @splat(23));
    const ee_nonzero_decimal = ee_nonzero | n_decimal;

    const u_decoded: @Vector(16, u32) = @select(u32, ee == @as(@Vector(16, u32), @splat(0)), ee_zero, ee_nonzero_decimal);

    const decoded: @Vector(16, u32) = sign | u_decoded;
    var res: @Vector(16, f32) = @as(@Vector(16, f32), @bitCast(decoded));
    res = res * @as(@Vector(16, f32), @splat(total_scale));

    out.* = res;
}

const GLOBALES = [_]f32{ 1.0, 0.5, 0.0078125, 3.7e-3, 448.0 };

// Toutes les echelles de bloc, tous les codes, dans les deux positions de quartet :
// un bloc rempli du meme octet couvre les 16 codes en 16 tours.
fn verifie(comptime candidat: fn (f32, u8, [8]u8, *[16]f32) void, nom: []const u8) void {
    var attendu: [16]f32 = undefined;
    var obtenu: [16]f32 = undefined;
    var cas: u64 = 0;

    for (GLOBALES) |g| {
        for (0..256) |s| {
            for (0..256) |b| {
                const bytes: [8]u8 = @splat(@intCast(b));
                const scale: u8 = @intCast(s);

                decodeBlockTable(g, scale, bytes, &attendu);
                candidat(g, scale, bytes, &obtenu);

                for (0..16) |i| {
                    const a: u32 = @bitCast(attendu[i]);
                    const o: u32 = @bitCast(obtenu[i]);
                    if (a != o) {
                        std.debug.print(
                            "{s} : global {d} echelle 0x{X:0>2} octet 0x{X:0>2} indice {d}\n" ++
                                "  attendu {d} (0x{X:0>8})\n  obtenu  {d} (0x{X:0>8})\n",
                            .{ nom, g, scale, b, i, attendu[i], a, obtenu[i], o },
                        );
                        return;
                    }
                }
                cas += 1;
            }
        }
    }

    std.debug.print("{s} : {d} blocs, 0 divergence\n", .{ nom, cas });
}

pub fn main() void {
    verifie(decodeBlockGather, "gather");
    verifie(decodeBlockSimd, "arithmetique");
}
