const std = @import("std");
pub fn main() !void {
    var prng = std.rand.DefaultPrng.init(1);
    const rnd = prng.random();
    const B: usize = 1000;
    var envs: [B][18]u8 = .{.{0} ** 18} ** B;

    const start = std.time.nanoTimestamp();
    var steps: usize = 0;
    while (@as(f64, @floatFromInt(std.time.nanoTimestamp() - start)) / 1e9 < 1.0) : (steps += B) {
        for (&envs) |*e| {
            const a = rnd.intRangeLessThan(usize, 0, 9);
            e[0] = 1;
            _ = a;
        }
    }
    std.debug.print("Zig: {d} steps/s\n", .{steps});
}
