const qmath = @This();

const std = @import("std");
const print = std.debug.print;
const assert = std.debug.assert;
const math = std.math;

/// Math Standard
///
/// Counter Clockwise Rotations
///
/// Two Dimensions:
/// +Y = Up
/// +X = Right
///
/// Three Dimensions:
/// +Y = Up
/// +X = Right
/// -Z = Forward

pub fn lerp(start: f32, end: f32, time: f32) f32 {
    return start + (time * (end - start));
}

pub fn smoothStep(value: f32) f32 {
    if(value < 0){
        return 0;
    }else if(value >= 1){
        return 1;
    }else{
        return (3 * (value * value)) - (2 * (value * value * value));
    }
}

pub fn distance(a: f32, b: f32) f32 {
    if(a >= b){
        return a - b;
    }else{
        return b - a;
    }
}

pub fn degreesToRadians(degrees: f32) f32 {
    return math.degreesToRadians(degrees);
}

pub fn radiansToDegrees(radians: f32) f32 {
    return math.radiansToDegrees(radians);
}

pub const Vec2f = struct {

    x: f32 = 0,
    y: f32 = 0,

    pub fn new(x: f32, y: f32) Vec2f {
        return .{
            .x = x,
            .y = y,
        };
    }

    pub fn add(a: Vec2f, b: Vec2f) Vec2f {
        return .{
            .x = a.x + b.x,
            .y = a.y + b.y,
        };
    }

    pub fn sub(a: Vec2f, b: Vec2f) Vec2f {
        return .{
            .x = a.x - b.x,
            .y = a.y - b.y,
        };
    }

    pub fn mult(a: Vec2f, b: Vec2f) Vec2f {
        return .{
            .x = a.x * b.x,
            .y = a.y * b.y,
        };
    }

    pub fn div(a: Vec2f, b: Vec2f) Vec2f {
        return .{
            .x = a.x / b.x,
            .y = a.y / b.y,
        };
    }

    pub fn equals(a: Vec2f, b: Vec2f) bool {
        return
            a.x == b.x and
            a.y == b.y;
    }

    pub fn scalar(vec: Vec2f, scale: f32) Vec2f {
        return .{
            .x = vec.x * scale,
            .y = vec.y * scale,
        };
    }

    pub fn lerp(start: Vec2f, end: Vec2f, time: f32) Vec2f {
        return .{
            .x = qmath.lerp(start.x, end.x, time),
            .y = qmath.lerp(start.y, end.y, time),
        };
    }

    pub fn fromAngle(angle: f32) Vec2f {
        return .{
            .x = @sin(angle),
            .y = @cos(angle),
        };
    }

    pub fn length(vec: Vec2f) f32 {
        const x = vec.x * vec.x;
        const y = vec.y * vec.y;
        return @sqrt(x + y);
    }

    pub fn distance(a: Vec2f, b: Vec2f) f32 {
        const x = b.x - a.x;
        const y = b.y - a.y;
        return @sqrt((x * x) + (y * y));
    }

    pub fn normalize(vec: Vec2f) Vec2f {
        const len = vec.length();
        if(len == 0){
            return vec;
        }
        const inv = 1 / len;
        return .{
            .x = vec.x * inv,
            .y = vec.y * inv,
        };
    }

    pub fn toVec2i(vec: Vec2f) Vec2i {
        var result = Vec2i{
            .x = @intFromFloat(vec.x),
            .y = @intFromFloat(vec.y),
        };
        if(vec.x < 0){
            result.x -= 1;
        }
        if(vec.y < 0){
            result.y -= 1;
        }
        return result;
    }

    pub fn toVec3f(vec: Vec2f, z: f32) Vec3f {
        return Vec3f.new(vec.x, vec.y, z);
    }

    pub fn toArray(vec: Vec2f) [2]f32 {
        return [2]f32{vec.x, vec.y};
    }

    pub const zero = Vec2f.new(0, 0);
    pub const one = Vec2f.new(1, 1);
    pub const half = Vec2f.new(0.5, 0.5);
    pub const up = Vec2f.new(0, 1);
    pub const right = Vec2f.new(1, 0);
};

pub const Vec3f = struct {

    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,

    pub fn new(x: f32, y: f32, z: f32) Vec3f {
        return .{
            .x = x,
            .y = y,
            .z = z,
        };
    }

    pub fn newScalar(value: f32) Vec3f {
        return .{
            .x = value,
            .y = value,
            .z = value,
        };
    }

    pub fn add(a: Vec3f, b: Vec3f) Vec3f {
        return .{
            .x = a.x + b.x,
            .y = a.y + b.y,
            .z = a.z + b.z,
        };
    }

    pub fn sub(a: Vec3f, b: Vec3f) Vec3f {
        return .{
            .x = a.x - b.x,
            .y = a.y - b.y,
            .z = a.z - b.z,
        };
    }

    pub fn mult(a: Vec3f, b: Vec3f) Vec3f {
        return .{
            .x = a.x * b.x,
            .y = a.y * b.y,
            .z = a.z * b.z,
        };
    }

    pub fn div(a: Vec3f, b: Vec3f) Vec3f {
        return .{
            .x = a.x / b.x,
            .y = a.y / b.y,
            .z = a.z / b.z,
        };
    }

    pub fn equals(a: Vec3f, b: Vec3f) bool {
        return
            a.x == b.x and
            a.y == b.y and
            a.z == b.z;
    }

    pub fn scalar(vec: Vec3f, scale: f32) Vec3f {
        return .{
            .x = vec.x * scale,
            .y = vec.y * scale,
            .z = vec.z * scale,
        };
    }

    pub fn lerp(start: Vec3f, end: Vec3f, time: f32) Vec3f {
        return .{
            .x = qmath.lerp(start.x, end.x, time),
            .y = qmath.lerp(start.y, end.y, time),
            .z = qmath.lerp(start.z, end.z, time),
        };
    }

    pub fn length(vec: Vec3f) f32 {
        const x = vec.x * vec.x;
        const y = vec.y * vec.y;
        const z = vec.z * vec.z;
        return @sqrt(x + y + z);
    }

    pub fn distance(a: Vec3f, b: Vec3f) f32 {
        const x = b.x - a.x;
        const y = b.y - a.y;
        const z = b.z - a.z;
        return @sqrt((x * x) + (y * y) + (z * z));
    }

    pub fn normalize(vec: Vec3f) Vec3f {
        const len = vec.length();
        if(len == 0){
            return vec;
        }
        const inv = 1 / len;
        return .{
            .x = vec.x * inv,
            .y = vec.y * inv,
            .z = vec.z * inv,
        };
    }

    pub fn negate(vec: Vec3f) Vec3f {
        return Vec3f{
            .x = -vec.x,
            .y = -vec.y,
            .z = -vec.z
        };
    }

    pub fn transform(vec: Vec3f, mat: *const Mat4) Vec3f {
        const x =
            (vec.x * mat.m[0][0]) + (vec.y * mat.m[0][1]) +
            (vec.z * mat.m[0][2]) + mat.m[0][3];
        const y =
            (vec.x * mat.m[1][0]) + (vec.y * mat.m[1][1]) +
            (vec.z * mat.m[1][2]) + mat.m[1][3];
        const z =
            (vec.x * mat.m[2][0]) + (vec.y * mat.m[2][1]) +
            (vec.z * mat.m[2][2]) + mat.m[2][3];
        const w =
            (vec.x * mat.m[3][0]) + (vec.y * mat.m[3][1]) +
            (vec.z * mat.m[3][2]) + mat.m[3][3];
        if(w == 0){
            return Vec3f.zero;
        }else{
            return Vec3f{
                .x = x,
                .y = y,
                .z = z,
            };
        }
    }

    pub fn dot(a: Vec3f, b: Vec3f) f32 {
        return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
    }

    pub fn cross(a: Vec3f, b: Vec3f) Vec3f {
        return Vec3f{
            .x = a.y * b.z - a.z * b.y,
            .y = a.z * b.x - a.x * b.z,
            .z = a.x * b.y - a.y * b.x,
        };
    }

    pub fn getDataPtr(vec: Vec3f) [*c]f32 {
        return @ptrCast(&vec.x);
    }

    pub fn toVec2f(vec: Vec3f) Vec2f {
        return Vec2f.new(vec.x, vec.y);
    }

    pub fn toArray(vec: Vec3f) [2]f32 {
        return [3]f32{vec.x, vec.y, vec.z};
    }

    pub const zero = Vec3f.new(0, 0, 0);
    pub const up = Vec3f.new(0, 1, 0);
    pub const right = Vec3f.new(1, 0, 0);
    pub const forward = Vec3f.new(0, 0, -1);
};

pub const Vec4f = struct {
    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,
    w: f32 = 0,
};

pub const Quaternion = struct {

    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,
    w: f32 = 0,

    pub fn create(axis: Vec3f, angle: f32) Quaternion {
        return Quaternion{
            .x = axis.x * @sin(angle / 2),
            .y = axis.y * @sin(angle / 2),
            .z = axis.z * @sin(angle / 2),
            .w = @cos(angle / 2)
        };
    }

    pub fn mult(a: *const Quaternion, b: *const Quaternion) Quaternion {
        return Quaternion{
            .x = (a.w * b.x) + (a.x * b.w) + (a.y * b.z) - (a.z * b.y),
            .y = (a.w * b.y) + (a.y * b.w) + (a.z * b.x) - (a.x * b.z),
            .z = (a.w * b.z) + (a.z * b.w) + (a.x * b.y) - (a.y * b.x),
            .w = (a.w * b.w) - (a.x * b.x) - (a.y * b.y) - (a.z * b.z),
        };
    }

    pub fn transform(q: *const Quaternion, vec: Vec3f) Quaternion {
        return Quaternion{
            .x = - (q.x * vec.x) - (q.y * vec.y) - (q.z * vec.z),
            .y = (q.w * vec.x) + (q.y * vec.z) - (q.z * vec.y),
            .z = (q.w * vec.y) + (q.z * vec.y) - (q.x * vec.z),
            .w = (q.w * vec.z) + (q.x * vec.x) - (q.y * vec.x),
        };
    }

    pub fn conjugate(q: *const Quaternion) Quaternion {
        return Quaternion{
            .x = -q.x,
            .y = -q.y,
            .z = -q.z,
            .w = q.w,
        };
    }

    pub fn length(q: *const Quaternion) f32 {
        const x = q.x * q.x;
        const y = q.y * q.y;
        const z = q.z * q.z;
        const w = q.w * q.w;
        return @sqrt(x + y + z + w);
    }

    pub fn toMatrix(q: *const Quaternion) Mat4 {
        var result: Mat4 = undefined;

        result.m[0][0] = 1 - (2 * (q.y * q.y)) - (2 * (q.z * q.z));
        result.m[0][1] = (2 * (q.x * q.y)) - (2 * (q.w * q.z));
        result.m[0][2] = (2 * (q.x * q.z)) + (2 * (q.w * q.y));
        result.m[0][3] = 0;

        result.m[1][0] = (2 * (q.x * q.y)) + (2 * (q.w * q.z));
        result.m[1][1] = 1 - (2 * (q.x * q.x)) - (2 * (q.z * q.z));
        result.m[1][2] = (2 * (q.y * q.z)) - (2 * (q.w * q.x));
        result.m[1][3] = 0;

        result.m[2][0] = (2 * (q.x * q.z)) - (2 * (q.w * q.y));
        result.m[2][1] = (2 * (q.y * q.z)) + (2 * (q.w * q.x));
        result.m[2][2] = 1 - (2 * (q.x * q.x)) - (2 * (q.y * q.y));
        result.m[2][3] = 0;

        result.m[3][0] = 0;
        result.m[3][1] = 0;
        result.m[3][2] = 0;
        result.m[3][3] = 1;
        return result;
    }
};

pub const Vec2i = struct {

    x: i32 = 0,
    y: i32 = 0,

    pub fn new(x: i32, y: i32) Vec2i {
        return .{
            .x = x,
            .y = y,
        };
    }

    pub fn fromIndex(index: i32, bounds: Vec2i) Vec2i {
        return fromIndexXY(index, bounds.x, bounds.y);
    }

    pub fn fromIndexXY(index: i32, width: i32, height: i32) Vec2i {
        _ = height;
        return .{
            .x = index / width,
            .y = index % width,
        };
    }

    pub fn add(a: Vec2i, b: Vec2i) Vec2i {
        return .{
            .x = a.x + b.x,
            .y = a.y + b.y,
        };
    }

    pub fn sub(a: Vec2i, b: Vec2i) Vec2i {
        return .{
            .x = a.x - b.x,
            .y = a.y - b.y,
        };
    }

    pub fn mult(a: Vec2i, b: Vec2i) Vec2i {
        return .{
            .x = a.x * b.x,
            .y = a.y * b.y,
        };
    }

    pub fn div(a: Vec2i, b: Vec2i) Vec2i {
        return .{
            .x = a.x / b.x,
            .y = a.y / b.y,
        };
    }

    pub fn scalar(vec: Vec2i, scale: i32) Vec2i {
        return .{
            .x = vec.x * scale,
            .y = vec.y * scale,
        };
    }

    pub fn distance(a: Vec2i, b: Vec2i) i32 {
        const dx: i32 = @intCast(@abs(a.x - b.x));
        const dy: i32 = @intCast(@abs(a.y - b.y));
        if(dx > dy){
            return dy + (dx - dy);
        }else{
            return dx + (dy - dx);
        }
    }

    pub fn equals(a: Vec2i, b: Vec2i) bool {
        return a.x == b.x and a.y == b.y;
    }

    pub fn getArea(vec: Vec2i) usize {
        assert(vec.x > 0);
        assert(vec.y > 0);
        return @as(usize, @intCast(vec.x)) * @as(usize, @intCast(vec.y));
    }

    pub fn toVec2f(vec: Vec2i) Vec2f {
        return .{
            .x = @floatFromInt(vec.x),
            .y = @floatFromInt(vec.y),
        };
    }

    pub const zero = Vec2i.new(0, 0);
    pub const up = Vec2i.new(0, 1);
    pub const right = Vec2i.new(1, 0);
};

pub const Recti = struct {

    pos: Vec2i          = Vec2i{},
    len: Vec2i          = Vec2i{},

    pub fn new(pos: Vec2i, len: Vec2i) Recti {
        assert(len.x > 0);
        assert(len.y > 0);

        return .{
            .pos = pos,
            .len = len,
        };
    }

    pub fn create(x: i32, y: i32, width: i32, height: i32) Recti {
        assert(width > 0);
        assert(height > 0);

        return .{
            .pos = Vec2i.new(x, y),
            .len = Vec2i.new(width, height),
        };
    }

    pub fn createFromCentre(centre: Vec2i, radius: Vec2i) Recti {
        return .{
            .pos = centre.sub(radius),
            .len = radius.scalar(2).add(Vec2i.new(1, 1)),
        };
    }

    pub fn getFarX(rect: Recti) i32 {
        return rect.pos.x + rect.len.x - 1;
    }

    pub fn getFarY(rect: Recti) i32 {
        return rect.pos.y + rect.len.y - 1;
    }

    pub fn getFarPoint(rect: Recti) Vec2i {
        return .{
            .x = rect.getFarX(),
            .y = rect.getFarY(),
        };
    }

    pub fn isInBoundsXY(rect: Recti, x: i32, y: i32) bool {
        return isInBoundsVec(rect, Vec2i.new(x, y));
    }

    pub fn isInBoundsVec(rect: Recti, vec: Vec2i) bool {
        return 
            vec.x >= rect.pos.x and
            vec.y >= rect.pos.y and
            vec.x < (rect.pos.x + rect.len.x) and
            vec.y < (rect.pos.y + rect.len.y);
    }

    pub fn isInBoundsRect(rect: Recti, inside: Recti) bool {
        return inside.isInside(rect);
    }

    pub fn isInside(rect: Recti, bounds: Recti) bool {
        return
            rect.pos.x >= bounds.pos.x and
            rect.pos.y >= bounds.pos.y and
            rect.getFarX() <= bounds.getFarX() and
            rect.getFarY() <= bounds.getFarY();
    }

    pub fn isOnEdgeXY(rect: Recti, x: i32, y: i32) bool {
        return
            x == rect.pos.x or
            x == rect.getFarX() or
            y == rect.pos.y or
            y == rect.getFarY();
    }

    pub fn isOnEdgeVec(rect: Recti, vec: Vec2i) bool {
        return rect.isOnEdgeXY(vec.x, vec.y);
    }

    pub fn doesIntersectRect(a: Recti, b: Recti) bool {
        return
            a.pos.x < (b.pos.x + b.len.x) and
            b.pos.x < (a.pos.x + a.len.x) and
            a.pos.y < (b.pos.y + b.len.y) and
            b.pos.y < (a.pos.y + a.len.y);
    }

    pub fn getCentrePoint(rect: Recti) Vec2i {
        return .{
            .x = rect.pos.x + @divTrunc(rect.len.x, 2),
            .y = rect.pos.y + @divTrunc(rect.len.y, 2),
        };
    }

    pub fn getIndexXY(rect: Recti, x: i32, y: i32) usize {
        assert(rect.isInBoundsXY(x, y));

        const iw = @as(usize, @intCast(rect.len.x));
        const ix = @as(usize, @intCast(x - rect.pos.x));
        const iy = @as(usize, @intCast(y - rect.pos.y));

        return (iy * iw) + ix;
    }

    pub fn getIndexVec(rect: Recti, vec: Vec2i) usize {
        return rect.getIndexXY(vec.x, vec.y);
    }

    pub fn getArea(rect: Recti) usize {
        return 
            @as(usize, @intCast(rect.len.x)) *
            @as(usize, @intCast(rect.len.y));
    }

    pub fn equals(a: Recti, b: Recti)  bool {
        return a.pos.equals(b.pos) and a.len.equals(b.len);
    }

    pub fn toRectf(rect: Recti) Rectf {
        return .{
            .pos = rect.pos.toVec2f(),
            .len = rect.len.toVec2f()
        };
    }

    pub fn isLegal(rect: Recti) bool {
        return rect.len.x > 0 and rect.len.y > 0;
    }
};

pub const Rectf = struct {

    pos: Vec2f          = Vec2f{},
    len: Vec2f          = Vec2f{},

    pub fn new(x: f32, y: f32, width: f32, height: f32) Rectf {
        return Rectf{
            .pos = Vec2f{.x = x, .y = y},
            .len = Vec2f{.x = width, .y = height}
        };
    }

    pub fn scaleFromCentre(rect: *const Rectf, scale: f32) Rectf {
        const len = rect.len.scalar(scale);
        const diff = len.sub(rect.len);
        const pos = rect.pos.sub(diff.scalar(0.5));

        return .{
            .pos = pos,
            .len = len,
        };
    }

    pub fn scaleAll(rect: *const Rectf, scale: f32) Rectf {
        return .{
            .pos = rect.pos.scalar(scale),
            .len = rect.len.scalar(scale),
        };
    }
};

pub const UV = struct {

    uv00: Vec2f = Vec2f{},
    uv11: Vec2f = Vec2f{},

    pub fn new(uv00: Vec2f, uv11: Vec2f) UV {
        return .{
            .uv00 = uv00,
            .uv11 = uv11,
        };
    }

    pub fn toRect(uv: *const UV, _src: *const Rectf) Rectf {
        const pos = uv.uv00.mult(_src.len);
        return .{
            .pos = pos.add(_src.pos),
            .len = uv.uv11.mult(_src.len).sub(pos),
        };
    }

    pub fn uv01(uv: *const UV) Vec2f {
        return Vec2f.new(uv.uv00.x, uv.uv11.y);
    }

    pub fn uv10(uv: *const UV) Vec2f {
        return Vec2f.new(uv.uv11.x, uv.uv00.y);
    }

    pub const half = UV.new(Vec2f.new(0, 0), Vec2f.new(0.5, 0.5));
    pub const max = UV.new(Vec2f.new(0, 0), Vec2f.new(1, 1));
};

pub const Mat4 = struct {

    m: [4][4]f32 = .{
        .{0, 0, 0, 0},
        .{0, 0, 0, 0},
        .{0, 0, 0, 0},
        .{0, 0, 0, 0}
    },

    pub fn createOrthographic(
        left: f32,
        bottom: f32,
        right: f32,
        top: f32,
        near: f32,
        far: f32
    ) Mat4 {
        var result = Mat4{};

        result.m[0][0] = 2 / (right - left);
        result.m[1][1] = 2 / (top - bottom);
        result.m[2][2] = -2 / (far - near);

        result.m[3][0] = -((right + left) / (right - left));
        result.m[3][1] = -((top + bottom) / (top - bottom));
        result.m[3][2] = -((far + near) / (far - near));

        result.m[3][3] = 1;
        return result;
    }

    pub fn createPerspective(
        fov: f32,
        aspectratio: f32,
        near: f32,
        far: f32
    ) Mat4 {
        var result = Mat4{};
        const tangent = @tan(fov / 2);
        const right = near * tangent;
        const top = right / aspectratio;

        result.m[0][0] = near / right;
        result.m[1][1] = near / top;

        result.m[2][2] = -(far + near) / (far - near);
        result.m[3][2] = -(2 * far * near) / (far - near);
        result.m[2][3] = -1;

        result.m[3][3] = 1;
        return result;
    }

    pub fn createScaling(x: f32, y: f32, z: f32) Mat4 {
        return Mat4{
            .m = [4][4]f32{
                .{x, 0, 0, 0},
                .{0, y, 0, 0},
                .{0, 0, z, 0},
                .{0, 0, 0, 1},
            }
        };
    }

    pub fn createTranslation(x: f32, y: f32, z: f32) Mat4 {
        return Mat4{
            .m = [4][4]f32{
                .{1, 0, 0, 0},
                .{0, 1, 0, 0},
                .{0, 0, 1, 0},
                .{x, y, z, 1},
            }
        };
    }

    pub fn createTranslationV(vec: Vec3f) Mat4 {
        return createTranslation(vec.x, vec.y, vec.z);
    }

    pub fn createRotationX(angle: f32) Mat4 {
        return Mat4{
            // .m = [4][4]f32{
            //     .{1, 0, 0, 0},
            //     .{0, @cos(angle), -@sin(angle), 0},
            //     .{0, @sin(angle), @cos(angle), 0},
            //     .{0, 0, 0, 1},
            // }
            .m = [4][4]f32{
                .{1, 0, 0, 0},
                .{0, @cos(angle), @sin(angle), 0},
                .{0, -@sin(angle), @cos(angle), 0},
                .{0, 0, 0, 1},
            }
        };
    }

    pub fn createRotationY(angle: f32) Mat4 {
        return Mat4{
            // .m = [4][4]f32{
            //     .{@cos(angle), 0, @sin(angle), 0},
            //     .{0, 1, 0, 0},
            //     .{-@sin(angle), 0, @cos(angle), 0},
            //     .{0, 0, 0, 1}
            // }
            .m = [4][4]f32{
                .{@cos(angle), 0, -@sin(angle), 0},
                .{0, 1, 0, 0},
                .{@sin(angle), 0, @cos(angle), 0},
                .{0, 0, 0, 1}
            }
        };
    }

    pub fn createRotationZ(angle: f32) Mat4 {
        return Mat4{
            // .m = [4][4]f32{
            //     .{@cos(angle), -@sin(angle), 0, 0},
            //     .{@sin(angle), @cos(angle), 0, 0},
            //     .{0, 0, 1, 0},
            //     .{0, 0, 0, 1}
            // }
            .m = [4][4]f32{
                .{@cos(angle), @sin(angle), 0, 0},
                .{-@sin(angle), @cos(angle), 0, 0},
                .{0, 0, 1, 0},
                .{0, 0, 0, 1}
            }
        };
    }

    pub fn createRotationXYZ(angle: Vec3f) Mat4 {
        const m00 = @cos(angle.y) * @cos(angle.z);
        const m01 = -@cos(angle.y) * @sin(angle.z);
        const m02 = @sin(angle.y);

        const m10 =
            (@cos(angle.x) * @sin(angle.z)) +
            (@sin(angle.x) * @sin(angle.y) * @cos(angle.z));
        const m11 =
            (@cos(angle.x) * @cos(angle.z)) -
            (@sin(angle.x) * @sin(angle.y) * @sin(angle.z));
        const m12 = -@sin(angle.x) * @cos(angle.y);

        const m20 =
            (@sin(angle.x) * @sin(angle.z)) -
            (@cos(angle.x) * @sin(angle.y) * @cos(angle.z));
        const m21 =
            (@sin(angle.x) * @cos(angle.z)) +
            (@cos(angle.x) * @sin(angle.y) * @sin(angle.z));
        const m22 = @cos(angle.x) * @cos(angle.y);

        return Mat4{
            .m = [4][4]f32{
                .{m00, m01, m02, 0},
                .{m10, m11, m12, 0},
                .{m20, m21, m22, 0},
                .{0, 0, 0, 1}
            }
        };
    }

    pub fn createRotationZYX(angle: Vec3f) Mat4 {
        const m00 = @cos(angle.y) * @cos(angle.z);
        const m01 =
            (@cos(angle.z) * @sin(angle.x) * @sin(angle.y)) -
            (@cos(angle.x) * @sin(angle.z));
        const m02 =
            (@cos(angle.x) * @cos(angle.z) * @sin(angle.y)) +
            (@sin(angle.x) * @sin(angle.z));

        const m10 = @cos(angle.y) * @sin(angle.z);
        const m11 =
            (@cos(angle.x) * @cos(angle.z)) +
            (@sin(angle.x) * @sin(angle.y) * @sin(angle.z));
        const m12 =
            (-@cos(angle.z) * @sin(angle.x)) +
            (@cos(angle.x) * @sin(angle.y) * @sin(angle.z));

        const m20 = -@sin(angle.y);
        const m21 = @cos(angle.y) * @sin(angle.x);
        const m22 = @cos(angle.x) * @cos(angle.y);

        return Mat4{
            .m = [4][4]f32{
                .{m00, m01, m02, 0},
                .{m10, m11, m12, 0},
                .{m20, m21, m22, 0},
                .{0, 0, 0, 1}
            }
        };
    }

    pub fn createRotationAxisAngle(axis: Vec3f, angle: f32) Mat4 {
        // Rodrigues' Formula
        // https://www.songho.ca/opengl/glrotate.html

        const c = @cos(angle);
        const s = @sin(angle);

        var result: Mat4 = Mat4{};

        result.m[0][0] = ((1 - c) * (axis.x * axis.x)) + c;
        result.m[0][1] = ((1 - c) * (axis.x * axis.y)) - (s * axis.z);
        result.m[0][2] = ((1 - c) * (axis.x * axis.z)) + (s * axis.y);

        result.m[1][0] = ((1 - c) * (axis.x * axis.y)) + (s * axis.z);
        result.m[1][1] = ((1 - c) * (axis.y * axis.y)) + c;
        result.m[1][2] = ((1 - c) * (axis.y * axis.z)) - (s * axis.x);

        result.m[2][0] = ((1 - c) * (axis.x * axis.z)) - (s * axis.y);
        result.m[2][1] = ((1 - c) * (axis.y * axis.z)) + (s * axis.x);
        result.m[2][2] = ((1 - c) * (axis.z * axis.z) + c);

        result.m[3][3] = 1;
        return result;
    }

    pub fn createRotationEuler(angle: Vec3f) Mat4 {
        return Mat4.createRotationZYX(
            Vec3f{
                .x = angle.x,
                .y = -angle.y,
                .z = angle.z
            }
        );
        // return Mat4.createRotationX(angle.x).mult(
        //     &Mat4.createRotationY(-angle.y).mult(
        //         &Mat4.createRotationZ(angle.z)
        //     )
        // );
    }

    pub fn lookAt(pos: Vec3f, target: Vec3f, up: Vec3f) Mat4 {
        const look_forward = pos.sub(target).normalize();
        const look_left = up.cross(look_forward).normalize();
        const look_up = look_forward.cross(look_left).normalize();

        const x =
            (-look_left.x * pos.x) -
            (look_left.y * pos.y) -
            (look_left.z * pos.z);
        const y =
            (-look_up.x * pos.x) -
            (look_up.y * pos.y) -
            (look_up.z * pos.z);
        const z =
            (-look_forward.x * pos.x) -
            (look_forward.y * pos.y) -
            (look_forward.z * pos.z);
        return Mat4{
            .m = [4][4]f32{
                .{look_left.x, look_up.x, look_forward.x, 0},
                .{look_left.y, look_up.y, look_forward.y, 0},
                .{look_left.z, look_up.z, look_forward.z, 0},
                .{x, y, z, 1},
            }
        };
    }

    pub fn mult(a: *const Mat4, b: *const Mat4) Mat4 {
        var result: Mat4 = undefined;
        inline for (0..4) |row| {
            inline for (0..4) |col| {
                var value: f32 = 0.0;
                inline for (0..4) |i| {
                    value += a.m[row][i] * b.m[i][col];
                }
                result.m[row][col] = value;
            }
        }
        return result; 
    }

    pub fn transpose(mat: *const Mat4) Mat4 {
        return Mat4{
            .m = [4][4]f32{
                .{mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0]},
                .{mat.m[0][1], mat.m[1][1], mat.m[2][1], mat.m[3][1]},
                .{mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2]},
                .{mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3]},
            }
        };
    }

    pub fn invert(mat: *const Mat4) Mat4 {
        const a00 = mat.m[0][0];
        const a01 = mat.m[0][1];
        const a02 = mat.m[0][2];
        const a03 = mat.m[0][3];
        const a10 = mat.m[1][0];
        const a11 = mat.m[1][1];
        const a12 = mat.m[1][2];
        const a13 = mat.m[1][3];
        const a20 = mat.m[2][0];
        const a21 = mat.m[2][1];
        const a22 = mat.m[2][2];
        const a23 = mat.m[2][3];
        const a30 = mat.m[3][0];
        const a31 = mat.m[3][1];
        const a32 = mat.m[3][2];
        const a33 = mat.m[3][3];

        const b00 = a00 * a11 - a01 * a10;
        const b01 = a00 * a12 - a02 * a10;
        const b02 = a00 * a13 - a03 * a10;
        const b03 = a01 * a12 - a02 * a11;
        const b04 = a01 * a13 - a03 * a11;
        const b05 = a02 * a13 - a03 * a12;
        const b06 = a20 * a31 - a21 * a30;
        const b07 = a20 * a32 - a22 * a30;
        const b08 = a20 * a33 - a23 * a30;
        const b09 = a21 * a32 - a22 * a31;
        const b10 = a21 * a33 - a23 * a31;
        const b11 = a22 * a33 - a23 * a32;

        var det =
            b00 * b11 - b01 * b10 + b02 * b09 +
            b03 * b08 - b04 * b07 + b05 * b06;

        det = 1.0 / det;

        return Mat4{
            .m = [4][4]f32{
                .{
                    (a11 * b11 - a12 * b10 + a13 * b09) * det,
                    (a02 * b10 - a01 * b11 - a03 * b09) * det,
                    (a31 * b05 - a32 * b04 + a33 * b03) * det,
                    (a22 * b04 - a21 * b05 - a23 * b03) * det,
                },
                .{
                    (a12 * b08 - a10 * b11 - a13 * b07) * det,
                    (a00 * b11 - a02 * b08 + a03 * b07) * det,
                    (a32 * b02 - a30 * b05 - a33 * b01) * det,
                    (a20 * b05 - a22 * b02 + a23 * b01) * det,
                },
                .{
                    (a10 * b10 - a11 * b08 + a13 * b06) * det,
                    (a01 * b08 - a00 * b10 - a03 * b06) * det,
                    (a30 * b04 - a31 * b02 + a33 * b00) * det,
                    (a21 * b02 - a20 * b04 - a23 * b00) * det,
                },
                .{
                    (a11 * b07 - a10 * b09 - a12 * b06) * det,
                    (a00 * b09 - a01 * b07 + a02 * b06) * det,
                    (a31 * b01 - a30 * b03 - a32 * b00) * det,
                    (a20 * b03 - a21 * b01 + a22 * b00) * det,
                }
            }
        };
    }

    pub fn getDataPtr(matrix: *const Mat4) [*]const f32 {
        return &matrix.m[0][0];
    }

    pub const identity = Mat4{
        .m = [4][4]f32{
            .{1, 0, 0, 0},
            .{0, 1, 0, 0},
            .{0, 0, 1, 0},
            .{0, 0, 0, 1}
        }
    };
};


pub const RGB = struct{

    r: f32 = 0,
    g: f32 = 0,
    b: f32 = 0,

    pub fn new(r: f32, g: f32, b: f32) RGB {
        return .{
            .r = r,
            .g = g,
            .b = b,
        };
    }

    pub fn newGrayScale(gray: f32) RGB {
        return .{
            .r = gray,
            .g = gray,
            .b = gray,
        };
    }

    pub fn scalar(rgb: RGB, scale: f32) RGB {
        return .{
            .r = rgb.r * scale,
            .g = rgb.g * scale,
            .b = rgb.b * scale,
        };
    }

    pub fn toRGBA(rgb: *const RGB, alpha: f32) RGBA {
        return .{
            .r = rgb.r,
            .g = rgb.g,
            .b = rgb.b,
            .a = alpha
        };
    }

};

pub const RGBA = struct {

    r: f32 = 0,
    g: f32 = 0,
    b: f32 = 0,
    a: f32 = 0,

    pub fn new(r: f32, g: f32, b: f32, a: f32) RGBA {
        return .{
            .r = r,
            .g = g,
            .b = b,
            .a = a,
        };
    }

    pub fn newGrayScale(gray: f32) RGBA {
        return .{
            .r = gray,
            .g = gray,
            .b = gray,
            .a = 1,
        };
    }

    pub fn mult(a: *const RGBA, b: *const RGBA) RGBA {
        return .{
            .r = a.r * b.r,
            .g = a.g * b.g,
            .b = a.b * b.b,
            .a = a.a * b.a,
        };
    }

    pub fn toRGB(colour: RGBA) RGB {
        return .{
            .r = colour.r / colour.a,
            .g = colour.g / colour.a,
            .b = colour.b / colour.a,
        };
    }

    pub const white     = RGBA.new(1, 1, 1, 1);
    pub const red       = RGBA.new(1, 0, 0, 1);
    pub const green     = RGBA.new(0, 1, 0, 1);
    pub const blue      = RGBA.new(0, 0, 1, 1);
    pub const black     = RGBA.new(0, 0, 0, 1);

};

pub const Direction = enum {

    north,
    north_east,
    east,
    south_east,
    south,
    south_west,
    west,
    north_west,

    pub const length = std.meta.fields(Direction).len;

    pub const all = [length]Direction{
        .north,
        .north_east,
        .east,
        .south_east,
        .south,
        .south_west,
        .west,
        .north_west,
    };

    pub fn getOpposite(dir: Direction) Direction {
        switch(dir){
            .north => return .south,
            .north_east => return .south_west,
            .east => return .west,
            .south_east => return .north_west,
            .south => return .north,
            .south_west => return .north_east,
            .west => return .east,
            .north_west => return .south_east,
        }
    }

    pub fn fromVec2i(vec: Vec2i) ?Direction {
        if(vec.x > 0){
            if(vec.y > 0){
                return .north_east;
            }else if(vec.y < 0){
                return .south_east;
            }else{
                return .east;
            }
        }else if(vec.x < 0){
            if(vec.y > 0){
                return .north_west;
            }else if(vec.y < 0){
                return .south_west;
            }else{
                return .west;
            }
        }else{
            if(vec.y > 0){
                return .north;
            }else if(vec.y < 0){
                return .south;
            }else{
                return null;
            }
        }
    }

    pub fn fromPosAndTarget(pos: Vec2i, target: Vec2i) ?Direction {
        return fromVec2i(target.sub(pos));
    }

    pub fn fromPosAndTargetUnsafe(pos: Vec2i, target: Vec2i) Direction {
        const dir = fromPosAndTarget(pos, target);
        assert(dir != null);
        return dir.?;
    }

    pub fn toVec2i(dir: Direction) Vec2i {
        switch(dir){
            .north => return Vec2i.new(0, 1),
            .north_east => return Vec2i.new(1, 1),
            .east => return Vec2i.new(1, 0),
            .south_east => return Vec2i.new(1, -1),
            .south => return Vec2i.new(0, -1),
            .south_west => return Vec2i.new(-1, -1),
            .west => return Vec2i.new(-1, 0),
            .north_west => return Vec2i.new(-1, 1),
        }
    }

    pub fn toVec2f(dir: Direction) Vec2i {
        return dir.toVec2i().toVec2f();
    }

    pub fn getComponents(dir: Direction) []const Direction {
        switch(dir){
            .north => return .{.north},
            .north_east => return .{.north, .east, .north_east},
            .east => return .{.east},
            .south_east => return .{.south, .east, .south_east},
            .south => return .{.south},
            .south_west => return .{.south, .west, .south_west},
            .west => return .{.west},
            .north_west => return .{.north, .west, .north_west},
        }
    }

    pub fn toFlags(dir: Direction) Direction.Flags {
        switch(dir){
            .north => return .{.is_north = true},
            .north_east => return .{.is_north = true, .is_east = true},
            .east => return .{.is_east = true},
            .south_east => return .{.is_south = true, .is_east = true},
            .south => return .{.is_south = true},
            .south_west => return .{.is_south = true, .is_west = true},
            .west => return .{.is_west = true},
            .north_west => return .{.is_north = true, .is_west = true},
        }
    }

    pub const Flags = packed struct {

        is_north: bool = false,
        is_east: bool = false,
        is_south: bool = false,
        is_west: bool = false,

        padding: u4 = 0,

    };
};

pub const Cardinal = enum {

    north,
    east,
    south,
    west,

    pub fn getOpposite(cardinal: Cardinal) Cardinal {
        switch(cardinal){
            .north => return .south,
            .east => return .west,
            .south => return .north,
            .west => return .east,
        }
    }

    pub fn toVec2i(cardinal: Cardinal) Vec2i {
        switch(cardinal){
            .north => return Vec2i.new(0, 1),
            .east => return Vec2i.new(1, 0),
            .south => return Vec2i.new(0, -1),
            .west => return Vec2i.new(-1, 0),
        }
    }

    pub fn toVec2f(cardinal: Cardinal) Vec2i {
        return cardinal.toVec2i().toVec2f();
    }
};

test "quickmath" {
    try std.testing.expect(1 + 2 == 3);
}
