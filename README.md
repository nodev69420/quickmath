# Quickmath Library
A very slow zig math library.

# To Use
Run the command on project root:
`zig fetch --save git+https://github.com/nodev69420/quickmath/#HEAD`

Then add to your `build.zig`:
```zig
 const qmath_mod = b.dependency("quickmath", .{});
 exe.root_module.addImport("quickmath", qmath_mod.module("quickmath"));
```

# Contains
- Real Math
- Vector2 (Floating Point)
- Vector3 (Floating Point)
- Vector4 (Floating Point)
- Quaternions
- Vector2 (Integer)
- Rectangle (Integer)
- Rectangle (Floating Point)
- UV Coords
- Matrix4x4
- Colour RGB
- Colour RGBA
- Direction
- Cardinal
