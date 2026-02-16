#!/usr/bin/env python3
"""
Comprehensive test script to compare poly_warp.py and polycheck.py implementations.
This script tests all functions to ensure they provide the same functionality.
"""

import numpy as np
import sys
import os
import time

# Add the polycheck directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "polycheck"))


def comparison_imports():
    """Test that both modules can be imported"""
    print("Testing imports...")

    try:
        import polycheck as pc

        print("✓ Successfully imported polycheck")
    except ImportError as e:
        print(f"✗ Failed to import polycheck: {e}")
        return False, None, None

    try:
        import poly_warp as pw

        print("✓ Successfully imported poly_warp")
    except ImportError as e:
        print(f"✗ Failed to import poly_warp: {e}")
        return False, None, None

    return True, pc, pw


def compare_arrays(arr1, arr2, tolerance=1e-5, name="arrays"):
    """Compare two numpy arrays with given tolerance"""
    if arr1.shape != arr2.shape:
        print(f"✗ {name} shape mismatch: {arr1.shape} vs {arr2.shape}")
        return False

    diff = np.abs(arr1 - arr2)
    max_diff = np.max(diff)

    if max_diff > tolerance:
        print(f"✗ {name} values differ by {max_diff:.2e} (tolerance: {tolerance:.2e})")
        print(
            f"   Number of differing elements: {np.sum(diff > tolerance)}/{arr1.size}"
        )
        return False
    else:
        print(f"✓ {name} match within tolerance ({max_diff:.2e})")
        return True


def comparison_contains(pc, pw):
    """Test the contains function"""
    print("\nTesting contains function...")

    # Test case 1: Simple square polygon
    poly1 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    points1 = np.array(
        [
            [0.5, 0.5],  # inside
            [1.5, 0.5],  # outside
            [0.0, 0.0],  # on boundary
            [0.25, 0.75],  # inside
            [-0.1, 0.5],  # outside
            [0.999, 0.999],  # inside, near boundary
        ],
        dtype=np.float32,
    )

    try:
        pc_result1 = pc.contains(poly1, points1)
        pw_result1 = pw.contains(poly1, points1)

        success1 = compare_arrays(pc_result1, pw_result1, name="contains (square)")

        # Test case 2: Complex polygon from the example
        poly2 = np.array(
            [
                [5.0, 5.0],
                [0.0, 0.5],
                [5.0, -5.0],
                [0.5, -0.5],
                [-5.0, -5.0],
                [0.0, -0.5],
                [-5.0, 5.0],
                [-0.5, 0.5],
            ],
            dtype=np.float32,
        )

        # Create a grid of test points
        dots = np.linspace(-8, 8, 50)  # Smaller grid for faster testing
        xs, ys = np.meshgrid(dots, dots)
        points2 = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)

        pc_result2 = pc.contains(poly2, points2)
        pw_result2 = pw.contains(poly2, points2)

        success2 = compare_arrays(pc_result2, pw_result2, name="contains (complex)")

        return success1 and success2

    except Exception as e:
        print(f"✗ Contains test failed with exception: {e}")
        return False


def comparison_visibility(pc, pw):
    """Test the visibility function"""
    print("\nTesting visibility function...")

    # Create a 10x10 grid with some obstacles
    data = np.zeros((10, 10), dtype=np.float32)
    data[4:6, 4:6] = 0.8  # obstacle block
    data[2, 7] = 1.0  # single obstacle

    start = np.array([0, 0], dtype=np.int32)

    # Test various end points
    ends = np.array([[9, 9], [5, 5], [2, 7], [1, 1], [8, 3]], dtype=np.int32)

    try:
        pc_result = pc.visibility(data, start, ends)
        pw_result = pw.visibility(data, start, ends)

        return compare_arrays(pc_result, pw_result, name="visibility")

    except Exception as e:
        print(f"✗ Visibility test failed with exception: {e}")
        return False


def comparison_visibility_from_region(pc, pw):
    """Test the visibility_from_region function"""
    print("\nTesting visibility_from_region function...")

    # Create a smaller grid for faster testing
    data = np.zeros((8, 8), dtype=np.float32)
    data[3:5, 3:5] = 0.5  # obstacle block

    starts = np.array([[0, 0], [7, 0], [0, 7]], dtype=np.int32)

    ends = np.array([[7, 7], [4, 4], [2, 6], [6, 2]], dtype=np.int32)

    try:
        pc_result = pc.visibility_from_region(data, starts, ends)
        pw_result = pw.visibility_from_region(data, starts, ends)

        return compare_arrays(pc_result, pw_result, name="visibility_from_region")

    except Exception as e:
        print(f"✗ visibility_from_region test failed with exception: {e}")
        return False


def comparison_visibility_from_real_region(pc, pw):
    """Test the visibility_from_real_region function"""
    print("\nTesting visibility_from_real_region function...")

    # Create a small grid
    data = np.zeros((6, 6), dtype=np.float32)
    data[2:4, 2:4] = 0.7  # obstacle block

    origin = [0.0, 0.0]
    resolution = 1.0

    starts = np.array([[0.5, 0.5], [5.5, 0.5], [0.5, 5.5]], dtype=np.float32)

    ends = np.array([[5.5, 5.5], [3.5, 3.5], [2.5, 4.5]], dtype=np.float32)

    try:
        pc_result = pc.visibility_from_real_region(
            data, origin, resolution, starts, ends
        )
        pw_result = pw.visibility_from_real_region(
            data, origin, resolution, starts, ends
        )

        return compare_arrays(pc_result, pw_result, name="visibility_from_real_region")

    except Exception as e:
        print(f"✗ visibility_from_real_region test failed with exception: {e}")
        return False


def comparison_faux_scan(pc, pw):
    """Test the faux_scan function"""
    print("\nTesting faux_scan function...")

    # Create simple polygons - triangle and square
    polygons = [
        np.array([[2.0, 2.0], [3.0, 2.0], [2.5, 3.0]], dtype=np.float32),
        np.array([[4.0, 1.0], [5.0, 1.0], [5.0, 2.0], [4.0, 2.0]], dtype=np.float32),
    ]

    origin = [0.0, 0.0]
    angle_start = 0.0
    angle_inc = np.pi / 180  # 1 degree increments
    num_rays = 90  # Quarter circle for faster testing
    max_range = 10.0
    resolution = 0.1

    try:
        pc_result = pc.faux_scan(
            polygons, origin, angle_start, angle_inc, num_rays, max_range, resolution
        )
        pw_result = pw.faux_scan(
            polygons, origin, angle_start, angle_inc, num_rays, max_range, resolution
        )

        return compare_arrays(pc_result, pw_result, name="faux_scan")

    except Exception as e:
        print(f"✗ faux_scan test failed with exception: {e}")
        return False


def benchmark_performance(pc, pw):
    """Benchmark performance comparison between implementations"""
    print("\nPerformance benchmarking...")

    # Test contains performance
    poly = np.array(
        [
            [5.0, 5.0],
            [0.0, 0.5],
            [5.0, -5.0],
            [0.5, -0.5],
            [-5.0, -5.0],
            [0.0, -0.5],
            [-5.0, 5.0],
            [-0.5, 0.5],
        ],
        dtype=np.float32,
    )

    dots = np.linspace(-8, 8, 100)
    xs, ys = np.meshgrid(dots, dots)
    points = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)

    # Benchmark contains
    start_time = time.time()
    _ = pc.contains(poly, points)
    pc_time = time.time() - start_time

    start_time = time.time()
    _ = pw.contains(poly, points)
    pw_time = time.time() - start_time

    print("Contains performance:")
    print(f"  PyCUDA: {pc_time:.4f}s")
    print(f"  Warp:   {pw_time:.4f}s")
    print(f"  Speedup: {pc_time/pw_time:.2f}x" if pw_time > 0 else "  N/A")


def main():
    """Run all comparison tests"""
    print("Comprehensive comparison test: poly_warp.py vs polycheck.py")
    print("=" * 60)

    # Test imports
    success, pc, pw = comparison_imports()
    if not success:
        return 1

    # Run functional tests
    tests = [
        ("contains", lambda: comparison_contains(pc, pw)),
        ("visibility", lambda: comparison_visibility(pc, pw)),
        ("visibility_from_region", lambda: comparison_visibility_from_region(pc, pw)),
        (
            "visibility_from_real_region",
            lambda: comparison_visibility_from_real_region(pc, pw),
        ),
        ("faux_scan", lambda: comparison_faux_scan(pc, pw)),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} test PASSED")
            else:
                print(f"✗ {test_name} test FAILED")
        except Exception as e:
            print(f"✗ {test_name} test FAILED with exception: {e}")
        print()

    # Performance benchmarking
    try:
        benchmark_performance(pc, pw)
    except Exception as e:
        print(f"Benchmarking failed: {e}")

    # Summary
    print("=" * 60)
    print(f"SUMMARY: {passed}/{total} tests passed")

    if passed == total:
        print(
            "🎉 SUCCESS: poly_warp.py provides the same functionality as polycheck.py!"
        )
        return 0
    else:
        print("⚠️  ISSUES: Some tests failed. Check implementation details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
