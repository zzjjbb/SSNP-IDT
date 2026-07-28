from unittest import TestCase
import warnings

import numpy as np

from ssnp.utils.utils import Config
from ssnp.utils.array_pool import ManagedObj, ManagedArrayPool


class TestConfig(TestCase):
    def test_default(self):
        config = Config()
        self.assertListEqual(config._callbacks, [])
        with self.assertRaisesRegex(AttributeError, "res is uninitialized"):
            print(config.res)
        with self.assertRaisesRegex(AttributeError, "xyz pixel sizes are uninitialized"):
            print(config.xyz)
        with self.assertRaisesRegex(AttributeError, "wavelength lambda0 is uninitialized"):
            print(config.lambda0)
        self.assertEqual(config.n0, 1.0)

    def test_setter(self):
        config = Config()
        # normal assignment
        config.xyz = (3, 4, 5)
        self.assertTupleEqual(config.xyz, (3, 4, 5))
        for pix in config.xyz:
            self.assertIsInstance(pix, float)
        # bad assignment
        with self.assertRaises(AssertionError):
            config.xyz = (1, 2)
        with self.assertRaisesRegex(ValueError, 'could not convert string to float'):
            config.xyz = 'abc'
        with self.assertRaisesRegex(TypeError, "not iterable"):
            config.xyz = 1
        # do not change value if assignment fails
        self.assertTupleEqual(config.xyz, (3, 4, 5))
        # accept iterable
        for new_value in (range(4, 7), np.arange(4, 7), iter([4, 5, 6])):
            config.xyz = new_value
            self.assertTupleEqual(config.xyz, (4, 5, 6))
            config.xyz = (3, 4, 5)


class MockArray:
    id_counter = 0
    freed_obj = 0

    @classmethod
    def get_unique_id(cls):
        cls.id_counter += 1
        return cls.id_counter

    @classmethod
    def bump_freed(cls):
        cls.freed_obj += 1

    @classmethod
    def clear_counters(cls):
        cls.id_counter = cls.freed_obj = 0

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.unique_id = self.get_unique_id()

    @classmethod
    def empty_like(cls, other):
        return cls(other.shape, other.dtype)

    def method_return_self(self):
        return self

    def __iadd__(self, _):
        return self

    def __del__(self):
        self.bump_freed()

    @property
    def self(self):
        return self


class TestArrayPool(TestCase):
    def test_proxy(self):
        def helper(wrapper, expected):
            def type_assertion(i):
                self.assertIsInstance(a, MockArray)
                if expected[i]:
                    self.assertIsInstance(a, ManagedObj)
                else:
                    self.assertNotIsInstance(a, ManagedObj)

            array_pool = ManagedArrayPool(
                arr_proto, MockArray.empty_like, unique_id=lambda arr: arr.unique_id,
                method_wrapper=wrapper
            )
            # 0 - basic
            a = array_pool.get()
            type_assertion(0)
            # 1 - i-operator
            a += 1
            type_assertion(1)
            # 2 - explicit i-operator
            a = array_pool.get().__iadd__(2)
            type_assertion(2)
            # 3 - dynamic callable attr
            a = array_pool.get()
            a.dynamic_return_self = lambda x: x
            a = a.dynamic_return_self(a)
            type_assertion(3)
            # 4 - bound method
            a = array_pool.get().method_return_self().method_return_self().method_return_self()
            type_assertion(4)
            # 5 - getter property
            a = array_pool.get().self
            type_assertion(5)
            # 6 - class method (should NOT be proxied)
            a = array_pool.get()
            a = a.empty_like(a)
            type_assertion(6)

        arr_proto = MockArray([1024, 1024], float)
        helper(None, [True, True, True, True, False, False, False])
        helper('chainable', [True, True, True, True, True, False, False])
        helper('descriptor', [True, True, True, True, True, True, False])

    def test_pool(self):
        arr_proto = MockArray([1024, 1024], float)
        MockArray.clear_counters()
        array_pool = ManagedArrayPool(arr_proto, MockArray.empty_like, unique_id=lambda arr: arr.unique_id,
                                      method_wrapper='chainable')

        # Test basic get
        self.assertEqual(array_pool.allocated_count, 0)
        a = array_pool.get()
        self.assertEqual(array_pool.allocated_count, 1)
        self.assertIsInstance(a, MockArray)
        self.assertIsInstance(a, ManagedObj)

        # Test basic recycle
        arr_addr = id(a.__wrapped__)
        self.assertEqual(len(array_pool._pool), 0)
        a.dispose()
        self.assertEqual(len(array_pool._pool), 1)
        self.assertEqual(array_pool.allocated_count, 1)
        for i in range(10):
            del a
            a = array_pool.get()
        self.assertEqual(len(array_pool._pool), 0)
        self.assertEqual(array_pool.allocated_count, 1)
        self.assertEqual(id(a.__wrapped__), arr_addr)
        self.assertIsInstance(a, MockArray)
        self.assertIsInstance(a, ManagedObj)
        a = []
        a.append(array_pool.get())
        self.assertEqual(id(a[0].__wrapped__), arr_addr)

        # More recycle
        del a
        self.assertEqual(len(array_pool._pool), 1)
        self.assertEqual(array_pool.allocated_count, 1)
        self.assertEqual(MockArray.id_counter, 1)
        self.assertEqual(MockArray.freed_obj, 0)
        del array_pool
        self.assertEqual(MockArray.freed_obj, 1)
        array_pool = ManagedArrayPool(arr_proto, MockArray.empty_like, unique_id=lambda arr: arr.unique_id)
        a = array_pool.get()
        del array_pool
        self.assertEqual(MockArray.id_counter, 2)
        self.assertEqual(MockArray.freed_obj, 1)
        del a
        self.assertEqual(MockArray.freed_obj, 2)
