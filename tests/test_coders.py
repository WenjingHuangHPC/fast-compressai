import json
import unittest

from fast_compressai.coders import (
    Bitstream,
    BitstreamError,
    CoderInputError,
    CoderStateError,
    CompressAIANSBackend,
)


class CoderTests(unittest.TestCase):
    def setUp(self):
        self.cdfs = [[0, 32768, 65536]]
        self.indexes = [0, 0, 0, 0]
        self.offsets = [0]

    def _backend(self):
        try:
            return CompressAIANSBackend()
        except ImportError as exc:
            self.skipTest(str(exc))

    def test_round_trip(self):
        self.backend = self._backend()
        symbols = [0, 1, 0, 1]
        session = self.backend.create_session(symbol_count=len(symbols), shape=(2, 2))
        session.encode(symbols, self.indexes, self.cdfs, self.offsets)
        bitstream = session.finalize()
        self.assertEqual(self.backend.decode(bitstream, indexes=self.indexes, cdfs=self.cdfs, offsets=self.offsets), symbols)

    def test_empty_round_trip(self):
        self.backend = self._backend()
        session = self.backend.create_session(symbol_count=0)
        bitstream = session.finalize()
        self.assertEqual(self.backend.decode(bitstream, indexes=[], cdfs=self.cdfs, offsets=self.offsets), [])

    def test_bitstream_json_round_trip(self):
        original = Bitstream("test", b"payload", 2, shape=(1, 2), metadata={"kind": "test", "nested": [1, True]})
        restored = Bitstream.from_dict(json.loads(json.dumps(original.to_dict())))
        self.assertEqual(restored, original)

    def test_bitstream_rejects_coerced_values_and_non_json_metadata(self):
        for kwargs in (
            {"symbol_count": True},
            {"shape": (1.5, 2)},
            {"shape": ("2",)},
            {"schema_version": True},
            {"metadata": {1: "non-string key"}},
            {"metadata": {"value": float("inf")}},
            {"metadata": {"value": object()}},
        ):
            values = {"backend": "test", "payload": b"", "symbol_count": 0}
            values.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(BitstreamError):
                Bitstream(**values)

    def test_bitstream_rejects_bad_base64(self):
        value = Bitstream("test", b"data", 0).to_dict()
        value["payload"] = "%%"
        with self.assertRaises(BitstreamError):
            Bitstream.from_dict(value)

    def test_invalid_symbol_count_and_cdf_are_rejected(self):
        self.backend = self._backend()
        session = self.backend.create_session(symbol_count=1)
        with self.assertRaises(CoderInputError):
            session.encode([0, 1], [0, 0], self.cdfs, self.offsets)
        session = self.backend.create_session(symbol_count=1)
        with self.assertRaises(CoderInputError):
            session.encode([0], [0], [[1, 2]], self.offsets)
        for invalid in ([[0, 1, 2]], [[0, 32768, 65535]]):
            with self.subTest(cdf=invalid), self.assertRaises(CoderInputError):
                self.backend.create_session(symbol_count=1).encode([0], [0], invalid, [0])
        with self.assertRaises(CoderInputError):
            self.backend.create_session(symbol_count=1).encode([False], [0], self.cdfs, self.offsets)
        escaped = self.backend.create_session(symbol_count=1)
        escaped.encode([2], [0], self.cdfs, self.offsets)
        stream = escaped.finalize()
        self.assertEqual(self.backend.decode(stream, indexes=[0], cdfs=self.cdfs, offsets=[0]), [2])

    def test_session_cannot_be_reused_after_finalize(self):
        self.backend = self._backend()
        session = self.backend.create_session(symbol_count=0)
        session.finalize()
        with self.assertRaises(CoderStateError):
            session.finalize()
        with self.assertRaises(CoderStateError):
            session.encode([], [], self.cdfs, self.offsets)

    def test_encode_is_one_shot_and_failures_poison_session(self):
        self.backend = self._backend()
        session = self.backend.create_session(symbol_count=1)
        session.encode([1], [0], self.cdfs, self.offsets)
        with self.assertRaises(CoderStateError):
            session.encode([0], [0], self.cdfs, self.offsets)

        failed = self.backend.create_session(symbol_count=2)
        with self.assertRaises(CoderInputError):
            failed.encode([0], [0], self.cdfs, self.offsets)
        with self.assertRaises(CoderStateError):
            failed.finalize()

    def test_nonempty_session_must_encode_before_finalize(self):
        self.backend = self._backend()
        session = self.backend.create_session(symbol_count=1)
        with self.assertRaises(CoderStateError):
            session.finalize()

    def test_backend_and_schema_validation(self):
        self.backend = self._backend()
        stream = Bitstream(self.backend.name, b"x", 0)
        with self.assertRaises(CoderInputError):
            self.backend.decode(Bitstream("other", b"x", 0), indexes=[], cdfs=self.cdfs, offsets=self.offsets)
        payload = stream.to_dict()
        payload["schema_version"] = 99
        with self.assertRaises(BitstreamError):
            Bitstream.from_dict(payload)

    def test_decode_metadata_length_is_checked(self):
        self.backend = self._backend()
        session = self.backend.create_session(symbol_count=1)
        session.encode([0], [0], self.cdfs, self.offsets)
        stream = session.finalize()
        with self.assertRaises(CoderInputError):
            self.backend.decode(stream, indexes=[], cdfs=self.cdfs, offsets=self.offsets)

    def test_decode_checks_coding_fingerprint(self):
        self.backend = self._backend()
        session = self.backend.create_session(symbol_count=1)
        session.encode([0], [0], self.cdfs, self.offsets)
        stream = session.finalize()
        self.assertEqual(self.backend.decode(stream, indexes=[0], cdfs=self.cdfs, offsets=[0]), [0])
        with self.assertRaisesRegex(BitstreamError, "fingerprint"):
            self.backend.decode(stream, indexes=[0], cdfs=[[0, 10000, 65536]], offsets=[0])


if __name__ == "__main__":
    unittest.main()
