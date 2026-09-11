from __future__ import annotations

from .corpus import CorpusTask

CORPUS2_VERSION = "semantic-repair-v1"

# Symptom-style statements throughout: the model sees misbehavior + contract,
# never the fault label. Hidden tests establish the invariant.


def _v2(
    task_id: str,
    title: str,
    family: str,
    stratum: str,
    problem_statement: str,
    starter_code: str,
    hidden_tests: str,
    reference_solution: str,
    expected_note: str,
    support_files: tuple[tuple[str, str], ...] = (),
) -> CorpusTask:
    return CorpusTask(
        task_id=task_id,
        title=title,
        fault_class=family,
        problem_statement=problem_statement,
        starter_code=starter_code,
        hidden_tests=hidden_tests,
        reference_solution=reference_solution,
        expected_note=expected_note,
        stratum=stratum,
        family=family,
        support_files=support_files,
    )


def _local_tasks() -> tuple[CorpusTask, ...]:
    return (
        _v2(
            "v2-dedup-order",
            "Deduplicated order",
            "deduplication",
            "local",
            "dedupe(xs) returns unique elements but scrambles their original order. "
            "Contract: unique elements in first-occurrence order.",
            "def dedupe(xs):\n    return list(set(xs))\n",
            "from candidate import dedupe\n"
            "assert dedupe([3, 1, 2, 1, 3]) == [3, 1, 2]\n"
            "assert dedupe(['b', 'a', 'b']) == ['b', 'a']\n"
            "assert dedupe([]) == []\nprint('hidden OK')\n",
            "def dedupe(xs):\n    return list(dict.fromkeys(xs))\n",
            "Uniqueness must not destroy order.",
        ),
        _v2(
            "v2-half-up-rounding",
            "Half away from zero",
            "rounding",
            "local",
            "round_half_up(x) maps 2.5 to 2. Contract: halves round away from zero "
            "(2.5 -> 3, -2.5 -> -3).",
            "def round_half_up(x):\n    return round(x)\n",
            "from candidate import round_half_up\n"
            "assert round_half_up(2.5) == 3\nassert round_half_up(3.5) == 4\n"
            "assert round_half_up(-2.5) == -3\nassert round_half_up(2.4) == 2\n"
            "print('hidden OK')\n",
            "import math\n"
            "def round_half_up(x):\n"
            "    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)\n",
            "Banker's rounding is the wrong rule here.",
        ),
        _v2(
            "v2-splitlines-cr",
            "Line splitting",
            "line-splitting",
            "local",
            "lines(text) leaves trailing carriage returns on Windows-style input. "
            "Contract: split on any line boundary, no leftover control characters.",
            "def lines(text):\n    return text.split('\\n')\n",
            "from candidate import lines\n"
            "assert lines('a\\r\\nb\\r\\n') == ['a', 'b']\n"
            "assert lines('a\\nb') == ['a', 'b']\n"
            "assert lines('solo') == ['solo']\nprint('hidden OK')\n",
            "def lines(text):\n    return text.splitlines()\n",
            "Split on boundaries, not on one character.",
        ),
        _v2(
            "v2-merge-precedence",
            "Override precedence",
            "precedence",
            "local",
            "merge(base, override) lets base values win on key conflicts. "
            "Contract: override wins.",
            "def merge(base, override):\n    return {**override, **base}\n",
            "from candidate import merge\n"
            "assert merge({'a': 1}, {'a': 2}) == {'a': 2}\n"
            "assert merge({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}\n"
            "print('hidden OK')\n",
            "def merge(base, override):\n    return {**base, **override}\n",
            "Unpack order decides precedence.",
        ),
        _v2(
            "v2-top-k",
            "Largest three",
            "selection",
            "local",
            "top3(xs) returns the three smallest values. Contract: three largest, descending.",
            "def top3(xs):\n    return sorted(xs)[:3]\n",
            "from candidate import top3\n"
            "assert top3([5, 1, 4, 2, 3]) == [5, 4, 3]\n"
            "assert top3([1, 2]) == [2, 1]\nprint('hidden OK')\n",
            "def top3(xs):\n    return sorted(xs, reverse=True)[:3]\n",
            "Wrong end of the ordering.",
        ),
        _v2(
            "v2-chunk-remainder",
            "Chunk remainder",
            "chunking",
            "local",
            "chunks(xs, n) silently drops the final partial group. "
            "Contract: the last chunk may be shorter but must be present.",
            "def chunks(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs) - len(xs) % n, n)]\n",
            "from candidate import chunks\n"
            "assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n"
            "assert chunks([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]\n"
            "assert chunks([1], 3) == [[1]]\nprint('hidden OK')\n",
            "def chunks(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs), n)]\n",
            "Remainder is data, not waste.",
        ),
        _v2(
            "v2-count-ci",
            "Case-insensitive count",
            "case-folding",
            "local",
            "count_word(text, word) misses capitalized occurrences. "
            "Contract: matching ignores case.",
            "def count_word(text, word):\n    return text.split().count(word)\n",
            "from candidate import count_word\n"
            "assert count_word('Hi hi HI', 'hi') == 3\n"
            "assert count_word('a A b', 'b') == 1\nprint('hidden OK')\n",
            "def count_word(text, word):\n"
            "    return [w.lower() for w in text.split()].count(word.lower())\n",
            "Fold case on both sides.",
        ),
        _v2(
            "v2-add-days",
            "Calendar addition",
            "date-arithmetic",
            "local",
            "add_days(y, m, d, n) produces day=32 instead of rolling into the next month. "
            "Contract: real calendar arithmetic, returned as a (y, m, d) tuple.",
            "def add_days(y, m, d, n):\n    return (y, m, d + n)\n",
            "from candidate import add_days\n"
            "assert add_days(2026, 1, 31, 1) == (2026, 2, 1)\n"
            "assert add_days(2024, 2, 28, 1) == (2024, 2, 29)\n"
            "assert add_days(2026, 5, 5, 0) == (2026, 5, 5)\nprint('hidden OK')\n",
            "import datetime\n"
            "def add_days(y, m, d, n):\n"
            "    out = datetime.date(y, m, d) + datetime.timedelta(days=n)\n"
            "    return (out.year, out.month, out.day)\n",
            "Calendars roll over; integers do not.",
        ),
        _v2(
            "v2-strip-query",
            "Strip query, keep fragment",
            "url-parsing",
            "local",
            "strip_url(url) removes the #fragment but leaves the ?query. "
            "Contract: drop the query string, keep path and fragment.",
            "def strip_url(url):\n    return url.split('#')[0]\n",
            "from candidate import strip_url\n"
            "assert strip_url('/p?a=1#s') == '/p#s'\n"
            "assert strip_url('/p#s') == '/p#s'\n"
            "assert strip_url('/p?a=1') == '/p'\nprint('hidden OK')\n",
            "def strip_url(url):\n"
            "    head, _, frag = url.partition('#')\n"
            "    head = head.split('?')[0]\n"
            "    return head + ('#' + frag if frag else '')\n",
            "Wrong delimiter was stripped.",
        ),
        _v2(
            "v2-interleave",
            "Interleave tails",
            "interleave",
            "local",
            "interleave(a, b) drops trailing elements of the longer list. "
            "Contract: leftover elements are appended in order.",
            "def interleave(a, b):\n    return [x for pair in zip(a, b) for x in pair]\n",
            "from candidate import interleave\n"
            "assert interleave([1, 2, 3], [9]) == [1, 9, 2, 3]\n"
            "assert interleave([1], [7, 8]) == [1, 7, 8]\n"
            "assert interleave([1, 2], [3, 4]) == [1, 3, 2, 4]\nprint('hidden OK')\n",
            "def interleave(a, b):\n"
            "    out = [x for pair in zip(a, b) for x in pair]\n"
            "    out.extend(a[len(b):] if len(a) > len(b) else b[len(a):])\n"
            "    return out\n",
            "zip stops; tails still matter.",
        ),
    )


def _semantic_tasks() -> tuple[CorpusTask, ...]:
    return (
        _v2(
            "v2-percent-change",
            "Percent change base",
            "percent-base",
            "semantic",
            "pct_change(old, new) reports -100.0 when a value halves from 200 to 100. "
            "Contract: change relative to the old value (-50.0).",
            "def pct_change(old, new):\n    return (new - old) / new * 100\n",
            "from candidate import pct_change\n"
            "assert pct_change(200, 100) == -50.0\n"
            "assert pct_change(100, 150) == 50.0\n"
            "assert pct_change(80, 80) == 0.0\nprint('hidden OK')\n",
            "def pct_change(old, new):\n    return (new - old) / old * 100\n",
            "Change is relative to where you started.",
        ),
        _v2(
            "v2-median-empty",
            "Median of nothing",
            "empty-contract",
            "semantic",
            "median([]) returns 0.0. Contract: empty input has no median — raise ValueError.",
            "def median(xs):\n    s = sorted(xs)\n    if not s:\n        return 0.0\n"
            "    n = len(s)\n    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2\n",
            "from candidate import median\n"
            "assert median([3, 1, 2]) == 2\nassert median([1, 2, 3, 4]) == 2.5\n"
            "try:\n    median([])\nexcept ValueError:\n    pass\n"
            "else:\n    raise AssertionError('expected ValueError for empty input')\n"
            "print('hidden OK')\n",
            "def median(xs):\n    s = sorted(xs)\n    if not s:\n        raise ValueError('median of empty sequence')\n"
            "    n = len(s)\n    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2\n",
            "A sentinel is not an answer.",
        ),
        _v2(
            "v2-unique-token",
            "Token uniqueness",
            "uniqueness",
            "semantic",
            "session_token() returns equal tokens for consecutive calls in the same process. "
            "Contract: every call yields a distinct token.",
            "import os\ndef session_token():\n    return f'session-{os.getpid()}'\n",
            "from candidate import session_token\n"
            "a = session_token()\nb = session_token()\n"
            "assert a != b, 'tokens must be unique per call'\n"
            "assert isinstance(a, str) and a\nprint('hidden OK')\n",
            "import itertools, os\n_counter = itertools.count()\n"
            "def session_token():\n    return f'session-{os.getpid()}-{next(_counter)}'\n",
            "Identity must vary per call, not per process.",
        ),
        _v2(
            "v2-deep-merge",
            "Nested merge",
            "deep-merge",
            "semantic",
            "merge_cfg(base, override) drops nested keys that override does not mention. "
            "Contract: nested dicts merge recursively; inputs are never mutated.",
            "def merge_cfg(base, override):\n    return {**base, **override}\n",
            "from candidate import merge_cfg\n"
            "base = {'db': {'host': 'a', 'port': 1}}\n"
            "out = merge_cfg(base, {'db': {'port': 2}})\n"
            "assert out == {'db': {'host': 'a', 'port': 2}}\n"
            "assert base == {'db': {'host': 'a', 'port': 1}}, 'inputs must not mutate'\n"
            "print('hidden OK')\n",
            "def merge_cfg(base, override):\n"
            "    out = dict(base)\n"
            "    for key, value in override.items():\n"
            "        if key in out and isinstance(out[key], dict) and isinstance(value, dict):\n"
            "            out[key] = merge_cfg(out[key], value)\n"
            "        else:\n"
            "            out[key] = value\n"
            "    return out\n",
            "Shallow union destroys nested structure.",
        ),
        _v2(
            "v2-csv-line",
            "Quoted commas",
            "csv-quoting",
            "semantic",
            "parse_csv_line('a,\"b,c\",d') yields three-plus fragments instead of three fields. "
            "Contract: commas inside quotes do not separate fields.",
            "def parse_csv_line(line):\n    return line.split(',')\n",
            "from candidate import parse_csv_line\n"
            "assert parse_csv_line('a,\"b,c\",d') == ['a', 'b,c', 'd']\n"
            "assert parse_csv_line('x,y') == ['x', 'y']\n"
            "assert parse_csv_line('\"a\"\"b\",c') == ['a\"b', 'c']\nprint('hidden OK')\n",
            "import csv\ndef parse_csv_line(line):\n    return next(csv.reader([line]))\n",
            "CSV is a grammar, not a delimiter.",
        ),
        _v2(
            "v2-lru-evict",
            "Eviction end",
            "lru-eviction",
            "semantic",
            "LRUCache(2) evicts the entry just used instead of the least-recently-used one. "
            "Contract: on overflow the least-recently-used entry goes.",
            "from collections import OrderedDict\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity):\n        self.cap = capacity\n        self.data = OrderedDict()\n"
            "    def get(self, key):\n"
            "        if key not in self.data:\n            return None\n"
            "        self.data.move_to_end(key)\n        return self.data[key]\n"
            "    def put(self, key, value):\n"
            "        if key in self.data:\n            self.data.move_to_end(key)\n"
            "        self.data[key] = value\n"
            "        if len(self.data) > self.cap:\n"
            "            self.data.popitem(last=True)\n",
            "from candidate import LRUCache\n"
            "c = LRUCache(2)\nc.put('a', 1)\nc.put('b', 2)\nassert c.get('a') == 1\n"
            "c.put('c', 3)\nassert c.get('b') is None, 'b was least-recently-used'\n"
            "assert c.get('a') == 1\nassert c.get('c') == 3\nprint('hidden OK')\n",
            "from collections import OrderedDict\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity):\n        self.cap = capacity\n        self.data = OrderedDict()\n"
            "    def get(self, key):\n"
            "        if key not in self.data:\n            return None\n"
            "        self.data.move_to_end(key)\n        return self.data[key]\n"
            "    def put(self, key, value):\n"
            "        if key in self.data:\n            self.data.move_to_end(key)\n"
            "        self.data[key] = value\n"
            "        if len(self.data) > self.cap:\n"
            "            self.data.popitem(last=False)\n",
            "Evict the cold end, not the hot one.",
        ),
        _v2(
            "v2-size-format",
            "Binary units",
            "unit-format",
            "semantic",
            "fmt_size(1024) renders '1.0 KB'. Contract: binary units — 1024 bytes is '1.0 KiB', "
            "values below 1024 render as whole bytes.",
            "def fmt_size(n):\n    return f'{n / 1000:.1f} KB'\n",
            "from candidate import fmt_size\n"
            "assert fmt_size(1024) == '1.0 KiB'\nassert fmt_size(1536) == '1.5 KiB'\n"
            "assert fmt_size(500) == '500 B'\nassert fmt_size(1048576) == '1.0 MiB'\n"
            "print('hidden OK')\n",
            "def fmt_size(n):\n"
            "    if n < 1024:\n        return f'{n} B'\n"
            "    for unit in ('KiB', 'MiB', 'GiB'):\n"
            "        n = n / 1024\n"
            "        if n < 1024:\n            return f'{n:.1f} {unit}'\n"
            "    return f'{n:.1f} TiB'\n",
            "1000 and 1024 are different contracts.",
        ),
        _v2(
            "v2-stable-priority",
            "Tie order",
            "tie-order",
            "semantic",
            "schedule() returns equal-priority tasks latest-submitted-first. "
            "Contract: highest priority first; ties in submission (FIFO) order.",
            "def schedule(tasks):\n"
            "    out = []\n    remaining = list(tasks)\n"
            "    while remaining:\n"
            "        top = max(t[0] for t in remaining)\n"
            "        for i in range(len(remaining) - 1, -1, -1):\n"
            "            if remaining[i][0] == top:\n"
            "                out.append(remaining.pop(i)[1])\n"
            "                break\n"
            "    return out\n",
            "from candidate import schedule\n"
            "assert schedule([(1, 'a'), (1, 'b'), (2, 'c')]) == ['c', 'a', 'b']\n"
            "assert schedule([(3, 'x'), (2, 'y')]) == ['x', 'y']\n"
            "assert schedule([]) == []\nprint('hidden OK')\n",
            "def schedule(tasks):\n"
            "    out = []\n    remaining = list(tasks)\n"
            "    while remaining:\n"
            "        top = max(t[0] for t in remaining)\n"
            "        for i in range(len(remaining)):\n"
            "            if remaining[i][0] == top:\n"
            "                out.append(remaining.pop(i)[1])\n"
            "                break\n"
            "    return out\n",
            "Ties break toward the earliest claimant.",
        ),
        _v2(
            "v2-dt-roundtrip",
            "Datetime roundtrip",
            "roundtrip-codec",
            "semantic",
            "pack()/unpack() lose the seconds (and microseconds) of a datetime. "
            "Contract: unpack(pack(dt)) == dt for any naive datetime.",
            "def pack(dt):\n    return dt.strftime('%Y-%m-%d %H:%M')\n"
            "def unpack(s):\n    import datetime\n    return datetime.datetime.strptime(s, '%Y-%m-%d %H:%M')\n",
            "from candidate import pack, unpack\nimport datetime\n"
            "dt = datetime.datetime(2026, 1, 2, 3, 4, 5, 123456)\n"
            "assert unpack(pack(dt)) == dt\n"
            "assert unpack(pack(datetime.datetime(2026, 1, 2))) == datetime.datetime(2026, 1, 2)\n"
            "print('hidden OK')\n",
            "def pack(dt):\n    return dt.isoformat()\n"
            "def unpack(s):\n    import datetime\n    return datetime.datetime.fromisoformat(s)\n",
            "Truncating formats cannot round-trip.",
        ),
        _v2(
            "v2-max-discount",
            "Single best discount",
            "discount-semantics",
            "semantic",
            "total_with_discount(100, [20, 10]) charges 72.0. "
            "Contract: apply the single best discount only (80.0); no discounts means full price.",
            "def total_with_discount(price, discounts):\n"
            "    total = price\n"
            "    for pct in discounts:\n"
            "        total = total * (1 - pct / 100)\n"
            "    return total\n",
            "from candidate import total_with_discount as t\n"
            "assert t(100, [20, 10]) == 80.0\nassert t(100, [10]) == 90.0\n"
            "assert t(100, []) == 100\nprint('hidden OK')\n",
            "def total_with_discount(price, discounts):\n"
            "    if not discounts:\n        return price\n"
            "    return price * (1 - max(discounts) / 100)\n",
            "Discounts compete; they do not stack.",
        ),
    )


def _architectural_tasks() -> tuple[CorpusTask, ...]:
    return (
        _v2(
            "v2-lsp-square",
            "Independent dimensions",
            "lsp-violation",
            "architectural",
            "A Square accepts w=2 then h=3 but reports area 9 with dimensions (3, 3). "
            "Contract: width and height stay independent; area is w*h.",
            "class Rectangle:\n    def __init__(self):\n        self.w = 0\n        self.h = 0\n"
            "    def area(self):\n        return self.w * self.h\n"
            "class Square(Rectangle):\n"
            "    def __setattr__(self, name, value):\n"
            "        if name in ('w', 'h'):\n"
            "            object.__setattr__(self, 'w', value)\n"
            "            object.__setattr__(self, 'h', value)\n"
            "        else:\n"
            "            object.__setattr__(self, name, value)\n",
            "from candidate import Square\n"
            "s = Square()\ns.w = 2\ns.h = 3\n"
            "assert (s.w, s.h) == (2, 3)\nassert s.area() == 6\n"
            "print('hidden OK')\n",
            "class Rectangle:\n    def __init__(self):\n        self.w = 0\n        self.h = 0\n"
            "    def area(self):\n        return self.w * self.h\n"
            "class Square:\n    def __init__(self):\n        self.w = 0\n        self.h = 0\n"
            "    def area(self):\n        return self.w * self.h\n",
            "A square is not a mutable rectangle.",
        ),
        _v2(
            "v2-dup-capability",
            "One header parser",
            "duplicated-capability",
            "architectural",
            "parse_header('A: b:c') raises instead of returning {'A': 'b:c'}. "
            "Contract: one parser handles values containing colons; keep a single implementation.",
            "def parse_header_v1(line):\n    key, _, value = line.partition(':')\n    return {key.strip(): value.strip()}\n"
            "def parse_header_v2(line):\n    key, value = line.split(':')\n    return {key.strip(): value.strip()}\n"
            "parse_header = parse_header_v2\n",
            "from candidate import parse_header\n"
            "assert parse_header('A: b:c') == {'A': 'b:c'}\n"
            "assert parse_header('X: 1') == {'X': '1'}\n"
            "import candidate\n"
            "assert not hasattr(candidate, 'parse_header_v2'), 'second implementation must go'\n"
            "print('hidden OK')\n",
            "def parse_header(line):\n    key, _, value = line.partition(':')\n    return {key.strip(): value.strip()}\n",
            "Two implementations diverged; keep the general one.",
        ),
        _v2(
            "v2-ambient-precision",
            "Fixed precision",
            "ambient-global",
            "architectural",
            "invoice_total([0.125, 10]) changes from 10.12 to 10.125 when unrelated code sets "
            "PRECISION=4. Contract: invoices always use 2 decimals regardless of the global.",
            "PRECISION = 2\n"
            "def invoice_total(items):\n    return round(sum(items), PRECISION)\n",
            "import candidate\n"
            "candidate.PRECISION = 4\n"
            "assert candidate.invoice_total([0.125, 10]) == 10.12\n"
            "assert candidate.invoice_total([10]) == 10\n"
            "print('hidden OK')\n",
            "PRECISION = 2\n"
            "def invoice_total(items):\n    return round(sum(items), 2)\n",
            "Money formatting must not depend on ambient state.",
        ),
        _v2(
            "v2-resource-owner",
            "Closable cache file",
            "resource-ownership",
            "architectural",
            "FileCache opens its file but offers no way to close it. "
            "Contract: usable as a context manager that closes the file on exit.",
            "class FileCache:\n"
            "    def __init__(self, path):\n        self._f = open(path, 'w')\n"
            "    def write(self, data):\n        self._f.write(data)\n",
            "from candidate import FileCache\n"
            "with FileCache('out.txt') as cache:\n"
            "    cache.write('x')\n    handle = cache._f\n"
            "assert handle.closed, 'file must be closed on context exit'\n"
            "print('hidden OK')\n",
            "class FileCache:\n"
            "    def __init__(self, path):\n        self._path = path\n        self._f = None\n"
            "    def __enter__(self):\n        self._f = open(self._path, 'w')\n        return self\n"
            "    def __exit__(self, *exc):\n        self._f.close()\n        return False\n"
            "    def write(self, data):\n        self._f.write(data)\n",
            "Ownership of a resource includes its release.",
        ),
        _v2(
            "v2-retry-once-accepted",
            "No duplicate send",
            "retry-semantics",
            "architectural",
            "publish() delivers some messages twice when the transport drops an acknowledgement. "
            "Contract: each message has exactly one visible effect; the transport answers "
            "delivered(msg) so a lost ack can be distinguished from a lost send.",
            "def publish(transport, msg):\n"
            "    for _ in range(3):\n"
            "        try:\n            transport.send(msg)\n            return True\n"
            "        except Exception:\n            continue\n"
            "    return False\n",
            "from candidate import publish\n"
            "class FakeTransport:\n"
            "    def __init__(self):\n        self.sends = []\n"
            "    def send(self, msg):\n"
            "        self.sends.append(msg)\n"
            "        if len(self.sends) == 1:\n"
            "            raise TimeoutError('ack lost')\n"
            "    def delivered(self, msg):\n"
            "        return msg in self.sends\n"
            "t = FakeTransport()\n"
            "assert publish(t, 'm') is True\n"
            "assert t.sends == ['m'], 'exactly one visible effect'\n"
            "print('hidden OK')\n",
            "def publish(transport, msg):\n"
            "    try:\n        transport.send(msg)\n"
            "    except Exception:\n"
            "        if transport.delivered(msg):\n"
            "            return True\n"
            "        transport.send(msg)\n"
            "    return True\n",
            "A lost ack is not a lost send.",
        ),
        _v2(
            "v2-cart-total",
            "Live cart total",
            "derived-state",
            "architectural",
            "Cart().total stays 15 after removing the 10-valued item. "
            "Contract: total always reflects current contents.",
            "class Cart:\n"
            "    def __init__(self):\n        self.items = []\n        self.total = 0\n"
            "    def add(self, price):\n        self.items.append(price)\n        self.total += price\n"
            "    def remove(self, price):\n        self.items.remove(price)\n",
            "from candidate import Cart\n"
            "c = Cart()\nc.add(10)\nc.add(5)\nc.remove(10)\n"
            "assert c.total == 5\nassert c.items == [5]\nprint('hidden OK')\n",
            "class Cart:\n"
            "    def __init__(self):\n        self.items = []\n"
            "    @property\n"
            "    def total(self):\n        return sum(self.items)\n"
            "    def add(self, price):\n        self.items.append(price)\n"
            "    def remove(self, price):\n        self.items.remove(price)\n",
            "Derive, do not cache, what you can compute.",
        ),
        _v2(
            "v2-transfer-guard",
            "Validate before mutating",
            "boundary-validation",
            "architectural",
            "transfer() of -5 mutates both balances and only afterwards complains. "
            "Contract: non-positive amounts raise ValueError with balances untouched.",
            "def format_receipt(src, dst, amount):\n"
            "    if amount <= 0:\n        raise ValueError('bad amount')\n"
            "    return f'{src}->{dst}:{amount}'\n"
            "def transfer(ledger, src, dst, amount):\n"
            "    ledger[src] -= amount\n    ledger[dst] += amount\n"
            "    return format_receipt(src, dst, amount)\n",
            "from candidate import transfer\n"
            "ledger = {'a': 10, 'b': 0}\n"
            "try:\n    transfer(ledger, 'a', 'b', -5)\nexcept ValueError:\n    pass\n"
            "else:\n    raise AssertionError('expected ValueError')\n"
            "assert ledger == {'a': 10, 'b': 0}, 'balances must be untouched'\n"
            "assert transfer({'a': 10, 'b': 0}, 'a', 'b', 4) == 'a->b:4'\n"
            "print('hidden OK')\n",
            "def format_receipt(src, dst, amount):\n"
            "    return f'{src}->{dst}:{amount}'\n"
            "def transfer(ledger, src, dst, amount):\n"
            "    if amount <= 0:\n        raise ValueError('bad amount')\n"
            "    ledger[src] -= amount\n    ledger[dst] += amount\n"
            "    return format_receipt(src, dst, amount)\n",
            "Guards belong where mutation begins.",
        ),
        _v2(
            "v2-unsound-cache",
            "Cache key honesty",
            "unsound-cache",
            "architectural",
            "member([9], 7) reports True after member([7], 7) was True. "
            "Contract: membership of the actual list, every call.",
            "_cache = {}\n"
            "def member(items, x):\n"
            "    key = (len(items), x)\n"
            "    if key not in _cache:\n"
            "        _cache[key] = x in items\n"
            "    return _cache[key]\n",
            "from candidate import member\n"
            "assert member([7], 7) is True\n"
            "assert member([9], 7) is False\n"
            "assert member([7, 8], 8) is True\nprint('hidden OK')\n",
            "def member(items, x):\n    return x in items\n",
            "A cache keyed on the wrong identity is a wrong answer.",
        ),
        _v2(
            "v2-pure-analyze",
            "No side files",
            "side-effecting-pure",
            "architectural",
            "analyze() creates scratch.tmp in the working directory on every call. "
            "Contract: pure function of its input; no files created.",
            "def analyze(xs):\n"
            "    with open('scratch.tmp', 'w') as handle:\n"
            "        handle.write(','.join(map(str, xs)))\n"
            "    return sum(xs) / len(xs)\n",
            "from pathlib import Path\nfrom candidate import analyze\n"
            "assert analyze([2, 4]) == 3\n"
            "assert not Path('scratch.tmp').exists(), 'must not create files'\n"
            "print('hidden OK')\n",
            "def analyze(xs):\n    return sum(xs) / len(xs)\n",
            "Computation is not persistence.",
        ),
        _v2(
            "v2-nesting-invariant",
            "Proper nesting",
            "nesting-invariant",
            "architectural",
            "is_balanced('([)]') reports True. Counting brackets is not enough. "
            "Contract: every closer must match the most recent unclosed opener.",
            "def is_balanced(s):\n"
            "    return s.count('(') == s.count(')') and s.count('[') == s.count(']')\n",
            "from candidate import is_balanced\n"
            "assert is_balanced('([)]') is False\n"
            "assert is_balanced('()[]') is True\n"
            "assert is_balanced('([') is False\n"
            "assert is_balanced('a(b)c') is True\n"
            "assert is_balanced('') is True\nprint('hidden OK')\n",
            "def is_balanced(s):\n"
            "    pairs = {')': '(', ']': '['}\n"
            "    stack = []\n"
            "    for ch in s:\n"
            "        if ch in '([':\n"
            "            stack.append(ch)\n"
            "        elif ch in pairs:\n"
            "            if not stack or stack.pop() != pairs[ch]:\n"
            "                return False\n"
            "    return not stack\n",
            "Order is part of the contract.",
        ),
    )


def _boundary_tasks() -> tuple[CorpusTask, ...]:
    db_module = (
        "support_db.py",
        ("_open = []\n"
        "class Connection:\n"
        "    def __init__(self):\n        _open.append(self)\n        self.closed = False\n"
        "    def close(self):\n        self.closed = True\n"
        "        _open.remove(self)\n"
        "    def query(self, name):\n        return {'name': name}\n"
        "def connect():\n    return Connection()\n"
        "def open_count():\n    return len(_open)\n"),
    )
    prices_module = (
        "support_prices.py",
        ("_prices = {'apple': 1.0}\n"
        "def set_price(item, price):\n    _prices[item] = price\n"
        "def get(item):\n    return _prices[item]\n"),
    )
    registry_module = (
        "support_registry.py",
        ("_handlers = {}\n"
        "def register(name, fn):\n    _handlers[name] = fn\n"
        "def handlers():\n    return dict(_handlers)\n"),
    )
    api_module = (
        "support_api.py",
        "def fetch():\n    return {'user_name': 'ada', 'user_id': 7}\n",
    )
    feed_module = (
        "support_feed.py",
        ("_readings = [10, 20]\n"
        "def add(value):\n    _readings.append(value)\n"
        "def latest():\n    return _readings[-1]\n"),
    )
    broker_module = (
        "support_broker.py",
        ("_attempts = 0\n_delivered = []\n"
        "def publish(msg):\n"
        "    global _attempts\n"
        "    _attempts += 1\n"
        "    if _attempts <= 2:\n        raise ConnectionError('transient flake')\n"
        "    _delivered.append(msg)\n"
        "    return True\n"
        "def delivered():\n    return list(_delivered)\n"),
    )
    store_module = (
        "support_store.py",
        ("_entries = []\n"
        "def append(entry):\n    _entries.append(entry)\n"
        "def entries():\n    return list(_entries)\n"),
    )
    wire_module = (
        "support_wire.py",
        ("_buf = b''\n"
        "def send(data):\n"
        "    global _buf\n"
        "    if not isinstance(data, bytes):\n"
        "        raise TypeError('wire requires bytes')\n"
        "    _buf = data\n"
        "def recv():\n    return _buf\n"),
    )
    return (
        _v2(
            "v2-connection-owner",
            "Shared connection",
            "connection-ownership",
            "boundary",
            "get_user() grows db.open_count() on every call. "
            "Contract: repeated calls leave the open-connection count unchanged.",
            "import support_db as db\n"
            "def get_user(name):\n    conn = db.connect()\n    return conn.query(name)\n",
            "import support_db as db\nfrom candidate import get_user\n"
            "assert get_user('a') == {'name': 'a'}\n"
            "before = db.open_count()\nget_user('b')\nget_user('c')\n"
            "assert db.open_count() == before, 'connections must not leak per call'\n"
            "print('hidden OK')\n",
            "import support_db as db\n_conn = db.connect()\n"
            "def get_user(name):\n    return _conn.query(name)\n",
            "One owner, not one connection per call.",
            (db_module,),
        ),
        _v2(
            "v2-price-freshness",
            "Read-through prices",
            "read-through-cache",
            "boundary",
            "price('apple') still returns 1.0 after prices.set_price('apple', 2.0). "
            "Contract: always the current table value.",
            "import support_prices as prices\n"
            "_cached = {'apple': prices.get('apple')}\n"
            "def price(item):\n    return _cached[item]\n",
            "import support_prices as prices\nfrom candidate import price\n"
            "assert price('apple') == 1.0\n"
            "prices.set_price('apple', 2.0)\n"
            "assert price('apple') == 2.0\n"
            "print('hidden OK')\n",
            "import support_prices as prices\n"
            "def price(item):\n    return prices.get(item)\n",
            "Import time is not query time.",
            (prices_module,),
        ),
        _v2(
            "v2-keyed-cache-length",
            "Cache identity",
            "keyed-cache",
            "boundary",
            "lookup([9, 0]) returns [1, 3] after lookup([3, 1]). "
            "Contract: sorted copy of the given list, every call.",
            "_cache = {}\n"
            "def lookup(xs):\n"
            "    key = len(xs)\n"
            "    if key not in _cache:\n"
            "        _cache[key] = sorted(xs)\n"
            "    return _cache[key]\n",
            "from candidate import lookup\n"
            "assert lookup([3, 1]) == [1, 3]\n"
            "assert lookup([9, 0]) == [0, 9]\n"
            "assert lookup([2]) == [2]\nprint('hidden OK')\n",
            "def lookup(xs):\n    return sorted(xs)\n",
            "Length is not identity.",
            (),
        ),
        _v2(
            "v2-env-timing",
            "Live environment",
            "env-timing",
            "boundary",
            "mode() keeps returning 'safe' after APP_MODE is set to 'fast'. "
            "Contract: reflect the current environment on every call.",
            "import os\n_mode = os.getenv('APP_MODE', 'safe')\n"
            "def mode():\n    return _mode\n",
            "import os\nfrom candidate import mode\n"
            "os.environ['APP_MODE'] = 'fast'\n"
            "assert mode() == 'fast'\n"
            "del os.environ['APP_MODE']\n"
            "assert mode() == 'safe'\n"
            "print('hidden OK')\n",
            "import os\n"
            "def mode():\n    return os.getenv('APP_MODE', 'safe')\n",
            "Read configuration when needed, not when loaded.",
            (),
        ),
        _v2(
            "v2-registry-pollution",
            "No registry writes",
            "registry-pollution",
            "boundary",
            "After run('ping'), the shared registry contains a 'fallback' entry nobody registered. "
            "Contract: dispatch 'ping' to 'pong', unknown names to 'fallback', registry untouched.",
            "import support_registry as registry\n"
            "def run(name, *args):\n"
            "    registry.register('fallback', lambda *a: 'fallback')\n"
            "    table = registry.handlers()\n"
            "    return table.get(name, table['fallback'])(*args)\n",
            "import support_registry as registry\nfrom candidate import run\n"
            "assert run('ping') == 'pong'\n"
            "before = set(registry.handlers())\n"
            "assert run('unknown-thing') == 'fallback'\n"
            "assert set(registry.handlers()) == before, 'registry must be untouched'\n"
            "print('hidden OK')\n",
            "import support_registry as registry\n"
            "def _fallback(*args):\n    return 'fallback'\n"
            "def _ping(*args):\n    return 'pong'\n"
            "def run(name, *args):\n"
            "    table = dict(registry.handlers())\n"
            "    table.setdefault('ping', _ping)\n"
            "    return table.get(name, _fallback)(*args)\n",
            "Fallbacks live in the caller, not the shared registry.",
            (registry_module,),
        ),
        _v2(
            "v2-field-drift",
            "Current field names",
            "schema-drift",
            "boundary",
            "label() raises KeyError 'name' against the current api module. "
            "Contract: return 'ada#7' from the fields the api actually provides.",
            "import support_api as api\n"
            "def label():\n    record = api.fetch()\n    return f\"{record['name']}#{record['id']}\"\n",
            "from candidate import label\n"
            "assert label() == 'ada#7'\nprint('hidden OK')\n",
            "import support_api as api\n"
            "def label():\n"
            "    record = api.fetch()\n"
            "    name = record.get('user_name', record.get('name'))\n"
            "    ident = record.get('user_id', record.get('id'))\n"
            "    return f'{name}#{ident}'\n",
            "Read the schema you have, not the one you remember.",
            (api_module,),
        ),
        _v2(
            "v2-latest-reading",
            "Latest feed value",
            "freshness",
            "boundary",
            "current() still returns 10 after feed.add(30). "
            "Contract: the latest reading at call time.",
            "import support_feed as feed\n_first = feed.latest()\n"
            "def current():\n    return _first\n",
            "import support_feed as feed\nfrom candidate import current\n"
            "assert current() == 20\n"
            "feed.add(30)\n"
            "assert current() == 30\n"
            "print('hidden OK')\n",
            "import support_feed as feed\n"
            "def current():\n    return feed.latest()\n",
            "A sensor reading is not a constant.",
            (feed_module,),
        ),
        _v2(
            "v2-retry-delivery",
            "Ride out flakes",
            "transient-retry",
            "boundary",
            "send('m') raises on the first attempt although the broker only flakes transiently. "
            "Contract: return True once the message is through.",
            "import support_broker as broker\n"
            "def send(msg):\n    broker.publish(msg)\n    return True\n",
            "import support_broker as broker\nfrom candidate import send\n"
            "assert send('m') is True\n"
            "assert broker.delivered() == ['m']\n"
            "print('hidden OK')\n",
            "import support_broker as broker\n"
            "def send(msg):\n"
            "    for _ in range(5):\n"
            "        try:\n            broker.publish(msg)\n"
            "            return True\n"
            "        except ConnectionError:\n"
            "            continue\n"
            "    raise ConnectionError('broker unavailable')\n",
            "Transient means retryable.",
            (broker_module,),
        ),
        _v2(
            "v2-single-append",
            "One entry per record",
            "double-append",
            "boundary",
            "record('x') leaves two entries in the store. "
            "Contract: exactly one entry per call.",
            "import support_store as store\n"
            "def record(event):\n    store.append(event)\n    _flush(event)\n"
            "def _flush(event):\n    store.append(event + ':flushed')\n",
            "import support_store as store\nfrom candidate import record\n"
            "record('x')\n"
            "assert store.entries() == ['x']\n"
            "record('y')\n"
            "assert store.entries() == ['x', 'y']\n"
            "print('hidden OK')\n",
            "import support_store as store\n"
            "def record(event):\n    store.append(event)\n",
            "One effect per action.",
            (store_module,),
        ),
        _v2(
            "v2-utf8-wire",
            "Text over bytes",
            "wire-encoding",
            "boundary",
            "transmit('héllo') raises TypeError: the wire takes bytes. "
            "Contract: any text round-trips through the wire.",
            "import support_wire as wire\n"
            "def transmit(text):\n    wire.send(text)\n    return wire.recv()\n",
            "from candidate import transmit\n"
            "assert transmit('héllo') == 'héllo'\n"
            "assert transmit('plain') == 'plain'\n"
            "print('hidden OK')\n",
            "import support_wire as wire\n"
            "def transmit(text):\n    wire.send(text.encode('utf-8'))\n"
            "    return wire.recv().decode('utf-8')\n",
            "Encode at the boundary, both directions.",
            (wire_module,),
        ),
    )


def semantic_corpus() -> tuple[CorpusTask, ...]:
    return _local_tasks() + _semantic_tasks() + _architectural_tasks() + _boundary_tasks()


def get_v2_task(task_id: str) -> CorpusTask:
    for task in semantic_corpus():
        if task.task_id == task_id:
            return task
    raise KeyError(f"unknown v2 task: {task_id}")

