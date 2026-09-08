"""Lossless byte-shuffled XZ blocks; standard-library-only offline restoration.

This never converts tensor dtypes. Restored safetensors are byte-for-byte the
original file. Small packed members avoid the single-member ZIP64 size issue.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import lzma
from pathlib import Path
import struct

MAGIC = b'DAVW77\x00\x01'
MAX_BLOCK = 4 * 1024**2
MAX_MEMBER = 2_000_000_000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(MAX_BLOCK), b''):
            digest.update(block)
    return digest.hexdigest()


def shuffle_bytes(data):
    end = len(data) // 4 * 4
    body = data[:end]
    return b''.join(body[i::4] for i in range(4)) + data[end:]


def unshuffle_bytes(data):
    end = len(data) // 4 * 4
    width = end // 4
    output = bytearray(len(data))
    for i in range(4):
        output[i:end:4] = data[i * width:(i + 1) * width]
    output[end:] = data[end:]
    return bytes(output)


def decode_block(payload, raw_size):
    if not 0 < raw_size <= MAX_BLOCK or len(payload) > 2 * MAX_BLOCK + 65536:
        raise ValueError('invalid block dimensions')
    decoder = lzma.LZMADecompressor(memlimit=128 * 1024**2)
    shuffled = decoder.decompress(payload, max_length=raw_size + 1)
    if len(shuffled) != raw_size or not decoder.eof or decoder.unused_data:
        raise ValueError('truncated, oversized, or trailing block data')
    return unshuffle_bytes(shuffled)


def _encode(block):
    payload = lzma.compress(shuffle_bytes(block), preset=6)
    if decode_block(payload, len(block)) != block:
        raise ValueError('lossless block round-trip failed')
    return len(block), payload, hashlib.sha256(block).hexdigest()


def pack_file(source, directory, *, workers=4, block_size=MAX_BLOCK, member_limit=MAX_MEMBER):
    source, directory = Path(source), Path(directory)
    if not 1 <= block_size <= MAX_BLOCK or not 1 <= workers <= 8:
        raise ValueError('invalid block size or worker count')
    if not 128 <= member_limit <= MAX_MEMBER:
        raise ValueError('invalid member limit')
    if source.stat().st_size <= 0:
        raise ValueError('empty input')
    directory.mkdir(parents=True, exist_ok=False)
    parts, part, offset, pending = [], None, 0, deque()
    digest = hashlib.sha256()
    source_stat = source.stat()
    initial_size = source_stat.st_size
    try:
        with source.open('rb') as stream, ThreadPoolExecutor(max_workers=workers) as executor:
            exhausted = False
            while pending or not exhausted:
                # Bound queued raw data; Executor.map would eagerly read a whole
                # multi-GB checkpoint on Python 3.11.
                while len(pending) < workers * 2 and not exhausted:
                    block = stream.read(block_size)
                    if block:
                        digest.update(block)
                        pending.append(executor.submit(_encode, block))
                    else:
                        exhausted = True
                if not pending:
                    break
                raw_size, payload, block_hash = pending.popleft().result()
                record = struct.pack('<II', raw_size, len(payload)) + payload
                if len(MAGIC) + len(record) > member_limit:
                    raise ValueError('one encoded block exceeds member limit')
                if part is None or part.tell() + len(record) > member_limit:
                    if part is not None:
                        part.close()
                        parts[-1]['sha256'] = sha256(directory / parts[-1]['name'])
                    name = f'weights-{len(parts):05d}.dw77'
                    part = (directory / name).open('xb')
                    part.write(MAGIC)
                    parts.append(dict(name=name, raw_offset=offset, raw_bytes=0, blocks=0))
                part.write(record)
                parts[-1]['raw_bytes'] += raw_size
                parts[-1]['blocks'] += 1
                offset += raw_size
                if offset % (256 * 1024**2) < block_size:
                    print(json.dumps(dict(source=source.name, raw_bytes=offset, total=initial_size)), flush=True)
        if part is not None:
            part.close()
            parts[-1]['sha256'] = sha256(directory / parts[-1]['name'])
        if offset != initial_size or digest.hexdigest() != sha256(source):
            raise ValueError('source changed during packing')
        for record in parts:
            record['packed_bytes'] = (directory / record['name']).stat().st_size
        manifest = dict(format='davian_lossless_weights_v77', source_name=source.name,
            raw_bytes=offset, raw_sha256=digest.hexdigest(), parts=parts,
            packed_bytes=sum(p['packed_bytes'] for p in parts),
            tensor_dtype_changes=False, all_blocks_roundtrip_exact=True)
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        return manifest
    finally:
        if part is not None:
            part.close()


def restore_file(directory, destination):
    """Refuse overwrite; on failure preserve partial output for diagnosis."""
    directory, destination = Path(directory), Path(destination)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('format') != 'davian_lossless_weights_v77' or not manifest.get('parts'):
        raise ValueError('invalid weight manifest')
    digest, total, names = hashlib.sha256(), 0, set()
    with destination.open('xb') as output:
        for record in manifest['parts']:
            name = record['name']
            if Path(name).name != name or name in names or not name.endswith('.dw77'):
                raise ValueError('unsafe/duplicate weight part')
            names.add(name)
            path = directory / name
            if (path.stat().st_size != record['packed_bytes'] or path.stat().st_size > MAX_MEMBER
                    or sha256(path) != record['sha256'] or record['raw_offset'] != total):
                raise ValueError('weight part hash, size, or order mismatch')
            blocks, count = 0, 0
            with path.open('rb') as stream:
                if stream.read(len(MAGIC)) != MAGIC:
                    raise ValueError('invalid weight part magic')
                while header := stream.read(8):
                    if len(header) != 8:
                        raise ValueError('truncated block header')
                    raw_size, compressed_size = struct.unpack('<II', header)
                    if not 0 < compressed_size <= 2 * MAX_BLOCK + 65536:
                        raise ValueError('invalid compressed size')
                    payload = stream.read(compressed_size)
                    if len(payload) != compressed_size:
                        raise ValueError('truncated block')
                    block = decode_block(payload, raw_size)
                    output.write(block)
                    digest.update(block)
                    blocks += 1
                    count += len(block)
            if blocks != record['blocks'] or count != record['raw_bytes']:
                raise ValueError('block count/size mismatch')
            total += count
    if total != manifest['raw_bytes'] or digest.hexdigest() != manifest['raw_sha256']:
        raise ValueError('restored weights failed whole-file validation')
    return dict(restored_bytes=total, restored_sha256=digest.hexdigest())
