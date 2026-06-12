"""Batch-fetch git blobs over the smart HTTP protocol (v2) in a single request.

The Hub's git server (`https://huggingface.co/<repo>.git/git-upload-pack`) supports protocol v2
and accepts arbitrary blob oids in `want` lines. This lets us fetch the content of many regular
(non-LFS) files in ONE `POST` request instead of one `GET /resolve/...` per file.

Only what's needed for this use case is implemented: full-object and delta entries (OFS/REF)
are supported, blobs are verified against their git oid (sha1).
"""

import hashlib
import struct
import zlib

import httpx

from . import constants
from .utils import build_hf_headers, hf_raise_for_status, logging


logger = logging.get_logger(__name__)

OBJ_COMMIT, OBJ_TREE, OBJ_BLOB, OBJ_TAG, OBJ_OFS_DELTA, OBJ_REF_DELTA = 1, 2, 3, 4, 6, 7


def git_repo_url(repo_id: str, repo_type: str, endpoint: str | None = None) -> str:
    endpoint = endpoint if endpoint is not None else constants.ENDPOINT
    prefix = "" if repo_type == "model" else f"{repo_type}s/"
    return f"{endpoint.rstrip('/')}/{prefix}{repo_id}.git"


def _pkt_line(payload: str) -> bytes:
    data = payload.encode() + b"\n"
    return f"{len(data) + 4:04x}".encode() + data


def fetch_blobs(
    repo_id: str,
    repo_type: str,
    oids: list[str],
    *,
    endpoint: str | None = None,
    token: bool | str | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> dict[str, bytes]:
    """Fetch git blobs by oid in a single `git-upload-pack` request. Returns {oid: content}."""
    url = f"{git_repo_url(repo_id, repo_type, endpoint)}/git-upload-pack"
    body = b"".join(
        [_pkt_line("command=fetch"), b"0001", _pkt_line("no-progress")]
        + [_pkt_line(f"want {oid}") for oid in sorted(set(oids))]
        + [_pkt_line("done"), b"0000"]
    )
    hf_headers = build_hf_headers(token=token, headers=headers)
    hf_headers.update(
        {
            "Git-Protocol": "version=2",
            "Content-Type": "application/x-git-upload-pack-request",
            "Accept": "application/x-git-upload-pack-result",
        }
    )
    response = httpx.post(url, content=body, headers=hf_headers, timeout=timeout)
    hf_raise_for_status(response)
    pack = _extract_packfile(response.content)
    objects = _parse_pack(pack)
    result: dict[str, bytes] = {}
    for obj_type, data in objects:
        if obj_type != OBJ_BLOB:
            continue
        oid = hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()
        result[oid] = data
    missing = set(oids) - set(result)
    if missing:
        raise ValueError(f"git-upload-pack response is missing {len(missing)} requested blobs.")
    return result


def _extract_packfile(raw: bytes) -> bytes:
    """Demultiplex the pkt-line + side-band framed upload-pack response into the raw packfile."""
    pack = bytearray()
    pos = 0
    in_packfile_section = False
    while pos < len(raw):
        length_hex = raw[pos : pos + 4]
        pos += 4
        length = int(length_hex, 16)
        if length == 0:  # flush-pkt
            continue
        if length == 1:  # delim-pkt
            continue
        payload = raw[pos : pos + length - 4]
        pos += length - 4
        if not in_packfile_section:
            if payload.rstrip(b"\n") == b"packfile":
                in_packfile_section = True
            continue
        band, data = payload[0], payload[1:]
        if band == 1:
            pack.extend(data)
        elif band == 3:
            raise ValueError(f"git-upload-pack error: {data.decode(errors='replace').strip()}")
    if not pack.startswith(b"PACK"):
        raise ValueError("No packfile found in git-upload-pack response.")
    return bytes(pack)


def _parse_pack(pack: bytes) -> list[tuple[int, bytes]]:
    """Parse a packfile into a list of (object_type, content). Resolves OFS/REF deltas."""
    if pack[:4] != b"PACK":
        raise ValueError("Invalid packfile signature.")
    version, count = struct.unpack(">II", pack[4:12])
    if version not in (2, 3):
        raise ValueError(f"Unsupported packfile version: {version}")
    pos = 12
    by_offset: dict[int, tuple[int, bytes]] = {}
    by_sha: dict[str, tuple[int, bytes]] = {}
    objects: list[tuple[int, bytes]] = []
    for _ in range(count):
        obj_offset = pos
        byte = pack[pos]
        pos += 1
        obj_type = (byte >> 4) & 0x7
        size = byte & 0x0F
        shift = 4
        while byte & 0x80:
            byte = pack[pos]
            pos += 1
            size |= (byte & 0x7F) << shift
            shift += 7

        base: tuple[int, bytes] | None = None
        if obj_type == OBJ_OFS_DELTA:
            byte = pack[pos]
            pos += 1
            offset = byte & 0x7F
            while byte & 0x80:
                byte = pack[pos]
                pos += 1
                offset = ((offset + 1) << 7) | (byte & 0x7F)
            base = by_offset[obj_offset - offset]
        elif obj_type == OBJ_REF_DELTA:
            base_sha = pack[pos : pos + 20].hex()
            pos += 20
            base = by_sha[base_sha]

        decompressor = zlib.decompressobj()
        data = decompressor.decompress(pack[pos:])
        data += decompressor.flush()
        consumed = len(pack) - pos - len(decompressor.unused_data)
        pos += consumed

        if base is not None:
            obj_type, data = base[0], _apply_delta(base[1], data)
        if len(data) != size and base is None:
            raise ValueError("Packfile object size mismatch.")

        entry = (obj_type, data)
        by_offset[obj_offset] = entry
        header = {OBJ_COMMIT: b"commit", OBJ_TREE: b"tree", OBJ_BLOB: b"blob", OBJ_TAG: b"tag"}.get(obj_type)
        if header is not None:
            sha = hashlib.sha1(header + b" %d\x00" % len(data) + data).hexdigest()
            by_sha[sha] = entry
        objects.append(entry)
    return objects


def _apply_delta(base: bytes, delta: bytes) -> bytes:
    """Apply a git delta (copy/insert opcodes) to a base object."""
    pos = 0

    def read_varint() -> int:
        nonlocal pos
        value, shift = 0, 0
        while True:
            byte = delta[pos]
            pos += 1
            value |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                return value

    read_varint()  # base size (unused)
    result_size = read_varint()
    out = bytearray()
    while pos < len(delta):
        opcode = delta[pos]
        pos += 1
        if opcode & 0x80:  # copy from base
            copy_offset = copy_size = 0
            for i in range(4):
                if opcode & (1 << i):
                    copy_offset |= delta[pos] << (8 * i)
                    pos += 1
            for i in range(3):
                if opcode & (1 << (4 + i)):
                    copy_size |= delta[pos] << (8 * i)
                    pos += 1
            if copy_size == 0:
                copy_size = 0x10000
            out.extend(base[copy_offset : copy_offset + copy_size])
        elif opcode:  # insert literal
            out.extend(delta[pos : pos + opcode])
            pos += opcode
        else:
            raise ValueError("Invalid delta opcode 0.")
    if len(out) != result_size:
        raise ValueError("Delta application size mismatch.")
    return bytes(out)
