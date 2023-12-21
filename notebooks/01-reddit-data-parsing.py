import zstandard
import json
import os
from glob import glob
from tqdm.auto import tqdm
from datetime import datetime
from p_tqdm import p_map


def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def read_lines_zst(file_name):
    with open(file_name, "rb") as file_handle:
        buffer = ""
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
        while True:
            chunk = read_and_decode(reader, 2**27, (2**29) * 2)
            if not chunk:
                break
            lines = (buffer + chunk).split("\n")

            for line in lines[:-1]:
                yield line, file_handle.tell()

            buffer = lines[-1]
        reader.close()


def process_file(f):
    file_size = os.stat(f).st_size
    id_comments = []
    bad_lines = 0
    file_lines = 0
    for line, file_bytes_processed in read_lines_zst(f):
        try:
            obj = json.loads(line)
            body = obj["body"]
            subreddit = obj["subreddit"]
            created = datetime.utcfromtimestamp(int(obj["created_utc"]))

            if subreddit == "indonesia":
                id_comments.append(obj)
        except json.JSONDecodeError as err:
            bad_lines += 1

        file_lines += 1
        if file_lines % 1_000_000 == 0:
            print(
                f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {file_lines:,} : {len(id_comments):,} : {file_bytes_processed:,}:{(file_bytes_processed / file_size) * 100:.0f}%"
            )
    
    return id_comments

files = sorted(glob("/mnt/block-volume/root/reddit/comments/RC_202*.zst"))
results = p_map(process_file, files)
id_comments = [comment for result in results for comment in result]

with open("/mnt/block-volume/root/reddit_comments_subreddit_indonesia_RC_2020-01-2023-09.json", "w") as f:
    json.dump(id_comments, f)
