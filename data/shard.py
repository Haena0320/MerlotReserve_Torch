# Turn the pile into smaller files that we can then merge

import zstandard
import io
import orjson as json
import argparse
import os
from tqdm import tqdm
import time
from google.cloud import storage

parser = argparse.ArgumentParser(description='SCRAPE!')
parser.add_argument(
    '-fold',
    dest='fold',
    default=0,
    type=int,
    help='which fold we are on from 0 to 30'
)
parser.add_argument(
    '-num_folds',
    dest='num_folds',
    default=30,
    type=int,
    help='num folds'
)
parser.add_argument(
    '-num_folds_output',
    dest='num_folds_output',
    default=16410,
    type=int,
    help='Number of folds for outputting',
)

args = parser.parse_args()

multiplier = args.num_folds_output // args.num_folds   # 547
assert args.num_folds_output % args.num_folds == 0

in_fn = '/home/rowan/datasets3/thepile/the-eye.eu/public/AI/pile/train/{:02d}.jsonl.zst'.format(args.fold)

def loader():
    start = time.time()
    with open(in_fn, 'rb') as fh:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(fh, read_size=16384) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for j, line in enumerate(text_stream):
                X = json.loads(line)
                if j % 10000 == 0:
                    te = time.time()-start
                    print("Read {} in {:.3f}sec".format(j, te), flush=True)
                yield X
folder = '/home/rowan/datasets3/thepile/shards/'
assert os.path.exists(folder)

out_fns = [os.path.join(folder, 'fold{:05d}of{}.jsonl.zst'.format(args.fold * multiplier + i, args.num_folds_output)) for i in range(multiplier)]
out_compressors = []
for fn in out_fns:
    cctx = zstandard.ZstdCompressor(level=6, threads=8)
    fh = open(fn, 'wb')
    compressor = cctx.stream_writer(fh)
    out_compressors.append(compressor)

for i, x in enumerate(loader()):
    x_compressed = json.dumps(x)
    writer = out_compressors[i % len(out_compressors)]
    writer.write(x_compressed + '\n'.encode('utf-8'))

for writer in out_compressors:
    writer.flush(zstandard.FLUSH_FRAME)
    writer.close()

gclient = storage.Client()
bucket = gclient.get_bucket('rowanyoutube3')
for fn in out_fns:
    iblob = bucket.blob(f'thepile/' + fn.split('/')[-1])
    iblob.upload_from_filename(fn)

# Get missing files
# all_blobs = set([x.name.split('/')[-1] for x in gclient.list_blobs('rowanyoutube3', prefix='thepile/')])
# missing_blobs = [f'fold{i:05d}of16410.jsonl.zst' for i in range(16410)]
# missing_blobs = [x for x in missing_blobs if x not in all_blobs]
# for x in tqdm(missing_blobs):
#     iblob = bucket.blob(f'thepile/{x}')
#     iblob.upload_from_filename(os.path.join('/home/rowan/datasets3/thepile/shards/', x))
