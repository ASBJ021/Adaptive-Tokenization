import os
import json
from typing import Any, Dict
from collections import defaultdict, Counter


def main():
    path = '/home/utn/firi22ka/Desktop/jenga/Adaptive-Tokenization/new_src/clane9/imagenet-100_100_1757930534.9602883.jsonl'
    records = []
    keep_pct = []
    num_img = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    records.append(record)
                    # print(f'{record =  }')
                except Exception:
                    print(Exception)
                    continue
    else:
        print('No path found ')

    # for item in record:
    #     if item['keep_pct'] in keep_pct:

    # by_keep_count = defaultdict(lambda: {'n': 0, 'keep_pct_sum': 0.0})
    # for r in records:
    #     # kc = r.get('keep_count')
    #     kp = r.get('keep_pct')
    #     # if kc is None:
    #     #     continue
    #     by_keep_count[kp]['n'] += 1
    #     # if isinstance(kp, (int, float)):
    #     #     by_keep_count[kc]['keep_pct_sum'] += float(kp)
    # print(by_keep_count)

    round_decimals = 2
    pct_counter = Counter(round(float(r['keep_pct']))
                          for r in records if 'keep_pct' in r and isinstance(r['keep_pct'], (int, float)))
    
    if pct_counter:
        pct_rows = sorted(pct_counter.items(), key=lambda x: x[0])
        print('\n=== Table: grouped by keep_pct (rounded) ===')
        print(f'{"keep_pct":>10} | {"num_images":>10}')
        print('-' * 27)
        for kp, n in pct_rows:
            print(f'{kp:>10.{round_decimals}f} | {n:>10}')


        
    print(len(records))

    

if __name__ == "__main__":
    main()