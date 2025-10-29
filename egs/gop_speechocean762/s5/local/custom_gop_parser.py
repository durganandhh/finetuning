def parse_gop_txt(file_path, phone_int2sym=None):
    SKIP_PHONES = {"SIL", "SPN", "NSN"}
    gop_data = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            utt_id = parts[0]
            pairs = parts[1:]
            phone_scores = []
            i = 0
            while i < len(pairs):
                if pairs[i] == '[' and i + 3 < len(pairs) and pairs[i + 3] == ']':
                    phone_id = int(pairs[i + 1])
                    score = float(pairs[i + 2])
                    if phone_int2sym:
                        phone_label = phone_int2sym.get(phone_id, '').split('_')[0]
                        if phone_label in SKIP_PHONES:
                            i += 4
                            continue
                    phone_scores.append((phone_id, score))
                    i += 4
                elif pairs[i].startswith('[') and pairs[i+2].endswith(']'):
                    phone_id = int(pairs[i][1:])
                    score = float(pairs[i+1])
                    if phone_int2sym:
                        phone_label = phone_int2sym.get(phone_id, '').split('_')[0]
                        if phone_label in SKIP_PHONES:
                            i += 3
                            continue
                    phone_scores.append((phone_id, score))
                    i += 3
                else:
                    i += 1
            gop_data[utt_id] = phone_scores
    return gop_data
